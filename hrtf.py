from pathlib import Path
import pyfar as pf
import numpy as np
import pooch
import spharpy
import scipy.signal as sgn

try:
    import shroom.utils.math_utils as sh_util
except ModuleNotFoundError:
    sh_util = None

class HRTF:
    """
    Helper for loading Head-Related Transfer Function (HRTF) data.

    This class attempts to load HRIR/HRTF data from a local SOFA file given
    by `path`. If loading from the local file fails, it will attempt to
    download a default HRTF from the web.

    Attributes
    ----------
    path : pathlib.Path or None
        Path to the SOFA file used to load the HRTF. `None` if the provided
        path could not be parsed.
    hrirs : pyfar.Signal
        Loaded HRIR signals
    sources : ndarray
        Source coordinate array associated with `hrirs`.
    """

    def __init__(self, path=None):
        """
        Initialize an `HRTF` instance and load HRTF data. Initialize a list of available headphone
        filters. 

        Parameters
        ----------
        path : str or pathlib.Path
            Path or path-like object pointing to a SOFA file to load.

        Notes
        -----
        The constructor will attempt to load the HRTF from the provided
        `path`. On failure it will try to download a default HRTF from the
        internet using `load_hrtf_from_web()`.
        """
        self.app_dir = Path(__file__).resolve().parent
        self.path = self._resolve_path(path)

        # load HRTF from files
        self.hrirs, self.sources = self.load_HRTF()
        self.hrirs_linear = self.hrirs.copy()

        # make a list of all subdirectories of our Headphone filters
        self.resources = self.app_dir / "resources"
        self.hp_dir = self.resources / "Headphones"
        hp_subdir = [x for x in self.hp_dir.iterdir() if x.is_dir()] if self.hp_dir.exists() else []
        self.hp_list = [x.name for x in hp_subdir]
        self.hp_list.append("Diffuse Field Equalization")

    def _resolve_path(self, path):
        if path is None:
            return None

        try:
            candidate = Path(path)
        except TypeError as error:
            print(f"Couldn't parse path: {path}. {error}")
            return None

        if candidate.is_absolute():
            return candidate

        local_candidate = self.app_dir / candidate
        if local_candidate.exists():
            return local_candidate

        return candidate

    def get_IR_length(self):
        """
        Return the number of samples in the loaded HRIRs.
        
        Returns
        -------------
        int
            Number of samples in self.hrirs.
        """
        # returns the length of the HRTF impulse response
        return self.hrirs.n_samples

    def load_HRTF(self):
        """
        Load HRTF data from a SOFA file or fall back to a web download.

        Returns
        -------
        tuple
            A tuple ``(hrirs, sources)`` where `hrirs` is a :class:`pyfar.Signal`
            containing the HRIRs and `sources` is an ndarray with source
            coordinates. If loading fails, both values may be `None`.
            
        Notes
        ----------
        If local loading fails, load_hrtf_from_web() is invoked as a fallback.
        """
        if self.path is not None:
            # load HRIRs and source positions
            try:
                hrirs, sources, _ = pf.io.read_sofa(self.path)
                print(f"Loaded HRTF from file: {self.path}")
                return hrirs, sources
            except Exception as e:
                print(f"Couldn't load the HRTF from file. Exception: {e}\nAttempting to load from web instead. ")

        # try to load HRTF from web
        return self.load_hrtf_from_web()

    # loads HRTFs from the internet
    def load_hrtf_from_web(self):
        """
        Download and load the FABIAN HRTF SOFA file.

        Returns
        -------
        tuple
            A tuple ``(hrirs, sources)`` where `hrirs` is a :class:`pyfar.Signal`
            containing the HRIRs and `sources` is an ndarray with source
            coordinates. If loading fails, both values may be `None`.
        """

        # Leave this as it is: This is the URL from which the data will be downloaded
        # and a hash for checking if the download worked.
        url = 'https://github.com/pyfar/files/raw/refs/heads/main/education/VAR_TUB/FABIAN_HRIR_measured_HATO_0.sofa?download='
        hash = '83ebbcd9a09d17679b95d201c9775438c0bb1199d565c3fc7a25448a905cdc3c'

        file = pooch.retrieve(
            url, hash, fname='FABIAN_HRIR_measured_HATO_0.sofa', path=None)

        # load HRIRs and source positions
        hrirs, sources, _ = pf.io.read_sofa(file)
        print("Loaded HRTF from web")
        return hrirs, sources
    
    # loads a specific headphone filter given by the path
    # applies it to the hrir loaded
    def load_hp_filter(self, name, min_phase=True, n_samples=512):
        """
        Load a headphone compensation filter by name and apply it to current HRIRs.
        
        Parameters
        -------------
        name : str
            Subdirectory name under Headphones identifying the headphone.
        min_phase : bool, optional
            A boolean telling the function to use minimum phase conversion of the headphone filter. If true, 
            the filter will be converted to minimum phase, resulting in a way shorter filter signal, and making
            real time computation possible. By default minimum phase conversion is turned on.
        n_samples : int, optional
            An integer indicating the desired length of the resulting min HRIR

        Returns
        -------------
        pyfar.Signal
            Loaded headphone filter signal.

        Notes
        -------------------
        Attempts to read a SOFA filter first and falls back to loading a WAV filter if needed.
        The loaded filter is convolved with self.hrirs before returning.
        """

        if name in (None, "", "None"):
            self.reset_hrirs()
            return None
        
        # do Diffuse Field Equalization as default
        if name == "Diffuse Field Equalization":
            return self.apply_dfe()

        path = self.hp_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Headphone filter folder not found: {path}")

        # load headphoen filter as WAV
        hp_filter = pf.io.read_audio(path / "HpFilter.wav")
        if isinstance(hp_filter, tuple):
            hp_filter = hp_filter[0]
        print(f"Loaded Headphone Filter {name} from wav")

        fs = self.hrirs_linear.sampling_rate
        if hp_filter.sampling_rate != fs:
            hp_filter = pf.dsp.resample(
                hp_filter,
                self.hrirs_linear.sampling_rate,
                match_amplitude="freq",
            )
        
        # apply minimum phase conversion
        if min_phase:
            desired_length = n_samples - self.get_IR_length()
            # apply it before convolution to the hp_filter
            hp_filter = pf.dsp.minimum_phase(hp_filter)

            # apply a kaiser window to our filter, makin the filter 'n_samples' long
            hp_filter = pf.dsp.time_window(hp_filter, 
                                            (0, desired_length-1), 
                                            window=('kaiser', 8), 
                                            shape='right', 
                                            crop='window')
            
        # apply headphone filter to hrirs
        self.hrirs = pf.dsp.convolve(self.hrirs_linear, hp_filter, mode='full')
        print(f"Samplelength of HRIR: {self.get_IR_length()}")
        return hp_filter
    
    # sets hrirs to hrirs_linear
    def reset_hrirs(self):
        """
        Restore the original unmodified HRIRs saved at initialization.
        This resets self.hrirs to the copy stored in self.hrirs_linear.
        """
        self.hrirs = self.hrirs_linear.copy()

    def apply_dfe(self):
        # averaging each HRTF with the average of all HRTF
        average = pf.dsp.average(self.hrirs_linear, mode='power',caxis=0)
        # Inversion
        regularized = pf.dsp.RegularizedSpectrumInversion.from_frequency_range(
            average, [50, 16e3], beta='max')
        inverted = regularized.invert
        # minimum phase
        min_phase_dfe = pf.dsp.minimum_phase(inverted, truncate=False)

        # convolve hrirs with the dfe filter
        self.hrirs = pf.dsp.convolve(self.hrirs_linear, min_phase_dfe, mode='full')

        print(f"Samplelength of HRIR: {self.get_IR_length()}")
        return min_phase_dfe

    
# a class making different HRTF preprocessing algorithms available
class Processing:
    
    def __init__(self):
        """
        Create an HRTF preprocessing controller with available algorithms and defaults.
        
        Attributes
        -------------
        algorithms : list
            Supported algorithm names (e.g., 'LS', 'MagLS', ...).
        current_algorithm : str
            Name of the currently selected algorithm.
            Default FFT bin count used internally.
        __gain : dict
            Algorithm-specific pre-gain factors.
        """

        # the different algorithms available
        self.algorithms = ['LS', 'MagLS', 'TA', 'BiMagLS']
        # the current algorithm
        self.current_algorithm = 'LS'
        # for each algorithm, a different pre-gain might be necessary.
        self._gain = {
            'LS': 1., 
            'MagLS': 1.28, 
            'TA': 1., 
            'BiMagLS': 1.28
        }

    def get_gain(self):
        """
        Return the pre-gain factor for the currently selected preprocessing algorithm.
        
        Returns
        ----------
        float
            Algorithm-specific pre-gain multiplier.
        """

        return self._gain[self.current_algorithm]

    # apply the chosen preprocessing algorithm. use MagLS as default
    def apply_preprocessing(self, hrirs, sh, algorithm='LS'):
        """
        Apply the selected preprocessing algorithm to HRIRs and return SH-domain data.
        
        Parameters
        -----------
        hrirs : pyfar.Signal
            Time-domain HRIRs.
        sh : spharpy.SphericalHarmonics
            Spherical-harmonic providing the basis matrix.
        algorithm : str, optional
            Algorithm identifier ('LS', 'MagLS', 'TA', 'BiMagLS'). Default 'LS'.

        Returns
        ------------
        pyfar.Signal
            Spherical-harmonic coefficients.
        """

        match algorithm:
            case 'LS':
                print("Using standard spherical harmonics processing (Least Squares): Simple matrix multiplication.")
                self.current_algorithm = algorithm
                return self.__ls(hrirs, sh)
            case 'MagLS':
                if not self._has_magls_backend():
                    print("MagLS backend shroom.utils is not available. Falling back to LS preprocessing.")
                    self.current_algorithm = 'LS'
                    return self.__ls(hrirs, sh)
                print("Using MagLS HRTF Preprocessing")
                self.current_algorithm = algorithm
                return self.__mag_ls(hrirs, sh)
            case 'TA':
                print("TA not implemented yet. Falling back to MagLS/LS preprocessing.")
                return self.apply_preprocessing(hrirs, sh, algorithm='MagLS')
            case 'BiMagLS':
                print("BiMagLS not implemented yet. Falling back to MagLS/LS preprocessing.")
                return self.apply_preprocessing(hrirs, sh, algorithm='MagLS')
            case _:
                print("Unknown preprocessing algorithm. Using LS preprocessing.")
                self.current_algorithm = 'LS'
                return self.__ls(hrirs, sh)

    def _has_magls_backend(self):
        return sh_util is not None and hasattr(sh_util, "magls")

    # solves the Least Squares Problem. This means just applying the spherical Harmonics to the HRTF
    def __ls(self, hrirs: pf.Signal, sh: spharpy.SphericalHarmonics):
        """
        Compute spherical-harmonic coefficients via least-squares (direct matrix multiplication).

        Parameters
        -----------
        hrirs : pyfar.Signal
            Time-domain HRIR data.
        sh : spharpy.SphericalHarmonics
            SH object containing the basis matrix.

        Returns
        pyfar.Signal
        Result of sh.basis_inv @ hrirs, transposed to the expected shape.
        """

        return (sh.basis_inv @ hrirs).T

    # apply the magnitude least squares algorithm for better results for low order ambisonics
    # ramp = 0 -> no ramp. ramp = 1 -> default ramp (cutoff * (1/sqrt(2))). ramp > 1 -> specific ramp in freq
    def __mag_ls(self, hrirs: pf.Signal, sh: spharpy.SphericalHarmonics, cutoff=3000, ramp=1):
        """
        Compute Magnitude-Least-Squares spherical-harmonic coefficients with optional ramping.
        
        Parameters
        -------------
        hrirs : pyfar.Signal
            HRIR data (time or freq representation used internally).
        sh : spharpy.SphericalHarmonics
            Spherical-harmonic providing basis matrix.
        cutoff : float, optional
            Frequency (Hz) above which magnitude-only optimization is applied.
        ramp : float or bool, optional
            If > 0, apply a smooth ramp before cutoff frequency; if 1 use a 
            default width of cutoff * (1 / sqrt(2)).

        Returns
        -----------
        pyfar.Signal
            pyfar Signal containing spherical-harmonic coefficients converted back to time domain.

        Notes
        ------------
        This function performs per-frequency optimization and is computationally heavier than LS.
        """
        # make our data frequency data
        hrirs_freq = hrirs.freq_raw.copy()
        # find corresponding frequency bin
        freq_axis = hrirs.frequencies
        stop = hrirs.find_nearest_frequency(cutoff)

        # create alpha value for each frequency: possibly with a ramp up? or just 0/1?
        # with just 0/1 we might have to smooth the phase later
        alpha = np.zeros_like(freq_axis, dtype=float)
        alpha[stop:] = 1
        start = stop

        # make a ramp up if needed. Use Hanning for smoothness
        if ramp > 0:
            # use default ramp of f * (1/sqrt(2))
            if ramp == 1:
                ramp = cutoff * (1 / np.sqrt(2))
            # find start and stop indices. find_nearest_freq should always return valid indices
            start = hrirs.find_nearest_frequency(cutoff - ramp)
            stop = hrirs.find_nearest_frequency(cutoff)
            length = max(stop - start, 0)
            # create a Hanning window, use the half that goes from 0 -> 1
            win = np.hanning(2*length)[:length]
            alpha[start:start+length] = win

        # below cutoff: do simple least squares
        # do regular  least squares -> simple matrix mult
        hrirs_sh = (sh.basis_inv @ hrirs).T.freq_raw
        nm_magls = hrirs_sh.copy()
        sh_basis = sh.basis.copy()

        # for each ear, for each frequency, starting at the cutoff bin
        for ear in range(2):
            for f in range(start, freq_axis.size):
                # for each ear, all directional data, and the current frequency: find magls
                # solve the magnitude only optimization
                # above cutoff: do least squares, magnitude from original, phase from min-phase
                nm_magls[ear, :, f] = (
                    # use shroom library for speed
                    # only returns real numbers????
                    alpha[f] * sh_util.magls(A=sh_basis, 
                                             b=hrirs_freq[:, ear, f], 
                                             x_prev=nm_magls[ear, :, f-1]
                                             )
                    # our own solution
                    # alpha[f] * self.__mag_ls_solver(Y=sh_basis, 
                    #                                 target=hrirs_freq[:, ear, f], 
                    #                                 x_prev=nm_magls[ear, :, f-1]
                    #                                 )
                    + (1 - alpha[f]) * hrirs_sh[ear, :, f]
                )
        
        # create time domain by creating a pyfar Signal from frequency data
        hrirs_sh_time = pf.Signal(nm_magls, hrirs.sampling_rate, n_samples=hrirs.n_samples, domain='freq')
        return hrirs_sh_time
