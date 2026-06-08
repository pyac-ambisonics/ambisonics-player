from pathlib import Path
import pyfar as pf
import pooch
import numpy as np

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
        # set the path for this HRTF
        try:
            self.path = Path(path)
        except Exception as e:
            print(f"Couldn't parse path: {path}. {e}")
            self.path = None

        # load HRTF from files
        self.hrirs, self.sources = self.load_HRTF()
        self.hrirs_linear = self.hrirs.copy()

        # make a list of all subdirectories of our Headphone filters
        self.resources = Path("resources")
        self.hp_dir = self.resources / "Headphones"
        hp_subdir = [x for x in self.hp_dir.iterdir() if x.is_dir()]
        self.hp_list = [x.name for x in hp_subdir]

    def load_HRTF(self):
        """
        Load HRTF data from a SOFA file or fall back to a web download.

        Returns
        -------
        tuple
            A tuple ``(hrirs, sources)`` where `hrirs` is a :class:`pyfar.Signal`
            containing the HRIRs and `sources` is an ndarray with source
            coordinates. If loading fails, both values may be `None`.
        """
        try:
            # load HRIRs and source positions
            hrirs, sources, _ = pf.io.read_sofa(self.path)
            print("Loaded HRTF from file")
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
        hrirs : pyfar.Signal
            Loaded HRIR signals.
        sources : ndarray
            Source coordinate array associated with `hrirs`.
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
    def load_hp_filter(self, name):
        path = self.hp_dir / name
        # load HRIRs and source positions
        try: 
            # this somehow always fails because of some bullshit with sofa conventions.
            # thats why we load wavs instead
            hp_filter, *_ = pf.io.read_sofa(path / "HpIRs.sofa")
            print(f"Loaded Headphone Filter {name} from sofa")
        except Exception as e:
            print(e)
            hp_filter = pf.io.read_audio(path / "HpFilter.wav")
            print(f"Loaded Headphone Filter {name} from wav")
        
        # apply headphone filter to hrirs
        self.hrirs = pf.dsp.convolve(self.hrirs, hp_filter, mdoe='full')
        return hp_filter
    
    # sets hrirs to hrirs_linear
    def reset_hrirs(self):
        self.hrirs = self.hrirs_linear.copy()
    
