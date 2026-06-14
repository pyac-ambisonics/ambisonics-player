# a class that handles the conversion of HRTF data into spherical harmonics

import numpy as np
import pyfar as pf
import spharpy as sh
from scipy import signal as sgn
from hrtf import HRTF, Processing

class SphericalHarmonics:
    """
    Convert HRTFs into spherical-harmonic domain and apply rotations.

    This class uses an HRTF dataset (SOFA), converts HRIRs to
    spherical-harmonic coefficients for a given Ambisonic order, and provides
    rotation + interpolation utilities and a fast HRTF application method.

    Parameters
    ----------
    hrtf : Tuple or None.
        Tuple containing (pyfar.Signal, array_like) containing the signal and sources of
        the HRTF dataset, or None. If None, the FABIAN dataset is downloaded.
    sampling_rate : int
        Target sampling rate in Hz for HRIR resampling (default 48000).
    ambi_order : int
        Ambisonic order to use for spherical-harmonic decomposition.

    Attributes
    ----------
    hrirs : HRTF
        Loaded HRIRs (time-domain).
    sources : array_like
        Source coordinates corresponding to `hrirs`.
    spherical_harmonics : spharpy.SphericalHarmonics
        Computed spherical-harmonic basis and inverse.
    hrirs_nm : spharpy.SphericalHarmonicSignal
        HRIRs expressed in spherical-harmonic domain.
    rotation : spharpy.transforms.SphericalHarmonicRotation
        Current rotation transform (can be updated with `set_rotation`).
    """

    # Constructor
    def __init__(self, hrtf: HRTF = None, sampling_rate=48e3, ambi_order=1):
        """Initialize spherical-harmonic HRTF processing.

        Parameters
        -----------
        hrtf : HRTF or None, optional
            Preloaded HRTF instance. If None, a default dataset is loaded.
        sampling_rate : int or float, optional
            Target sampling rate in Hz for HRIR resampling. Default is 48000.
        ambi_order : int, optional
            Ambisonic order used for the spherical-harmonic decomposition.
        """
        # # set the hrtf. this should be a pyfar signal!
        # self.hrtf = hrtf

        # set sample rate. we should make sure this is the same as for the HRTFs!!!
        self.sampling_rate = sampling_rate

        if hrtf == None:
            # load FABIAN from the web if no HRTF was given
            self.hrtf = HRTF()
        else:
            self.hrtf = hrtf

        # make sure the sampling rate is correct and resample if necessary
        if self.hrtf.hrirs.sampling_rate != sampling_rate:
            self.hrtf.hrirs = pf.dsp.resample(self.hrtf.hrirs, 
                                              sampling_rate=sampling_rate, 
                                              match_amplitude='freq'
                                              )

        # store the sources in a Sampling Sphere
        self.sources = sh.SamplingSphere.from_coordinates(hrtf.sources)

        # create a spherical harmonics definition, corresponding to the AmbiX convention
        self.ambi_order = ambi_order
        self.sh_definition = sh.SphericalHarmonicDefinition(self.ambi_order, 
                                                            normalization="SN3D", 
                                                            basis_type='real', 
                                                            condon_shortley=False
                                                            )

        # create the spherical harmonics object from definition and sampling sphere
        self.spherical_harmonics = sh.SphericalHarmonics.from_definition(self.sh_definition, 
                                                                         self.sources, 
                                                                         inverse_method="pseudo_inverse"
                                                                         )

        # create hrtf processing unit
        self.process = Processing()

        # create h_nm matrix 
        print("applying preprocessing to hrtf to get hrirs_nm")
        hrirs_nm = self.process.apply_preprocessing(self.hrtf.hrirs, 
                                               self.spherical_harmonics,
                                               algorithm='MagLS'
                                               )
        print("convert to spherical harmonic signal")
        self.hrirs_nm = sh.SphericalHarmonicSignal.from_definition(self.sh_definition, 
                                                                   hrirs_nm.time, 
                                                                   hrirs_nm.sampling_rate
                                                                   ).time
        
        # make a copy as "base" hrirs_nm
        self.hrirs_nm_base = self.hrirs_nm.copy()
        # prepare an N3D object for rotations
        self.hrirs_nm_base_n3d = sh.spherical.renormalize(self.hrirs_nm_base, 
                                                          channel_convention='ACN',
                                                          current_norm='SN3D',
                                                          target_norm='N3D',
                                                          axis=1)
        
        *_, self.pad_to_length = self.hrirs_nm.shape

        # prepare a rotation matrix, with all angles 0 currently
        angles = [0, 0, 0]
        self.set_rotation(angles)
        # self.rotation = sh.transforms.SphericalHarmonicRotation.from_euler('xyz', np.deg2rad(angles))
        # # calculate the rotation matrix once
        # self.rotation_matrix = self.rotation.as_spherical_harmonic_matrix(self.sh_definition)

        # prepare pre_gain for gianstaging
        self.pre_gain = self.find_gain()

    def get_IR_length(self):
        """
        Return the impulse-response length (number of samples) of the current HRIRs.

        Returns
        -----------
        int
            Number of samples in the HRIR time-domain representation.
        """
        *_, n_samples = self.hrirs_nm.shape
        return n_samples
    
    # set the current rotation angle
    def set_rotation(self, angles):
        """
        Set the current spherical-harmonic rotation from Euler angles and update internal state.
        Updates the internal rotation matrix, applies the rotation to the base SH coefficients and
        refreshes any FFT buffers used for fast convolution.

        Parameters
        ----------
        angles : sequence of float
            Euler angles in degrees as (alpha, beta, gamma).
        """
        self.rotation_matrix =sh.transforms.wigner_d_rotation(self.ambi_order, *angles)
        self.rotation_matrix = self.rotation_matrix.astype(np.float32)

        # rotation only works in N3D normalization
        hrirs_nm_n3d = self.rotation_matrix @ self.hrirs_nm_base_n3d
        
        # convert back to SN3D normalization
        self.hrirs_nm = sh.spherical.renormalize(hrirs_nm_n3d, 
                                                 channel_convention='ACN',
                                                 current_norm='N3D',
                                                 target_norm='SN3D',
                                                 axis=1)
        
        # update the fft
        self.update_hrirs_fft(self.pad_to_length)
        #self.hrirs_nm = self.rotation.apply(self.hrirs_nm_base)

    
    # apply the hrtf data to an ambisonics file
    def apply_hrtf(self, ambi_signal: pf.Signal, gain=1.):
        """
        Convolve an ambisonic signal with the rotated HRTFs to produce a stereo signal.

        Parameters
        ----------
        ambi_signal : pyfar.Signal
            Ambisonic input signal (should have cshape (n_channels,)).
        gain : float, optional
            A float between 0. and 1., applied after internal gain-staging, Default is 1.

        Returns
        -------
        numpy.ndarray
            Stereo time-domain array with shape (2, n_samples) containing left and right channels.

        Raises
        -----------
        AttributeError
            If gain is outside [0., 1.].
        ValueError
            If Ambisonic channel count mismatches the HRTF channels.
        """

        if gain > 1. or gain < 0:
            raise AttributeError("The gain must be in range [0., 1.].")

        # apply rotation to the sh_hrir
        sh_hrir = self.hrirs_nm.copy()

        # Check channel count by comparing the channel shape
        # we know the channel shape for ambi_signal is (channels,)
        ambi_ch, *_ = ambi_signal.cshape
        # we know the channel shape for sh_hrir is (2, channels)
        _, sh_hrir_ch, _ = sh_hrir.shape
        if ambi_ch != sh_hrir_ch:
            raise ValueError("Channel counts must match (16 for 3rd order).")

        # create our time signals
        left = sh_hrir[0, :, :]
        right = sh_hrir[1, :, :]

        # sh_hrir should have the shape (2, ambi_order)
        # Convolve each channel separately for left and right
        # instantly store it as time data
        left_conv = pf.dsp.convolve(
            ambi_signal,
            pf.Signal(left, 
                      self.sampling_rate, 
                      domain='time'
                      ), # need to get the left channel here
            mode='full',
            method='overlap_add'
        ).time
        right_conv = pf.dsp.convolve(
            ambi_signal,
            pf.Signal(right, 
                      self.sampling_rate, 
                      domain='time'
                      ), # need to get the right channel here
            mode='full',
            method='overlap_add'
        ).time

        # Sum over channels -> single‑channel binaural signals
        left_signal = np.sum(left_conv, axis=0)
        right_signal = np.sum(right_conv, axis=0)

        # do gain staging
        left_signal *= self.pre_gain * gain
        right_signal *= self.pre_gain * gain
        # possibly make a clipping warning
        self.test_clipping([left_signal, right_signal])

        # create stereo signal by stacking the time data horizontally
        stereo_time = np.vstack((left_signal, right_signal))
        # stereo = pf.Signal(stereo_time, 
        #                    sampling_rate=self.sampling_rate, 
        #                    domain='time'
        #                    )

        return stereo_time
    
    # apply the hrtf data to an ambisonics file
    def apply_hrtf_fast(self, ambi_signal: np.ndarray, block_size=1024, gain=1.):
        """
        Convolve an ambisonic signal with the rotated HRTFs in teh frequency domain to produce a stereo signal.

        Parameters
        ----------
        ambi_signal : np.ndarray
            Input block or stream (should have shape (n_samples, n_channels)).
        block_size : int, optional
            FFT size / processing block length. Default 1024.
        gain : float, optional
            A float between 0. and 1., applied after internal gain-staging, Default is 1.

        Returns
        -------
        numpy.ndarray
            Stereo time-domain array (2 x n_samples) after convolution and gain staging.

        Notes
        -------------
        Assumes self.hrirs_nm_fft has been prepared via update_hrirs_fft(block_size).
        """
    
        # checking of channel count should be done elsewhere
        # check for correct gain also elsewhere!

        # apply rotation to the sh_hrir
        # decide: rotate ambisonics signal, or rotate SH data. don't rotate both!
        #rotated_signal = self.rotation_matrix @ ambi_signal

        # sh_hrir should have the shape (2, ambi_channels, n bins)
        #fft_sh = np.fft.fft(self.hrirs_nm, n=block_size, axis=-1)

        # make fft of our ambi signal. shape (n_samples, n_channels)
        fft_ambi = np.fft.fft(ambi_signal, n=block_size, axis=0)

        # convolve by multiplication in time domain over all channels, for each ear
        n_samples, *_ = fft_ambi.shape
        #fft_conv = np.ndarray((2, chan_count, n_samples), dtype=np.float32)
        fft_sum = np.ndarray((2, n_samples), dtype=np.complex64)

        for ear in range(2):
            #for chan in range(chan_count):
                # cult in freq domain is convolution in time
            #fft_conv[ear, chan] = fft_ambi[:, chan] * fft_sh[ear, chan, :]
            fft_sum[ear] = np.sum(fft_ambi.T * self.hrirs_nm_fft[ear, :, :], axis=0)
        
        # fft_conv = [sgn.fftconvolve(ambi_signal.T, sh_hrir.time[0, :, :], mode='full', axes=-1),
        #             sgn.fftconvolve(ambi_signal.T, sh_hrir.time[1, :, :], mode='full', axes=-1)]

        # Sum over channels -> single‑channel binaural signals
        sum_conv = np.fft.ifft(fft_sum).real

        # do gain staging
        sum_conv *= self.pre_gain * gain
        # no test for clipping because of time
        return sum_conv
    
    # updates our hrirs_nm_fft by zero-padding the time signal so the resulting fft has the correct length for convolution
    # with ambisonics audio
    def update_hrirs_fft(self, block_size: int):
        """
        Compute and store FFTs of the spherical-harmonic HRIRs zero-padded to block_size.
        
        Parameters
        ----------------
        block_size : int
            FFT length used for convolution; HRIRs are zero-padded to this length.
        """
        self.pad_to_length = block_size
        self.hrirs_nm_fft = np.fft.fft(self.hrirs_nm, n=block_size, axis=-1)
    
    # find a good gain to apply to the stereo signal, based on the ambisonics order
    def find_gain(self):
        """
        Compute a pre-gain factor for ambisonic-to-binaural rendering.

        The method uses a simple empirical estimate (base ~4 dB, dependant on 
        HRTF Preprocessing Algorithm) and scales it with the square root of the 
        number of ambisonic channels to account forincoherent summation. The 
        returned value is a linear gain (not dB) that can be multiplied with the 
        rendered stereo signals before clipping checks and any user gain is applied.

        Returns
        -------
        float
            Linear gain factor to apply (positive scalar, typically < 1).
        """

        # thought process: 2 uncorrelated signals sum to +3dB
        # 2 identical signals sum to +6dB
        # so probably ours would sum to around +4.5dB? testing showed that 4 is good so far
        # for each doubling of summed channels, this number is also doubled
        estimate = 4

        # estimate is then adjusted, depending on the HRTF preprocessing algorithm used
        i = self.process.get_gain()
        estimate *= i

        # by taking the square-root of the ambisonics order
        # we get the doubling-factor to apply to our estimate
        estimate *= np.sqrt(self.order_to_channel_n())
        # B = 1 / 10^(estimate/20)
        return 1 / np.pow(10, estimate * 0.05, dtype=np.float32)
    
    # test if  the channels are clipping
    def test_clipping(self, channels):
        """
        Check provided channels for clipping and print a warning if detected.
        The function checks whether any
        sample reaches or exceeds the amplitude threshold of 1.0 and
        prints a warning message if clipping is found. This function has
        no return value and only produces a console side-effect.

        Parameters
        ----------
        channels : sequence of array_like
            Iterable of 1-D arrays containing time-domain samples for each
            channel (e.g., left and right). 

        Notes
        ----------------
        This function only prints a warning and does not modify data.
        """

        for channel in channels:
            # test for clipping
            if np.max(channel) >= 1:
                print("\n####################\n" \
                "WARNING! WARNING" \
                "clipping detected!\n" \
                "####################")
    
    # calculates the nummer of channels corresponding to the ambisonics order
    def order_to_channel_n(self):
        """
        Return the number of spherical-harmonic channels for the current order.

        For a given ambisonic order N the number of channels is (N + 1)^2.

        Returns
        -------
        int
            Number of spherical-harmonic/ambisonic channels.
        """

        return (self.ambi_order + 1)**2

