# a class that handles the conversion of HRTF data into spherical harmonics

import numpy as np
import pyfar as pf
import spharpy as sh
import time
from HRTF import HRTF

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
    hrirs : pyfar.Signal
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
    def __init__(self, hrtf=None, sampling_rate=48e3, ambi_order=1):
        # # set the hrtf. this should be a pyfar signal!
        # self.hrtf = hrtf

        # set sample rate. we should make sure this is the same as for the HRTFs!!!
        self.sampling_rate = sampling_rate

        if hrtf == None:
            # load FABIAN from the web. replace this later!
            self.hrtf = HRTF(None)
        else:
            self.hrtf = hrtf

        # make sure the sampling rate is correct and resample if necessary
        if self.hrtf.hrirs.sampling_rate != sampling_rate:
            self.hrtf.hrirs = pf.dsp.resample(self.hrtf.hrirs, sampling_rate=sampling_rate, match_amplitude='freq')

        # store the sources in a Sampling Sphere
        self.sources = sh.SamplingSphere.from_coordinates(hrtf.sources)

        # create a spherical harmonics definition, corresponding to the AmbiX convention
        self.ambi_order = ambi_order
        self.sh_definition = sh.SphericalHarmonicDefinition(self.ambi_order, normalization="SN3D", 
                                                            basis_type='real', condon_shortley=False)

        # create the spherical harmonics object from definition and sampling sphere
        self.spherical_harmonics = sh.SphericalHarmonics.from_definition(self.sh_definition, self.sources, inverse_method="pseudo_inverse")

        # create h_nm matrix 
        print("doing matrix mult to get hrirs_nm")
        hrirs_nm = (self.spherical_harmonics.basis_inv @ self.hrtf.hrirs).T
        print("convert to spherical harmonic signal")
        self.hrirs_nm = sh.SphericalHarmonicSignal.from_definition(self.sh_definition, hrirs_nm.time, hrirs_nm.sampling_rate)

        # prepare a rotation matrix, with all angles 0 currently
        angles = [0, 0, 0]
        self.rotation = sh.transforms.SphericalHarmonicRotation.from_euler('xyz', np.deg2rad(angles))
        # calculate the rotation matrix once
        #self.rotation_matrix = self.rotation.as_spherical_harmonic_matrix(self.sh_definition)
    
    # set the current rotation angle
    def set_rotation(self, angles):
        """
        Set the current spherical-harmonic rotation from Euler angles.

        Parameters
        ----------
        angles : sequence of float
            Euler angles in degrees as (alpha, beta, gamma).
        """

        self.rotation = sh.transforms.SphericalHarmonicRotation.from_euler('xyz', np.deg2rad(angles))
        # recalculate the rotation matrix once
        #self.rotation_matrix = self.rotation.as_spherical_harmonic_matrix(self.sh_definition)

    # apply a rotation and return them as signal
    def apply_rotation(self):
        """
        Apply the current spherical-harmonic rotation to `hrirs_nm` and
        return the rotated Signal.

        Returns
        -------
        hrirs_rotated : pyfar.Signal
            Time-domain HRIRs after rotation.
        """

        # creates rotated HRIRs matrix
        #hrirs_nm_rotated = self.rotation_matrix @ self.hrirs_nm
        hrirs_nm_rotated = self.rotation.apply(self.hrirs_nm)
        return hrirs_nm_rotated
    
    # apply the hrtf data to an ambisonics file
    def apply_hrtf(self, ambi_signal, gain=1.):
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
        stereo : pyfar.Signal
            Stereo binaural signal (2 x n_samples) after convolution and gain staging.
        """

        if gain > 1. or gain < 0:
            raise AttributeError("The gain must be in range [0., 1.].")

        start = time.time()
        # apply rotation to the sh_hrir
        sh_hrir = self.apply_rotation()
        print(f"Applying rotation took {time.time() - start:.4f} seconds")
        # Check channel count by comparing the channel shape
        # we know the channel shape for ambi_signal is (channels,)
        ambi_ch, *_ = ambi_signal.cshape
        # we know the channel shape for sh_hrir is (2, channels)
        *_, sh_hrir_ch = sh_hrir.cshape
        if ambi_ch != sh_hrir_ch:
            raise ValueError("Channel counts must match (16 for 3rd order).")

        start = time.time()
        # sh_hrir should have the shape (2, ambi_order)
        # Convolve each channel separately for left and right
        # instantly store it as time data
        left_conv = pf.dsp.convolve(
            ambi_signal,
            pf.Signal(sh_hrir.time[0, :, :], self.sampling_rate, domain='time'), # need to get the left channel here
            mode='full',
            method='overlap_add'
        ).time
        right_conv = pf.dsp.convolve(
            ambi_signal,
            pf.Signal(sh_hrir.time[1, :, :], self.sampling_rate, domain='time'), # need to get the right channel here
            mode='full',
            method='overlap_add'
        ).time
        print(f"Convolving signals took {time.time() - start:.4f} seconds")

        start = time.time()
        # Sum over channels -> single‑channel binaural signals
        left_signal = np.sum(left_conv, axis=0)
        right_signal = np.sum(right_conv, axis=0)

        # do gain staging
        # Find the maximum absolute value across both channels
        peak = max(np.abs(left_signal).max(), np.abs(right_signal).max())
        # Avoid division by zero
        if peak > 0:
            pre_gain = 0.99 / peak   # 0.99 leaves a tiny headroom
            left_signal *= pre_gain * gain
            right_signal *= pre_gain * gain

        # create stereo signal by stacking the time data horizontally
        stereo_time = np.vstack((left_signal, right_signal))
        stereo = pf.Signal(stereo_time, sampling_rate=self.sampling_rate, domain='time')
        print(f"Summing signals, Gainstaging and creating stereo took {time.time() - start:.4f} seconds")

        return stereo