# a class that handles the conversion of HRTF data into spherical harmonics

import numpy as np
import pyfar as pf
import spharpy as sh
import time
import threading
import queue
from shroom.utils.rotation_utils import wigner_d_matrix
from scipy.spatial.transform import Rotation

from playamb.audio.engine.hrtf import HRTF, Processing
from playamb.audio.rotation.rotation import RotationMatrix
from playamb.utils.utils import next_power_of_two
from playamb.utils.utils import order_to_channel_n
from playamb.utils.utils import hanning_ramp

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
    def __init__(self, hrtf: HRTF = None, sampling_rate=44_100, ambi_order=1, preprocess='MagLS', block_size=None):
        """Initialize spherical-harmonic HRTF processing. 
        This blocks and will take a considerable time to finish processing.

        Parameters
        -----------
        hrtf : HRTF or None, optional
            Preloaded HRTF instance. If None, a default dataset is loaded.
        sampling_rate : int or float, optional
            Target sampling rate in Hz for HRIR resampling. Default is 48000.
        ambi_order : int, optional
            Ambisonic order used for the spherical-harmonic decomposition.
        prepocess : str, optional
            Preprocessing algorithm to use on HRTF. Currently LS and MagLS are implemented. MagLS is default
        block_size : int, optional
            This sets the block size of the audio blocks we expect to process. Necessary for correct OLA performance!
        """
        start = time.time()
        
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
        if self.hrtf.fs != sampling_rate:
            self.hrtf.resample(sampling_rate)

        # store the sources in a Sampling Sphere
        self.sources = sh.SamplingSphere.from_coordinates(self.hrtf.sources)

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
                                                                        #  inverse_method="pseudo_inverse"
                                                                         )

        # create hrtf processing unit
        self.process = Processing()

        # create h_nm matrix 
        hrirs_nm = self.process.apply_preprocessing(self.hrtf.hrirs, 
                                               self.spherical_harmonics,
                                               algorithm=preprocess
                                               )
        
        print("Convert to Spherical Harmonic Signal")
        self.hrir_nm = sh.SphericalHarmonicSignal.from_definition(self.sh_definition, 
                                                                   hrirs_nm.time, 
                                                                   hrirs_nm.sampling_rate
                                                                   ).time
        
        *_, pad_to_length = self.hrir_nm.shape

        # update this later by calling update_hrirs_fft() once we know the desired fft length
        self.hrir_nm_fft = np.fft.fft(self.hrir_nm, next_power_of_two(pad_to_length), axis=-1)

        # create SH rotation unit
        self.rotation = RotationMatrix()
        # prepare rotated hrir's, and our rotation matrix, with all angles 0 currently
        self.hrir_nm_rot = {
                           'old' : self.hrir_nm_fft.copy(),
                           'new' : self.hrir_nm_fft.copy()
                           }
        self._D = {
                  'old' : np.eye(order_to_channel_n(ambi_order)),
                  'new' : np.eye(order_to_channel_n(ambi_order))
                  }
        self._last_rotation = Rotation.from_euler('zyx', (0, 0, 0))
        self.atol = np.deg2rad(1)

        self._rotation_lock = threading.Lock()
        self._rotation_queue = queue.Queue(maxsize=1)
        self._rotation_update = threading.Event()
        self._stop_rotation_thread = threading.Event()

        # start the rotation thread
        self._rotation_thread = threading.Thread(
            target=self._rotation_worker,
            daemon=True,
        )
        self._rotation_thread.start()
        
        # use this function because we will not calculate D otherwise
        self.update_rotation_matrix([0, 0, 0])

        # States for overlap-add (must be reset on seek)
        self.overlap_buffer = None 
        self.overlap_buffer_old = None 
        # overlap add L = self.get_IR_length()
        if block_size is None:
            block_size = next_power_of_two(self.get_IR_length())
        self.block_size = block_size
        # overlap add N
        # compute N >= M + L - 1
        self.N = next_power_of_two(self.get_IR_length() + block_size - 1)

        # prepare ramp for crossfading between rotations
        self._cf_length = 32
        self._ramp = hanning_ramp(self._cf_length, 2)
        self._cf_flag = False

        # prepare pre_gain for gianstaging
        self.pre_gain = 0.1
        print(f"Setting up Spherical Harmonics took {time.time() - start:.2f}s")

    def update_order(self, order: int, block_size=None):
        """
        Updates the order of the Spherical Harmonics object. Recalculates a bunch of the internal variables. 
        This call blocks and will take a considerable time to finish processing.
        """
        # create a spherical harmonics definition, corresponding to the AmbiX convention
        self.ambi_order = order
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
        # create h_nm matrix 
        hrirs_nm = self.process.apply_preprocessing(self.hrtf.hrirs, 
                                               self.spherical_harmonics,
                                               algorithm='MagLS'
                                               )
        
        # update our hrir_nm
        self.hrir_nm = sh.SphericalHarmonicSignal.from_definition(self.sh_definition, 
                                                                   hrirs_nm.time, 
                                                                   hrirs_nm.sampling_rate
                                                                   ).time
        
        # prepare hrir_nm_fft
        if block_size is None:
            block_size = next_power_of_two(self.get_IR_length())
        # this calls update_hrirs_fft internally!
        self.update_process_variables(block_size)
        
    
    def update_process_variables(self, block_size: int):
        """
        Updated the variables necessary to perform successfull Overlap-Add algorithm on processing.

        Parameters
        ----------
        block_size : int
            The buffer size we are currently working with
        """
        self.block_size = block_size
        # compute N >= M + L - 1
        self.N = next_power_of_two(self.get_IR_length() + block_size - 1)

        # update our sh fft coefficients once (and on each rotation update)
        self.update_hrirs_fft(self.N)

        # allocate buffers
        self.overlap_buffer = np.zeros((self.get_IR_length() - 1, 2), dtype=np.float32)
        self.overlap_buffer_old = np.zeros_like(self.overlap_buffer)

    def get_IR_length(self):
        """
        Return the impulse-response length (number of samples) of the current HRIRs.

        Returns
        -----------
        int
            Number of samples in the HRIR time-domain representation.
        """
        *_, n_samples = self.hrir_nm.shape
        return n_samples
    
    def restart_rotation(self):
        """
        Closes the possibly still running rotation thread, then creates a new rotation thread and starts it.
        """
        # close old rotation thread if it exists and is alive
        if self._rotation_thread is not None:
            if self._rotation_thread.is_alive():
                # set the stop flag, set rotation update and hope old thread will close
                self._stop_rotation_thread.set()
                self._rotation_update.set()
                self._rotation_thread.join(1.)
                if self._rotation_thread.is_alive():
                    raise Exception("Couldn't close rotation thread.")
                
            self._rotation_thread = None

        # clear the stop and rotation flag
        self._stop_rotation_thread.clear()
        self._rotation_update.clear()
        # start a new rotation thread
        self._rotation_thread = threading.Thread(
            target=self._rotation_worker,
            daemon=True,
        )
        self._rotation_thread.start()

    def update_rotation_matrix(self, angles, convention='zyx'):
        """
        Directly Compute and update the internal Wigner-D matrix based on thie given Euler angles in degrees (zyx). This will block the thread!
        The updating is done in a thread-safe manner.

        Parameters
        ----------
        angles : sequence of float
            Euler angles in degrees as (z, y, x).
        convention : str
            A string identifying the used convention/order of the angles. 'zyx' is default.
        """
        rotation = Rotation.from_euler(convention, angles, degrees=True)
        alpha, beta, gamma = rotation.as_euler("zyz")
        new_D = self.rotation.real_wigner_d_matrix(self.ambi_order, alpha, beta, gamma)

         # using the fft, since we expect this to be faster than time domain
        # 3. Apply rotation
        # data is (Channels, SH, Time/Freq)
        # D is (SH, SH)
        # We want D @ data
        # einsum: ij, cjk -> cik
        # self.hrir_nm_rot = np.einsum("ij, cjk -> cik", self.D, hrir_rot)
        new_rot = new_D @ self.hrir_nm_fft
        with self._rotation_lock:
            self._D['old'] = self._D['new']
            self._D['new'] = new_D
            self.hrir_nm_rot['old'] = self.hrir_nm_rot['new']
            self.hrir_nm_rot['new'] = new_rot

    def set_rotation(self, angles, convention='zyx'):
        """
        Compute and update the internal Wigner-D matrix based on thie given Euler angles in degrees (zyx) by passing it to the worker thread.
        The updating is done in a thread-safe manner.

        Parameters
        ----------
        angles : sequence of float
            Euler angles in degrees as (z, y, x).
        convention : str
            A string identifying the used convention/order of the angles. 'zyx' is default.
        """
        try:
            self._rotation_queue.put_nowait((angles, convention))
        except queue.Full:
            try:
                # drop the last cached block. queue should be empty now, since we only have size 1
                self._rotation_queue.get_nowait()
                print("Queue is already full! Dropping this rotation update")
            except queue.Empty:
                pass
            # try putting a new block into the queue again after draining it
            self._rotation_queue.put_nowait((angles, convention))

        # finally, set the flag if we managed to successfully set a rotation
        self._rotation_update.set()

    def _rotation_worker(self):
        """
        A function continously updating the Wigner-D matrix. Called in a seperate thread from __init__
        """
        while not self._stop_rotation_thread.is_set():

            # sleep cheaply until something happens. this blocks!
            self._rotation_update.wait()
            if self._stop_rotation_thread.is_set():
                break
            # clear the update flag and consume the current angles and convention
            self._rotation_update.clear()

            # get pending angles and rotation
            try:
                angles, convention = self._rotation_queue.get_nowait()
            except queue.Empty:
                continue

            rotation = Rotation.from_euler(convention, angles, degrees=True)
            # skip this calculation if no meaningful rotation has happened
            if rotation.approx_equal(self._last_rotation, atol=self.atol):
                continue

            # update the last rotation
            self._last_rotation = rotation
            # compute wigner D matrix
            alpha, beta, gamma = rotation.as_euler("zyz")
            new_D = self.rotation.real_wigner_d_matrix(self.ambi_order, alpha, beta, gamma)
    
            # using the fft, since we expect this to be faster than time domain
            # 3. Apply rotation
            # data is (Channels, SH, Time/Freq)
            # D is (SH, SH)
            # We want D @ data
            # einsum: ij, cjk -> cik
            # self.hrir_nm_rot = np.einsum("ij, cjk -> cik", self.D, hrir_rot)
            new_rot = new_D @ self.hrir_nm_fft
            with self._rotation_lock:
                self._D['old'] = self._D['new']
                self._D['new'] = new_D
                self.hrir_nm_rot['old'] = self.hrir_nm_rot['new']
                self.hrir_nm_rot['new'] = new_rot
                # a crossfade needs to happen now!
                self._cf_flag = True
                

    def get_rotation_matrix(self):
        """
        Returns a copy fo the currently stored Wigner D matrix in a thread safe manner.
        Returns None if no rotation matrix is available.
        """
        with self._rotation_lock:
            if self._D is None:
                return None
            
            return self._D.copy()

    # set the current rotation angle
    def _apply_rotation(self):
        """
        Applies rotation to to the base SH coefficients in frequency domain. Thus, the already zero-padded
        signal is used, because that is what the frequency domain data is based on. Updates the internal state
        of hrir_nm_rot. 
        """
        # using the fft, since we expect this to be faster than time domain
        # 3. Apply rotation
        # data is (Channels, SH, Time/Freq)
        # D is (SH, SH)
        # We want D @ data
        # einsum: ij, cjk -> cik
        # self.hrir_nm_rot = np.einsum("ij, cjk -> cik", self.D, hrir_rot)
        D = self.get_rotation_matrix()
        # if D is None, do nothing and keep our latest rotated hrir_nm
        if D is not None:
            # using the fft, since we expect this to be faster than time domain
            self.hrir_nm_rot['old'] = D['old'] @ self.hrir_nm_fft
            self.hrir_nm_rot['new'] = D['new'] @ self.hrir_nm_fft
    
    # apply the hrtf data to an ambisonics file
    def apply_hrtf_full(self, ambi_signal: pf.Signal, gain=1.):
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
        sh_hrir = self.hrir_nm.copy()

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
    def apply_hrtf_chunk(self, ambi_signal: np.ndarray, block_size=1024):
        """
        Convolve an ambisonic signal with the rotated HRTFs in the frequency domain to produce a stereo signal.
        This function expects a signal as NDarray, as well as a blocksize to operate on (must fit the blocksize of `self.hrir_nm_rot`)

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
            Stereo time-domain array (n_samples, 2) after convolution and gain staging.

        Notes
        -------------
        Assumes self.hrirs_nm_fft has been prepared via update_hrirs_fft(block_size).
        """
    
        # checking of channel count should be done elsewhere
        # check for correct gain also elsewhere!

        # apply rotation to the sh_hrir
        # not needed, we only apply rotation and update hrir_nm_rot on a new rotation
        # self._apply_rotation()

        # sh_hrir should have the shape (2, ambi_channels, n bins)
        #fft_sh = np.fft.fft(self.hrirs_nm, n=block_size, axis=-1)

        # make fft of our ambi signal. shape (n_samples, n_channels)
        fft_ambi = np.fft.fft(ambi_signal, n=block_size, axis=0)

        # convolve by multiplication in time domain over all channels, for each ear
        n_samples, *_ = fft_ambi.shape
        fft_sum = np.ndarray((2, n_samples), dtype=np.complex64)

        for ear in range(2):
            with self._rotation_lock:
                # perform convolution in rotated frequency domain and sum in frequency domain
                fft_sum[ear] = np.sum(fft_ambi.T * self.hrir_nm_rot['new'][ear, :, :], axis=0)

        # Sum over channels -> single‑channel binaural signals
        sum_conv = np.fft.ifft(fft_sum).real

        # do gain staging
        sum_conv *= self.pre_gain
        # no test for clipping because of time
        # transpose, since sounddevice expects shape (n_samples, n_channels)
        return sum_conv.T
    
    # apply the hrtf data to an ambisonics file
    def apply_hrtf_chunk_rot(self, ambi_signal: np.ndarray, block_size=1024):
        """
        Convolve an ambisonic signal chunk with the rotated HRTFs in the frequency domain to produce a stereo signal.
        This function expects

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
        numpy.ndarray, numpy.ndarray, bool
            2x Stereo time-domain array (n_samples, 2) after convolution and gain staging.
            once for old rotation, once for new rotation.
            If a crossfade is required, bool will be True

        Notes
        -------------
        Assumes self.hrirs_nm_fft has been prepared via update_hrirs_fft(block_size).
        """
    
        # checking of channel count should be done elsewhere
        # check for correct gain also elsewhere!

        # apply rotation to the sh_hrir
        # not needed, we only apply rotation and update hrir_nm_rot on a new rotation
        # self._apply_rotation()

        # sh_hrir should have the shape (2, ambi_channels, n bins)
        #fft_sh = np.fft.fft(self.hrirs_nm, n=block_size, axis=-1)

        # make fft of our ambi signal. shape (n_samples, n_channels)
        fft_ambi = np.fft.fft(ambi_signal, n=block_size, axis=0)

        # convolve by multiplication in time domain over all channels, for each ear
        n_samples, *_ = fft_ambi.shape
        fft_sum = np.ndarray((2, n_samples), dtype=np.complex64)
        fft_sum_old = np.ndarray((2, n_samples), dtype=np.complex64)

        # snapshot the filters under lock
        with self._rotation_lock:
            hrir_new = self.hrir_nm_rot['new'].copy()
            hrir_old = self.hrir_nm_rot['old'].copy()
            do_fade = self._cf_flag
            # Consume the flag so it doesn't fire again on the next block
            self._cf_flag = False

        for ear in range(2):
            # perform convolution in rotated frequency domain and sum in frequency domain
            # get old and new rotation resluts
            fft_sum[ear] = np.sum(fft_ambi.T * hrir_new[ear, :, :], axis=0)
            fft_sum_old[ear] = np.sum(fft_ambi.T * hrir_old[ear, :, :], axis=0)

        sum_conv = np.fft.ifft(fft_sum).real
        sum_conv_old = np.fft.ifft(fft_sum_old).real

        # do gain staging
        sum_conv *= self.pre_gain
        sum_conv_old *= self.pre_gain
        # no test for clipping because of time
        # transpose, since sounddevice expects shape (n_samples, n_channels)
        return sum_conv.T, sum_conv_old.T, do_fade

    # process a single chunk of data, performing an overlap-add algorithm
    def process_ola(self, chunk: np.ndarray):
        """
        Process a single Ambisonics chunk with overlap-add.
        This method should maintains `self.overlap_buffer` and updates it.
        """
        # update our block size and sh_length
        block_size, *_ = chunk.shape
        sh_length = self.get_IR_length()

        # apply hrtf
        stereo = self.apply_hrtf_chunk(chunk, self.N)

        # add overlap to stereo output
        stereo[:sh_length-1] += self.overlap_buffer

        # save new overlap buffer
        self.overlap_buffer[:] = stereo[block_size:block_size + sh_length - 1]
    
        return stereo[:block_size]

    def process_ola_rot(self, chunk: np.ndarray):
        """
        Process a single Ambisonics chunk with overlap-add. and crossfading between old and new stereo input and overlap-add buffer
        This method maintains `self.overlap_buffer` and `self.overlap_buffer_old`and updates them.
        """

        # update our block size and sh_length
        block_size, *_ = chunk.shape
        sh_length = self.get_IR_length()

        # apply hrtf
        stereo_new, stereo_old, do_fade = self.apply_hrtf_chunk_rot(chunk, self.N)

        # take care of our overlap-adds before crossfading
        stereo_new[:sh_length-1] += self.overlap_buffer       # only new filter's tail
        stereo_old[:sh_length-1] += self.overlap_buffer_old   # only old filter's tail

        # extract tails for the next block
        tail_new = stereo_new[block_size:block_size + sh_length - 1].copy()
        tail_old = stereo_old[block_size:block_size + sh_length - 1].copy()

        if do_fade:
            # do crossfade
            output = stereo_new.copy()
            output[:self._cf_length] = (1.0 - self._ramp) * stereo_old[:self._cf_length] + \
                                               self._ramp * stereo_new[:self._cf_length]
            
            # crossfade our tails!
            # new tail is the currently incoming block!
            tail_output = tail_new.copy()
            # cf between the two. so we add the overlap buffer (one update behind) to the new tail (currently incoming block)
            tail_output[:self._cf_length] = (1.0 - self._ramp) * tail_old[:self._cf_length] + \
                                                    self._ramp * tail_new[:self._cf_length]

            # After the fade, we have fully transitioned to the new filter.
            # Unify both states to the crossfaded tail so the OLA stays coherent.
            self.overlap_buffer[:]     = tail_output
            self.overlap_buffer_old[:] = tail_output
        else: 
            # No rotation – both filters are identical; just use the new one.
            output = conv_new
            self.overlap_buffer[:]     = tail_new
            self.overlap_buffer_old[:] = tail_new
                
        return output[:block_size]
        

    def flush(self) -> np.ndarray:
        """Return the final M-1 tail samples after the last chunk."""
        return self.overlap_buffer.copy()

    # updates our hrirs_nm_fft by zero-padding the time signal so the resulting fft has the correct length for convolution
    # with ambisonics audio
    def update_hrirs_fft(self, block_size: int):
        """
        Compute and store FFTs of the spherical-harmonic HRIRs zero-padded to block_size.
        Additionally, recalculates the current rotation matrix to fit the necessary size
        
        Parameters
        ----------------
        block_size : int
            FFT length used for convolution; HRIRs are zero-padded to this length.
        """
        self.hrir_nm_fft = np.fft.fft(self.hrir_nm, n=block_size, axis=-1)
        self._apply_rotation()

    def close(self):
        """
        Cleans up this file so it can close correctly
        """
        self._stop_rotation_thread.set()
        self._rotation_update.set()

        # terminate the rotation_thread
        if self._rotation_thread is not None:
            self._rotation_thread.join(timeout=1.0)

            # debug message if processing_thread didn't terminate properly
            if self._rotation_thread.is_alive():
                print("Warning: Rotation thread did not terminate cleanly.")
    
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
                "WARNING! WARNING!\n" \
                f"Clipping detected!\n" \
                "####################")

    def set_atol(self, atol):
        """
        Sets the tolerance for which we accept rotation updates. Default tolerance is 2°.

        Parameters
        ----------
        atol : int
            The tolerance in degrees.
        """
        self.atol = np.deg2rad(atol)
