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
    Real-time binaural renderer using spherical-harmonic HRTF decomposition.

    This class manages the complete pipeline for head-tracked binaural audio:
    loads an HRTF dataset, decomposes HRIRs into spherical-harmonic coefficients,
    performs head-rotation transformations via Wigner-D matrices, and applies
    fast frequency-domain convolution with overlap-add buffering and smooth
    crossfading during rotation transitions.

    Parameters
    ----------
    hrtf : HRTF or None, optional
        Preloaded HRTF instance. If None, the default FABIAN HRTF is loaded.
        Default is None.
    sampling_rate : int or float, optional
        Target sampling rate in Hz. HRTF resampling is automatic if needed.
        Default is 44100.
    ambi_order : int, optional
        Ambisonic order (0–7) for spherical-harmonic decomposition.
        Default is 1 (first-order).
    preprocess : str, optional
        Preprocessing algorithm: 'LS' (Least Squares), 'MagLS' (recommended),
        'TA', or 'BiMagLS'. Default is 'MagLS'.
    block_size : int, optional
        Audio block size in samples. If None, auto-set to next power of two
        greater than IR length. Default is None.

    Attributes
    ----------
    hrtf : HRTF
        Loaded HRTF dataset instance.
    sampling_rate : int
        Current sampling rate in Hz.
    ambi_order : int
        Current Ambisonic order.
    sources : spharpy.SamplingSphere
        HRTF spatial sampling grid.
    hrir_nm : np.ndarray
        Spherical-harmonic HRIR coefficients (time domain).
        Shape: (2, n_sh_channels, n_samples).
    hrir_nm_fft : np.ndarray
        Zero-padded FFT of spherical-harmonic HRIRs for convolution.
        Shape: (2, n_sh_channels, n_freq_bins).
    hrir_nm_rot : dict
        Current and previous rotated SH-HRIR spectra with keys 'old' and 'new'.
    block_size : int
        Audio processing block size in samples.
    N : int
        FFT length for overlap-add (>= block_size + IR_length - 1).
    overlap_buffer : np.ndarray
        Tail from last block for overlap-add continuity. Shape: (IR_length-1, 2).
    overlap_buffer_old : np.ndarray
        Tail from previous rotation filter during crossfade.
    pre_gain : float
        Output gain multiplier (default 0.1).
    atol : float
        Rotation tolerance in radians for skipping negligible updates (default ≈ 1°).

    Notes
    -----
    - A daemon thread manages asynchronous rotation updates (see _rotation_worker)
    - Uses SN3D normalization and real spherical harmonics (AmbiX convention)
    - Crossfade ramp: 32-sample equal-power sine² curve
    - Call close() before object destruction to cleanly terminate the rotation thread
    - Thread-safe rotation updates via queue and locks
    - Supports smooth head-rotation transitions with zero audio clicks

    Examples
    --------
    >>> sh = SphericalHarmonics(ambi_order=3, sampling_rate=48000)
    >>> for chunk in ambi_stream:
    ...     sh.set_rotation([yaw, pitch, roll])  # Non-blocking queue
    ...     output = sh.process_ola_rot(chunk)
    ...     play(output)
    >>> sh.close()
    """

    # Constructor
    def __init__(self, hrtf: HRTF = None, sampling_rate=44_100, ambi_order=1, preprocess='MagLS', block_size=None):
        """
        Initialize spherical-harmonic HRTF processing and convolution engine.
        
        This constructor loads or receives an HRTF dataset, converts HRIRs to
        spherical-harmonic coefficients for the given Ambisonic order, and
        initializes the real-time convolution pipeline with overlap-add buffers,
        rotation matrices, and a background rotation worker thread. This call is
        blocking and may take considerable time (typically several seconds).

        Parameters
        ----------
        hrtf : HRTF or None, optional
            Preloaded HRTF instance. If None, the default FABIAN HRTF dataset
            is downloaded and loaded. Default is None.
        sampling_rate : int or float, optional
            Target sampling rate in Hz for HRIR resampling. The HRTF will be
            resampled if its native sampling rate differs. Default is 44100.
        ambi_order : int, optional
            Ambisonic order (0, 1, 2, ..., 7) for spherical-harmonic 
            decomposition. Higher orders require more computation but provide
            better directional accuracy. Default is 1 (first-order Ambisonics).
        preprocess : str, optional
            Preprocessing algorithm for HRTF conversion. Currently supported:
            'LS' (Least Squares), 'MagLS' (Magnitude Least Squares, recommended),
            'TA' (not yet implemented), 'BiMagLS' (not yet implemented).
            Default is 'MagLS'.
        block_size : int, optional
            Audio processing block size in samples. Must match the buffer size
            used in real-time playback for correct overlap-add convolution.
            If None, automatically set to the next power of two greater than
            the HRIR length. Default is None.

        Attributes
        ----------
        hrtf : HRTF
            Loaded HRTF instance.
        sampling_rate : int
            Current sampling rate in Hz.
        ambi_order : int
            Current Ambisonic order.
        sources : spharpy.SamplingSphere
            Spatial sampling points of the HRTF measurement directions.
        hrir_nm : np.ndarray
            Spherical-harmonic coefficients of HRIRs in time domain.
        hrir_nm_fft : np.ndarray
            FFT of zero-padded spherical-harmonic HRIRs (for convolution).
        hrir_nm_rot : dict
            Current rotated SH-HRIRs in frequency domain with keys 'old' and 'new'.
        block_size : int
            Audio processing block size in samples.
        N : int
            FFT length for overlap-add (>= block_size + IR_length - 1).
        overlap_buffer : np.ndarray
            Tail samples from last convolution (shape: (IR_length-1, 2)).
        overlap_buffer_old : np.ndarray
            Tail samples from previous rotation filter state.
        pre_gain : float
            Overall gain factor applied to all output (default 0.1).

        Raises
        ------
        Exception
            If HRTF loading (from file and web) both fail.

        Notes
        -----
        - A daemon thread (_rotation_worker) is started for asynchronous head
        rotation updates; terminate via close() before object destruction
        - Uses SN3D normalization and real spherical harmonics (AmbiX convention)
        - Crossfade ramp length is fixed at 32 samples by default
        - Default tolerance for rotation changes: 1 degree (radians)
        - Equal-power sine-squared ramps ensure smooth audio during transitions
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

        # States for overlap-add (must be reset on seek)
        self.overlap_buffer = np.zeros((self.get_IR_length() - 1, 2), dtype=np.float32)
        self.overlap_buffer_old = np.zeros_like(self.overlap_buffer)
        self.interpolate_rot = None
        # overlap add L = self.get_IR_length()
        if block_size is None:
            block_size = next_power_of_two(self.get_IR_length())
        self.block_size = block_size
        # overlap add N
        # compute N >= M + L - 1
        self.N = next_power_of_two(self.get_IR_length() + block_size - 1)

        # prepare ramp for crossfading between rotations
        self._cf_flag = 0
        self._set_crossfade_length(32)

        # use this function because we will not calculate D otherwise
        self.update_rotation_matrix([0, 0, 0])

        # prepare pre_gain for gianstaging
        self.pre_gain = 0.1
        print(f"Setting up Spherical Harmonics took {time.time() - start:.2f}s")

    def update_order(self, order: int, block_size=None):
        """
        Update the Ambisonic order and recompute all spherical-harmonic components.
        
        This method recalculates the spherical-harmonic basis, performs preprocessing
        on the HRIRs for the new order, and updates all internal FFT buffers and 
        rotation matrices. This call is blocking and may take considerable time.

        Parameters
        ----------
        order : int
            New Ambisonic order for spherical-harmonic decomposition.
        block_size : int, optional
            Audio processing block size in samples. If None, automatically computed 
            as the next power of two greater than IR length. Default is None.

        Raises
        ------
        ValueError
            If order is negative.
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
        self.update_rotation_matrix(self._last_rotation.as_euler('zyx', degrees=True))

        self._set_crossfade_length(32)
        self._ramp = hanning_ramp(self._cf_length, 2)
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
        Terminate the current rotation thread and start a new one.
        
        This method safely closes any existing rotation worker thread, clears
        synchronization flags, and spawns a fresh daemon thread. Use this after
        major configuration changes or when rotation state needs to be reset.

        Raises
        ------
        Exception
            If the existing rotation thread cannot be terminated within 1 second.
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
            self._D['old'] = self._D['new'].copy()
            self._D['new'] = new_D
            self.hrir_nm_rot['old'] = self.hrir_nm_rot['new'].copy()
            self.hrir_nm_rot['new'] = new_rot

            # also swap old OLA buffer. new one gets updated in OLA logic
            # self.overlap_buffer_old = self.overlap_buffer.copy()
            # self.overlap_buffer[:] = 0.0

            # a crossfade needs to happen now!
            self._cf_flag = 1
            self._cf_pos = 0

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
        Worker thread for asynchronous Wigner-D matrix updates.
        
        This method runs continuously in a separate daemon thread, waiting for
        rotation update requests from the queue. When a rotation is received,
        it computes the real Wigner-D matrix, applies it to the frequency-domain
        HRIRs, and updates the filter state in a thread-safe manner with proper
        synchronization for crossfade transitions.

        Notes
        -----
        - Runs as a daemon thread started in __init__
        - Monitors self._rotation_update event and self._stop_rotation_thread flag
        - Skips negligible rotations (< self.atol tolerance)
        - Triggers crossfade when filter state changes via self._cf_flag
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
                self._D['old'] = self._D['new'].copy()
                self._D['new'] = new_D
                self.hrir_nm_rot['old'] = self.hrir_nm_rot['new'].copy()
                self.hrir_nm_rot['new'] = new_rot

                # interpolate
                # self.interpolate_rot = self.interpolate(self._last_rotation, rotation, block_size=self._cf_length)

                # # also swap old OLA buffer. new one gets updated in OLA logic
                # self.overlap_buffer_old = self.overlap_buffer.copy()
                # self.overlap_buffer[:] = 0.0

                # a crossfade needs to happen now!
                self._cf_flag = 1
                self._cf_pos = 0

            # update the last rotation
            self._last_rotation = rotation

    def get_rotation_matrix(self):
        """
        Return a copy of the currently stored Wigner-D rotation matrices.
        
        Retrieves both the 'old' and 'new' Wigner-D matrices in a thread-safe
        manner (holding the rotation lock). This allows safe access to the
        current and previous filter rotations during interpolation.

        Returns
        -------
        dict or None
            Dictionary with keys 'old' and 'new', each containing a Wigner-D
            matrix of shape ((order+1)^2, (order+1)^2). Returns None if no
            rotation matrix is available.
        """
        with self._rotation_lock:
            if self._D is None:
                return None
            
            return self._D.copy()

    # set the current rotation angle
    def _apply_rotation(self):
        """
        Apply rotation matrices to the frequency-domain HRIR coefficients.
        
        Multiplies the zero-padded frequency-domain spherical-harmonic HRIRs
        by the current and previous (for crossfade) Wigner-D rotation matrices
        using efficient matrix-vector products. Updates self.hrir_nm_rot['old']
        and self.hrir_nm_rot['new'].

        Notes
        -----
        - Operates in frequency domain (self.hrir_nm_fft already zero-padded)
        - Matrix operation: (SH channels, SH order, FFT bins)
        - Used internally during overlap-add processing
        - Thread-safe via rotation lock
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
        fft_sum = np.ndarray((2, block_size), dtype=np.complex64)

        with self._rotation_lock:
            hrir_new = self.hrir_nm_rot['new'].copy()

        for ear in range(2):
                # perform convolution in rotated frequency domain and sum in frequency domain
                fft_sum[ear] = np.sum(fft_ambi.T * hrir_new[ear, :, :], axis=0)

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
        Convolve an Ambisonic signal chunk with rotated HRTFs for crossfade.
        
        Performs frequency-domain convolution of the input Ambisonic signal
        with both the old and new rotated HRTF filters, enabling smooth
        crossfading during head-rotation transitions. Returns both results
        plus a flag indicating whether a crossfade is in progress.

        Parameters
        ----------
        ambi_signal : np.ndarray
            Input audio block with shape (n_samples, n_channels).
        block_size : int, optional
            FFT size / processing block length. Default 1024.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, bool]
            - stereo_new : Binaural output using new rotation (n_samples, 2)
            - stereo_old : Binaural output using old rotation (n_samples, 2)
            - do_fade : Boolean flag; True if crossfade is active

        Notes
        -----
        - Requires self.hrir_nm_fft to be pre-computed via update_hrirs_fft()
        - Used internally by process_ola_rot() for rotation transitions
        - Thread-safe snapshot of rotation matrices taken under lock
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
        fft_sum = np.ndarray((2, block_size), dtype=np.complex64)
        fft_sum_old = np.ndarray((2, block_size), dtype=np.complex64)

        # snapshot the filters under lock
        with self._rotation_lock:
            hrir_new = self.hrir_nm_rot['new'].copy()
            hrir_old = self.hrir_nm_rot['old'].copy()
            do_fade = self._cf_flag > 0

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
        Process a single Ambisonic audio chunk using overlap-add convolution.
        
        Applies rotated HRTFs via frequency-domain convolution, combines with
        the overlap buffer from the previous chunk, and returns the valid 
        (non-overlapped) portion of the result. Maintains self.overlap_buffer
        for the next chunk.

        Parameters
        ----------
        chunk : np.ndarray
            Ambisonic input block with shape (n_samples, n_channels).

        Returns
        -------
        np.ndarray
            Stereo binaural output with shape (n_samples, 2), derived from 
            the convolution result after overlap-add.

        Notes
        -----
        - IR length (self.get_IR_length()) must be <= block size
        - self.overlap_buffer is updated in-place for continuity
        - Does not perform crossfading; use process_ola_rot() during head rotations
        """
        # update our block size and sh_length
        block_size, *_ = chunk.shape
        sh_length = self.get_IR_length()

        # apply hrtf
        # stereo = self.apply_hrtf_chunk_fft(chunk, self.N)
        # stereo = self.apply_hrtf_chunk_slerp(chunk, self.N)
        stereo = self.apply_hrtf_chunk(chunk, self.N)

        # add overlap to stereo output
        stereo[:sh_length-1] += self.overlap_buffer

        # save new overlap buffer
        self.overlap_buffer[:] = stereo[block_size:block_size + sh_length - 1]
    
        return stereo[:block_size]

    def process_ola_rot(self, chunk: np.ndarray):
        """
        Process an Ambisonic chunk with overlap-add and smooth rotation crossfade.
        
        Convolves the input with both old and new rotated filters, performs
        separate overlap-add for each, and applies a crossfade ramp between
        them as the head rotation occurs. Maintains both overlap_buffer and
        overlap_buffer_old to support smooth transitions.

        Parameters
        ----------
        chunk : np.ndarray
            Ambisonic input block with shape (n_samples, n_channels).

        Returns
        -------
        np.ndarray
            Stereo binaural output with shape (n_samples, 2). During crossfade
            windows, this blends the old and new filter outputs.

        Notes
        -----
        - Called during head-rotation transitions via set_rotation()
        - Manages self._cf_flag, self._cf_pos, and self._ramp for fade control
        - Thread-safe access to rotation matrices and crossfade state
        - Reverts to process_ola() behavior once crossfade completes
        """

        # update our block size and sh_length
        block_size, *_ = chunk.shape
        sh_length = self.get_IR_length()

        # apply hrtf
        stereo_new, stereo_old, do_fade = self.apply_hrtf_chunk_rot(chunk, self.N)

        # take care of our overlap-adds before crossfading
        stereo_new[:sh_length-1] += self.overlap_buffer
        stereo_old[:sh_length-1] += self.overlap_buffer_old

        fade_finished = not do_fade

        if do_fade:
            with self._rotation_lock:
                # do crossfade
                fade_start = self._cf_pos
                fade_end = min(fade_start + block_size, self._cf_length)
                fade_n = max(0, fade_end - fade_start)

            if fade_n > 0:
                ramp = self._ramp[fade_start:fade_end]
                stereo_new[:fade_n] = (1.0 - ramp) * stereo_old[:fade_n] + ramp * stereo_new[:fade_n]
                self._cf_pos = fade_end

            with self._rotation_lock:
                # Do not overwrite a newly started fade from the worker thread.
                if self._cf_pos == fade_start:
                    self._cf_pos = fade_end

                    if self._cf_pos >= self._cf_length:
                        self._cf_flag = 0
                        self._cf_pos = 0
                        fade_finished = True
                    else:
                        fade_finished = False
                else:
                    # A newer rotation was published during this audio block.
                    # Its fade will begin during the next block.
                    fade_finished = False
        
        else:
            self._cf_pos = 0
            
        # Save the current/new filter tail for the next block.
        new_tail = stereo_new[block_size:block_size + sh_length - 1].copy()
        old_tail = stereo_old[block_size:block_size + sh_length - 1].copy()
        
        # extract tails for the next block
        self.overlap_buffer[:] = new_tail
        
        if fade_finished:
            # The active decoder is now the new filter. Its history must also
            # become the "old" history used at the next rotation transition.
            self.overlap_buffer_old[:] = new_tail
        else:
            # A fade is still active, so retain both independent OLA states.
            self.overlap_buffer_old[:] = old_tail
                
        return stereo_new[:block_size]
        

    def flush(self) -> np.ndarray:
        """
        Retrieve final tail samples after processing the last audio chunk.
        
        Returns the remaining M-1 samples (where M is the IR length) stored
        in the overlap buffer, which must be output to complete the stream.

        Returns
        -------
        np.ndarray
            Tail of the last convolution result with shape (IR_length-1, 2).
        """
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
        Clean up and terminate the rotation worker thread.
        
        Sets stop and update flags, joins the rotation thread with a 1-second
        timeout, and prints a warning if the thread does not terminate cleanly.
        Call this before destroying the SphericalHarmonics instance.

        Notes
        -----
        - Thread join timeout is 1.0 second
        - Safe to call multiple times
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

    def _set_crossfade_length(self, length: int) -> None:
        """
        Configure the sample-accurate equal-power crossfade ramp length.
        
        Creates an equal-power sine-squared ramp (starting at 0, ending at 1)
        of the specified sample length. The ramp is independent of audio block
        size and used for smooth transitions during head rotations.

        Parameters
        ----------
        length : int
            Crossfade duration in samples (must be >= 2).

        Raises
        ------
        ValueError
            If length < 2.

        Notes
        -----
        - Ramp phase: sin²(phase) where phase ∈ [0, π/2]
        - Equal-power property ensures constant total energy during fade
        - Stored in self._ramp and used by process_ola_rot()
        """
        if length < 2:
            raise ValueError("Crossfade length must be at least two samples.")

        self._cf_length = int(length)
        phase = np.linspace(0.0, np.pi / 2.0, self._cf_length)
        self._ramp = np.sin(phase) ** 2
        self._cf_pos = 0
