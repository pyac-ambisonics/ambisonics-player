import numpy as np
import sounddevice as sd
import queue
import threading
import soundfile as sf

from playamb.audio.data.ambifile import AmbisonicsFile
from playamb.audio.engine.spherical import SphericalHarmonics
from playamb.utils.utils import next_power_of_two
from playamb.utils.utils import hanning_ramp

class AudioPlayer:
    """
    Streaming audio player for GUI control.

    Supports:
    - load WAV
    - load NumPy audio array
    - load pyfar.Signal
    - play
    - pause / resume
    - stop
    - real-time volume control
    - start offset
    - seek with progress bar
    - loop playback
    """

    def __init__(self, ambi_file: AmbisonicsFile, sh: SphericalHarmonics, gain=1.0):
        self.position = 0
        self.is_loaded = False
        self.source_name = None
        
        # Check channel count by comparing the channel shape
        # we know the channel shape for sh_hrir is (2, channels)
        _, sh_hrir_ch, _ = sh.hrir_nm.shape
        if ambi_file.get_num_channels() != sh_hrir_ch:
            raise ValueError("Channel counts must match (e.g. 16 for 3rd order).")
        
        if gain > 1. or gain < 0:
            raise AttributeError("The gain must be in range [0., 1.].")
        
        # spherical processing of ambisonics file instances
        self.ambi_file = ambi_file
        self.sh = sh
        # some important internal variables
        self.gain = gain
        self.fs = ambi_file.get_samplerate()
        # binarual is always stereo
        self.n_channels = 2

        self.audio_queue = queue.Queue(maxsize=10)

        # some control flags for handling playing, stopping and pausing
        self.play_event = threading.Event()
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.seek_event = threading.Event()
        self.seek_target = 0.0
        self.loop = False

        # keeping track of blocks
        self.current_block = None
        self.current_block_pos = 0

        # Thread and stream objects
        self.processing_thread = None
        self.stream = None
        # construction received a valid, validated source: ready to play
        self.is_loaded = True

    def _put_stop_aware(self, item) -> bool:
        """
        Put an item into the audio queue, waiting while playback is alive.

        Returns False (without putting) once stop_event is set, so the
        processing thread never blocks past a stop and never drops blocks.
        """
        while not self.stop_event.is_set():
            try:
                self.audio_queue.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False
    
    def update_process_variables(self):
        """
        Wraps Spherical Harmonics process variables update
        """
        self.sh.update_process_variables(self.ambi_file.get_chunk_size())
        
    def _process_loop(self):
        """Processing thread: reads chunks, processes, and puts into queue."""
        
        print("Start processing thread.")

        # ensures we only wait a fraction of the time we generally need
        #put_timeout = (self.block_size / self.fs) * 0.1

        # while we are not stopping
        while not self.stop_event.is_set():

            # seek handling: update current time and internal state
            if self.seek_event.is_set():
                # clear the seek event
                self.seek_event.clear()
                self.current_block = None
                self.current_block_pos = 0
                self._clear_audio_queue()
                self.ambi_file.seek_to_time(self.seek_target)
                self.sh.update_process_variables(self.ambi_file.get_chunk_size())
                #continue to enxt loop iteration
                continue

            # Pause handling: block if paused but not stopped
            if self.pause_event.is_set():
                # Wait for play_event (resume or stop) with a bounded timeout
                # so a stop racing past this check cannot strand the thread.
                while not self.play_event.wait(timeout=0.1):
                    if self.stop_event.is_set():
                        break
            
                # If stop was triggered, exit the outer loop
                if self.stop_event.is_set():
                    break

            # Read next chunk
            chunk, end_of_file = self.ambi_file.get_next_chunk()
            if chunk is None:
                break
            
            # processes one chunk and return a stereo block, 
            # updating the overlap buffer internally.
            processed_block = self.sh.process_ola(chunk)

            # # Put into queue (waits instead of dropping; aborts on stop)
            # if not self._put_stop_aware(processed_block):
            #     break

            # Put into queue
            try:
                self.audio_queue.put(processed_block, timeout=0.1)
            except queue.Full:
                # probably the playback was stopped or paused. in this case we can safely drop the block without notifying
                if not self.pause_event.is_set() and not self.stop_event.is_set():
                    # if not we print a warning
                    print("Warning!!! Queue is full! Dropping this block")

            # at end of file flush any remaining output from SH-overlap-buffer
            if end_of_file:
                if not self._put_stop_aware(self.sh.flush()):
                    break

                #check for looping
                if self.loop:
                    # resets processor and will read chunk 0 next
                    self.sh.update_process_variables(self.ambi_file.get_chunk_size())
                    self.ambi_file.reset_position()
                else:
                    # queue None-item as flag that playback has ended
                    if not self._put_stop_aware(None):
                        break
                    # clear the play event
                    self.play_event.clear()

    def load_audio(self, audio, sample_rate, source_name="Audio array"):
        """
        Load audio directly from a NumPy array.

        Internal format:
        audio.shape == (samples, 2)
        """

        audio = np.asarray(audio, dtype="float32")

        # Mono to stereo
        if audio.ndim == 1:
            audio = np.column_stack((audio, audio))

        # Convert (channels, samples) to (samples, channels)
        # This is useful for pyfar.Signal, which may use (2, samples)
        if audio.ndim == 2 and audio.shape[0] == 2 and audio.shape[1] > 2:
            audio = audio.T

        # Player only accepts binaural stereo signal
        if audio.ndim != 2 or audio.shape[1] != 2:
            raise ValueError(
                f"Player expects stereo audio with shape (samples, 2), "
                f"but got {audio.shape}"
            )

        # Avoid clipping
        max_value = np.max(np.abs(audio))
        if max_value > 1.0:
            audio = audio / max_value
            print("Audio was normalized to avoid clipping.")

        # Stop previous playback before loading new audio
        self.stop()

        self.audio = audio
        self.fs = int(sample_rate)
        self.position = 0
        self.is_loaded = True
        self.source_name = source_name

        print("\nAudio loaded successfully.")
        self.print_info()

    def load_wav(self, file_path: str):
        """
        Load a WAV file from disk.
        """

        audio, sample_rate = sf.read(file_path, dtype="float32")
        self.load_audio(audio, sample_rate, source_name=file_path)

    def load_pyfar_signal(self, signal):
        """
        Load audio from a pyfar.Signal object.

        Expected:
        - signal.time contains audio data
        - signal.sampling_rate contains the sampling rate
        """

        audio = signal.time
        sample_rate = signal.sampling_rate
        self.load_audio(audio, sample_rate, source_name="pyfar.Signal")

    def _audio_callback_robust(self, outdata, frames, time, status):
        """sounddevice callback – outputs from queue or silence if paused."""

        # debug if something went wrong last callback
        if status:
            print(f"Audio callback status: {status}")

        # check pasue status
        if self.pause_event.is_set():
            # fill with silence if paused
            outdata.fill(0)
            return
        
        # prepare output buffer for this callback
        out = np.zeros((frames, self.n_channels), dtype=np.float32)
        need = frames
        write_pos = 0

        while need > 0:
            # if no curent block is availabel or it hasn't been fully consumed yet, process a new block
            if self.current_block is None or self.current_block_pos >= len(self.current_block):
                try:
                    data = self.audio_queue.get_nowait()
                except queue.Empty:
                    print("Audio queue underrun: outputting silence")
                    # Underflow: not enough processed audio ready: output 0's
                    outdata[:] = out * self.gain
                    return
                
                # we set data = None in our _process_loop() if we reached the end of file to signal playback has ended
                if data is None:
                    # End of file
                    outdata.fill(0)
                    # tell sounddevice to cleanup and stop the callback
                    raise sd.CallbackStop
                
                # ensure shape (n_samples, n_channels) and correct dtype
                self.current_block = np.asarray(data, dtype=np.float32)
                self.current_block_pos = 0

            # write the current block to output
            # get sample counts for writing
            available = len(self.current_block) - self.current_block_pos
            take = min(available, need)
            
            # copy current block to output buffer
            out[write_pos:write_pos + take] = self.current_block[self.current_block_pos:self.current_block_pos + take]
            self.current_block_pos += take
            write_pos += take
            need -= take

        # write output to outdata
        outdata[:] = out * self.gain

    def play(self):
        """
        Start or resume playback.
        """

        if not self.is_loaded:
            print("No audio loaded.")
            return

        # If currently paused, resume
        if self.pause_event.is_set():
            self.resume()
            return

        # If already playing, do nothing
        if self.stream is not None and self.play_event.is_set():
            print("Already playing.")
            return

        # If stream exists but is not playing, close it first
        if self.stream is not None:
            self.stop()

        # reset stop and pause flags
        self.stop_event.clear()
        self.pause_event.clear()

        # reset all processing variables before starting the processing thread. 
        # also ensures block size is calculated before we initialize the stream
        self.sh.update_process_variables(self.ambi_file.get_chunk_size())

        # Start processing thread
        self.processing_thread = threading.Thread(target=self._process_loop)
        self.processing_thread.start()

        self.stream = sd.OutputStream(
            samplerate=self.fs,
            channels=self.n_channels,
            dtype="float32",
            callback=self._audio_callback_robust,
            # let sounddevice choose blocksize
            # blocksize=self.N,
            finished_callback=self._on_stream_finished
        )

        # set palying flag to true
        self.play_event.set()

        print(f"Playing from {self.ambi_file.get_current_time():.2f} seconds...")
        self.stream.start()

    def _on_stream_finished(self):
        """Called when stream stops (e.g., at end of file)."""
        print("Stream finished.")
        # set the stop flag
        self.stop_event.set()

    def pause(self):
        """
        Pause playback and keep current position.
        """

        if not self.play_event.is_set():
            return

        self.pause_event.set()
        self.play_event.clear()
        
        print(f"Paused at {self.ambi_file.get_current_time():.2f} seconds.")

    def resume(self):
        """
        Resume playback from current position.
        """

        if not self.is_loaded:
            return

        self.pause_event.clear()

        if self.stream is None:
            self.play()
        else:
            self.play_event.set()
            print(f"Resumed at {self.ambi_file.get_current_time():.2f} seconds.")

    def stop(self, reset_position=True):
        """
        Stop playback. Clears the stream and the processing thread. Call this before exiting the program.

        Parameters
        ----------
        reset_position : bool
            If True, reset playback position to the beginning.
        """

        # set the stop flag
        self.stop_event.set()
        # clear the pause flag, just in case it was blocking our processing thread
        self.pause_event.clear()
        # set the play flag to true, to wake the waiting processing thread
        self.play_event.set()

        # clean up the stream
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        # terminate the processing_thread
        if self.processing_thread is not None:
            self.processing_thread.join(timeout=1.0)

            # debug message if processing_thread didn't terminate properly
            if self.processing_thread.is_alive():
                print("Warning: Processing thread did not terminate cleanly.")

        # Clear queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

        if reset_position:
            self.ambi_file.reset_position()

        # clear the play event at the end
        self.play_event.clear()
        print("Playback stopped.")

    def close(self):
        """
        Closes the player completely, preparing it for termination. The player cannot be used after calling this.
        """
        self.stop()
        self.sh.close()

    def set_volume(self, volume: float):
        """
        Set playback volume between 0.0 and 1.0.
        """

        self.gain = max(0.0, min(float(volume), 1.0))
        print(f"Volume set to {self.gain:.2f}")

    def set_start_offset(self, seconds: float):
        """
        Set the playback start position in seconds.
        """

        if not self.is_loaded:
            print("No audio loaded.")
            return

        duration = self.get_duration()

        if seconds >= duration:
            print(f"Offset is too large. Audio duration is only {duration:.2f} seconds.")
            return
        
        # in case we are not already streaming/processing data just update internal state
        if self.stream is None or self.processing_thread is None:
            self.sh.update_process_variables(self.ambi_file.get_chunk_size())
            self.ambi_file.seek_to_time(seconds)
            self._clear_audio_queue()
            self.position = int(seconds * self.fs)
            print(f"Seeked to {seconds:.2f} seconds.")
            return

        self._request_seek(seconds)
        self.position = int(seconds * self.fs)
        print(f"Seek requested to {seconds:.2f} seconds.")

    def seek_to(self, seconds: float):
        """
        Jump to a specific playback position in seconds.
        This is mainly used by the GUI progress bar.
        """

        if not self.is_loaded:
            print("No audio loaded.")
            return
        
        duration = self.get_duration()

        if seconds >= duration:
            print(f"Offset is too large. Audio duration is only {duration:.2f} seconds.")
            return

        # in case we are not already streaming/processing data just update internal state
        if self.stream is None or self.processing_thread is None:
            self.sh.update_process_variables(self.ambi_file.get_chunk_size())
            self.ambi_file.seek_to_time(seconds)
            self._clear_audio_queue()
            self.position = int(seconds * self.fs)
            print(f"Seeked to {seconds:.2f} seconds.")
            return

        self._request_seek(seconds)
        self.position = int(seconds * self.fs)
        print(f"Seek requested to {seconds:.2f} seconds.")

    def _clear_audio_queue(self):
        """
        Drains and discards all queued items in the audio_queue
        """
        while not self.audio_queue.empty():
            try:
                # drain the next item in the queue
                self.audio_queue.get_nowait()
            except queue.Empty:
                # exit when queue is empty
                break

    def _request_seek(self, seconds: float):
        self.seek_target = max(0.0, min(seconds, self.get_duration()))
        self.current_block = None
        self.current_block_pos = 0
        self._clear_audio_queue()
        self.seek_event.set()

    def set_loop(self, enabled: bool):
        """
        Enable or disable loop playback.
        """

        self.loop = bool(enabled)
        print(f"Loop set to {self.loop}")

    def get_duration(self):
        """
        Return total duration in seconds.
        """

        if not self.is_loaded:
            return 0.0

        return self.ambi_file.get_duration()

    def get_current_time(self):
        """
        Return current playback time in seconds.
        """

        if not self.is_loaded:
            return 0.0
        
        # in case we are currently in a seek event: update time correctly
        if self.seek_event.is_set():
            return self.seek_target

        return self.ambi_file.get_current_time()

    def get_info_text(self):
        """
        Return audio information as formatted text.
        """

        if not self.is_loaded:
            return "No audio loaded."

        return (
            f"Source: {self.source_name}\n"
            f"Sample rate: {self.fs} Hz\n"
            f"Channels: {self.ambi_file.get_num_channels()}\n"
            f"Duration: {self.get_duration():.2f} seconds\n"
            f"Current time: {self.get_current_time():.2f} seconds\n"
            f"Current volume: {self.gain:.2f}\n"
            f"Loop: {self.loop}"
        )

    def print_info(self):
        """
        Print audio information.
        """

        print(self.get_info_text())

    def set_loaded(self, loaded: bool):
        self.is_loaded = loaded
