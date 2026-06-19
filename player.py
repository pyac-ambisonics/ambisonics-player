import queue
import threading
import sounddevice as sd
import numpy as np
from spherical import SphericalHarmonics
from ambisonics_file_English import AmbisonicsFile
import time

class BinauralPlayer:

    def __init__(self, ambi_file: AmbisonicsFile, sh: SphericalHarmonics, gain=1.0):
        
        # Check channel count by comparing the channel shape
        # we know the channel shape for sh_hrir is (2, channels)
        _, sh_hrir_ch, _ = sh.hrirs_nm.shape
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

        # some control flags for handling stopping and pausing
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.looping = False

        # States for overlap-add (must be reset on seek)
        self.overlap_buffer = None 
        # overlap add L
        self.sh_length = None
        # overlap add M
        self.block_size = ambi_file.get_chunk_size()
        # overlap add N
        self.N = None

        # keeping track of blocks
        self.current_block = None
        self.current_block_pos = 0

        # Thread and stream objects
        self.processing_thread = None
        self.stream = None

    # reset all variables in case of seek or stop
    def _reset_process_variables(self):
        # Reset the file reader to beginning
        #self.ambi_file.reset_position()

        # reset internal state 
        # current impulse response length M
        self.sh_length = self.sh.get_IR_length()
        # block size L
        self.block_size = self.ambi_file.get_chunk_size()

        # compute N >= M + L - 1
        self.N = self.next_power_of_two(self.sh_length + self.block_size - 1)

        # update our sh fft coefficients once (and on each rotation update)
        self.sh.update_hrirs_fft(self.N)

        # allocate buffers
        self.overlap_buffer = np.zeros((self.sh_length - 1, 2), dtype=np.float32)

    def _process_loop(self):
        """Processing thread: reads chunks, processes, and puts into queue."""
        
        print("Start processing thread.")

        # ensures we only wait a fraction of the time we generally need
        #put_timeout = (self.block_size / self.fs) * 0.1

        # while we are not stopping
        while not self.stop_event.is_set():
            # Pause handling: block if paused but not stopped
            if self.pause_event.is_set():
                # Block indefinitely until pause_event is cleared (resume or stop)
                self.pause_event.wait()
            
                # If stop was triggered, exit the outer loop
                if self.stop_event.is_set():
                    break
            
            # Read next chunk
            chunk, end_of_file = self.ambi_file.get_next_chunk()
            if chunk is None:
                break
            
            # processes one chunk and return a stereo block, 
            # updating the overlap buffer internally.
            processed_block = self._process_chunk(chunk)

            # Put into queue
            try:
                self.audio_queue.put(processed_block, timeout=1)
            except queue.Full:
                # If queue is full, skip this block (or handle gracefully)
                print("Warning!!! Queue is full! Dropping this block")

            # at end of file add the overlap buffer a final time
            if end_of_file:
                # add the remaining overlapp buffer to the queue
                self.audio_queue.put(self.overlap_buffer.copy())

                #check for looping
                if self.looping:
                    # resets processor and will read chunk 0 next
                    self._reset_process_variables()
                else:
                    # queue None-item as flag that playback has ended
                    self.audio_queue.put(None)
                    break

    # process a signle chunk of data, performing an overlap-add algorithm
    def _process_chunk(self, chunk):
        """
        Process a single Ambisonics chunk with overlap-add.
        This method should maintain self.overlap_buffer and update it.
        For now, we simulate with random data – replace with your actual processing.
        """
        
        # TAKES AROUND 0.0008 seconds to run or faster
        
        # update our block size
        # using self.blocksize instead. but this may lead to problem? check
        block_size, *_ = chunk.shape

        # apply hrtf
        stereo = self.sh.apply_hrtf_fast(chunk, self.N)
        # add overlap to stereo output
        stereo[:self.sh_length-1] += self.overlap_buffer
        # save new overlap buffer
        self.overlap_buffer[:] = stereo[block_size:block_size + self.sh_length - 1]

    
        return stereo[:block_size]
    
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

            
    def _audio_callback(self, outdata, frames, time, status):
        """sounddevice callback – outputs from queue or silence if paused."""

        # debug if something went wrong last callback
        if status:
            print(f"Audio callback status: {status}")

        # check pasue status
        if self.pause_event.is_set():
            # fill with silence if paused
            outdata.fill(0)
            return

        # get our next audio block
        try:
            data = self.audio_queue.get_nowait()
        except queue.Empty:
            # return silence if something went wrong and our queue is empty
            outdata.fill(0)
            return

        # we set data = None in our _process_loop() if we reached the end of file to signal playback has ended
        if data is None:
            # End of file
            print("Filling with 0")
            outdata.fill(0)
            # tell sounddevice to cleanup and stop the callback
            raise sd.CallbackStop
        
        # Copy data to output
        # pad with zeroes if  data is shorter than the buffersize
        if len(data) < len(outdata):
            outdata[:len(data)] = data
            outdata[len(data):] = 0
            raise sd.CallbackStop
        else:
            outdata[:] = data * self.gain

    def play(self):
        """Start playback from current position."""
        print("Start playback.")
        # clean up old strema before next playback
        if self.stream is not None:
            self.stop()

        # reset stop and pause flags
        self.stop_event.clear()
        self.pause_event.clear()

        # reset all processing variables before starting the processing thread. 
        # also ensures block size is calculated before we initialize the stream
        self._reset_process_variables()

        # Start processing thread
        self.processing_thread = threading.Thread(target=self._process_loop)
        self.processing_thread.start()

        # Start audio stream
        self.stream = sd.OutputStream(
            samplerate=self.fs,
            channels=self.n_channels,
            callback=self._audio_callback_robust,
            # N, not blocksize! since blocksize gets convoluted, it gets bigger!
            # let sounddevice choose blocksize
            # blocksize=self.N,
            finished_callback=self._on_stream_finished
        )
        self.stream.start()

    def _on_stream_finished(self):
        """Called when stream stops (e.g., at end of file)."""
        print("Stream finished.")
        # set the stop flag
        self.stop_event.set()

    def pause(self):
        """Pause playback."""
        print("Pause playback.")
        # set the pause flag
        self.pause_event.set()

    def resume(self):
        """Resume playback."""
        print("Resume playback.")
        # cleat the pause flag
        self.pause_event.clear()

    def seek_to_sample(self, position_samples):
        """
        Seek to a new position (in samples) in the file.
        This stops playback, resets the reader and processor, and restarts.
        """
        print(f"Seeking to sample {position_samples}")
        # Stop current playback and processing
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.stop_event.set()
        if self.processing_thread is not None:
            self.processing_thread.join(timeout=1.0)

        # Reset processor state (overlap buffer, etc.)
        self._reset_process_variables()

        # Reset the file reader to the new position
        self.ambi_file.seek_to_position(position_samples)

        # Clear the audio queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

        # Restart playback
        self.play()

    def seek_to_time(self, position_time):
        """
        Seek to a new position (in time) in the file.
        This stops playback, resets the reader and processor, and restarts.
        """
        print(f"Seeking to time {position_time}")
        # Stop current playback and processing
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.stop_event.set()
        if self.processing_thread is not None:
            self.processing_thread.join(timeout=1.0)

        # Reset processor state (overlap buffer, etc.)
        self._reset_process_variables()

        # Reset the file reader to the new position
        self.ambi_file.seek_to_time(position_time)

        # Clear the audio queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

        # Restart playback
        self.play()

    def stop(self):
        """Stop playback and clean up."""
        print("Stop playback.")
        # set the stop flag
        self.stop_event.set()

        # clear the pause flag, just in case it was blocking our processing thread
        self.pause_event.clear()

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

        # reset the stop flag
        self.pause_event.clear()

    def set_volume(self, volume: float):
        """
        Set playback volume between 0.0 and 1.0.
        """

        self.gain = max(0.0, min(float(volume), 1.0))
        print(f"Volume set to {self.gain:.2f}")

        
    def set_loop(self, enabled: bool):
        """
        Enable or disable loop playback.
        """

        self.loop = bool(enabled)
        print(f"Loop set to {self.loop}")

        
    def next_power_of_two(self, n: int) -> int:
        """
        Return the smallest power of two greater than or equal to n.

        Parameters
        --------------
        n : int
            Input integer (n >= 1).

        Returns
        ------------
        int
            Smallest power of two >= n.
        """
        # make use of bitshifts to quickly calculate the enxt power of two
        return 1 << (n - 1).bit_length()