import numpy as np
import sounddevice as sd
import soundfile as sf


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

    def __init__(self):
        self.audio = None
        self.sample_rate = None
        self.volume = 1.0
        self.position = 0
        self.is_loaded = False
        self.source_name = None

        self.stream = None
        self.is_playing = False
        self.is_paused = False
        self.loop = False

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
        self.sample_rate = int(sample_rate)
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

    def audio_callback(self, outdata, frames, time, status):
        """
        This function is repeatedly called by sounddevice.
        It sends small audio blocks to the output device.
        """

        if status:
            print(status)

        if self.audio is None or self.is_paused:
            outdata[:] = np.zeros((frames, 2), dtype="float32")
            return

        end_position = self.position + frames

        if end_position <= len(self.audio):
            block = self.audio[self.position:end_position]
            self.position = end_position

        else:
            remaining = self.audio[self.position:]

            if self.loop:
                missing = frames - len(remaining)

                # If missing is larger than the full audio length, repeat safely
                if missing > len(self.audio):
                    repeats = int(np.ceil(missing / len(self.audio))) + 1
                    loop_source = np.tile(self.audio, (repeats, 1))
                    loop_part = loop_source[:missing]
                else:
                    loop_part = self.audio[:missing]

                block = np.vstack((remaining, loop_part))
                self.position = missing % len(self.audio)

            else:
                block = np.zeros((frames, 2), dtype="float32")
                block[:len(remaining)] = remaining
                self.position = len(self.audio)
                self.is_playing = False
                raise sd.CallbackStop

        outdata[:] = block * self.volume

    def play(self):
        """
        Start or resume playback.
        """

        if not self.is_loaded:
            print("No audio loaded.")
            return

        # If currently paused, resume
        if self.is_paused:
            self.resume()
            return

        # If already playing, do nothing
        if self.stream is not None and self.is_playing:
            print("Already playing.")
            return

        # If stream exists but is not playing, close it first
        if self.stream is not None:
            self.stop(reset_position=False)

        self.stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=2,
            dtype="float32",
            callback=self.audio_callback,
            blocksize=1024
        )

        self.is_playing = True
        self.is_paused = False

        print(f"Playing from {self.position / self.sample_rate:.2f} seconds...")
        self.stream.start()

    def pause(self):
        """
        Pause playback and keep current position.
        """

        if not self.is_playing:
            return

        self.is_paused = True
        self.is_playing = False
        print(f"Paused at {self.position / self.sample_rate:.2f} seconds.")

    def resume(self):
        """
        Resume playback from current position.
        """

        if not self.is_loaded:
            return

        self.is_paused = False

        if self.stream is None:
            self.play()
        else:
            self.is_playing = True
            print(f"Resumed at {self.position / self.sample_rate:.2f} seconds.")

    def stop(self, reset_position=True):
        """
        Stop playback.

        Parameters
        ----------
        reset_position : bool
            If True, reset playback position to the beginning.
        """

        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        self.is_playing = False
        self.is_paused = False

        if reset_position:
            self.position = 0

        print("Playback stopped.")

    def set_volume(self, volume: float):
        """
        Set playback volume between 0.0 and 1.0.
        """

        self.volume = max(0.0, min(float(volume), 1.0))
        print(f"Volume set to {self.volume:.2f}")

    def set_start_offset(self, seconds: float):
        """
        Set the playback start position in seconds.
        """

        if not self.is_loaded:
            print("No audio loaded.")
            return

        if seconds < 0:
            seconds = 0

        duration = self.get_duration()

        if seconds >= duration:
            print(f"Offset is too large. Audio duration is only {duration:.2f} seconds.")
            return

        self.position = int(seconds * self.sample_rate)
        print(f"Start offset set to {seconds:.2f} seconds.")

    def seek_to(self, seconds: float):
        """
        Jump to a specific playback position in seconds.
        This is mainly used by the GUI progress bar.
        """

        if not self.is_loaded:
            print("No audio loaded.")
            return

        duration = self.get_duration()

        if seconds < 0:
            seconds = 0

        if seconds > duration:
            seconds = duration

        self.position = int(seconds * self.sample_rate)
        print(f"Seeked to {seconds:.2f} seconds.")

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

        return len(self.audio) / self.sample_rate

    def get_current_time(self):
        """
        Return current playback time in seconds.
        """

        if not self.is_loaded:
            return 0.0

        return self.position / self.sample_rate

    def get_info_text(self):
        """
        Return audio information as formatted text.
        """

        if not self.is_loaded:
            return "No audio loaded."

        return (
            f"Source: {self.source_name}\n"
            f"Sample rate: {self.sample_rate} Hz\n"
            f"Channels: {self.audio.shape[1]}\n"
            f"Duration: {self.get_duration():.2f} seconds\n"
            f"Current time: {self.get_current_time():.2f} seconds\n"
            f"Current volume: {self.volume:.2f}\n"
            f"Loop: {self.loop}"
        )

    def print_info(self):
        """
        Print audio information.
        """

        print(self.get_info_text())