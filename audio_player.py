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
        audio = np.asarray(audio, dtype="float32")

        if audio.ndim == 1:
            audio = np.column_stack((audio, audio))

        if audio.ndim == 2 and audio.shape[0] == 2 and audio.shape[1] > 2:
            audio = audio.T

        if audio.ndim != 2 or audio.shape[1] != 2:
            raise ValueError(
                f"Player expects stereo audio with shape (samples, 2), but got {audio.shape}"
            )

        max_value = np.max(np.abs(audio))
        if max_value > 1.0:
            audio = audio / max_value
            print("Audio was normalized to avoid clipping.")

        self.stop()

        self.audio = audio
        self.sample_rate = int(sample_rate)
        self.position = 0
        self.is_loaded = True
        self.source_name = source_name

        print("\nAudio loaded successfully.")
        self.print_info()

    def load_wav(self, file_path: str):
        audio, sample_rate = sf.read(file_path, dtype="float32")
        self.load_audio(audio, sample_rate, source_name=file_path)

    def load_pyfar_signal(self, signal):
        audio = signal.time
        sample_rate = signal.sampling_rate
        self.load_audio(audio, sample_rate, source_name="pyfar.Signal")

    def audio_callback(self, outdata, frames, time, status):
        """
        This function is called repeatedly by sounddevice.
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
                loop_part = self.audio[:missing]
                block = np.vstack((remaining, loop_part))
                self.position = missing
            else:
                block = np.zeros((frames, 2), dtype="float32")
                block[:len(remaining)] = remaining
                self.position = len(self.audio)
                self.is_playing = False
                raise sd.CallbackStop

        outdata[:] = block * self.volume

    def play(self):
        if not self.is_loaded:
            print("No audio loaded.")
            return

        if self.is_paused:
            self.resume()
            return

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
        if not self.is_playing:
            return

        self.is_paused = True
        self.is_playing = False
        print(f"Paused at {self.position / self.sample_rate:.2f} seconds.")

    def resume(self):
        if not self.is_loaded:
            return

        self.is_paused = False

        if self.stream is None:
            self.play()
        else:
            self.is_playing = True
            print(f"Resumed at {self.position / self.sample_rate:.2f} seconds.")

    def stop(self, reset_position=True):
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
        self.volume = max(0.0, min(float(volume), 1.0))
        print(f"Volume set to {self.volume:.2f}")

    def set_start_offset(self, seconds: float):
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

    def set_loop(self, enabled: bool):
        self.loop = bool(enabled)
        print(f"Loop set to {self.loop}")

    def get_duration(self):
        if not self.is_loaded:
            return 0.0
        return len(self.audio) / self.sample_rate

    def get_current_time(self):
        if not self.is_loaded:
            return 0.0
        return self.position / self.sample_rate

    def get_info_text(self):
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
        print(self.get_info_text())