import numpy as np
import pyfar as pf
from audio_player import AudioPlayer


def main():
    sample_rate = 48000
    duration = 3.0

    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    left = 0.2 * np.sin(2 * np.pi * 440 * t)
    right = 0.2 * np.sin(2 * np.pi * 660 * t)

    # pyfar commonly uses shape: (channels, samples)
    audio = np.vstack([left, right])

    signal = pf.Signal(audio, sample_rate)

    player = AudioPlayer()
    player.load_pyfar_signal(signal)
    player.play()

    input("Press Enter to stop playback...")
    player.stop()


if __name__ == "__main__":
    main()