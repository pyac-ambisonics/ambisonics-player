"""
Real-time binaural playback demo with basic transport control.

Plays an AmbiX file through the default audio output for a few seconds,
demonstrating play / volume / stop without the GUI:

    python examples/playback_demo.py input_ambix.wav --duration 10 --volume 0.8

Tip: no AmbiX material at hand?  Create some first:

    python examples/make_test_signal.py test_ambix.wav
    python examples/playback_demo.py test_ambix.wav
"""

import argparse
import time

from playamb import AmbisonicsFile, AudioPlayer, HRTF, SphericalHarmonics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Play an AmbiX file binaurally for a few seconds (no GUI)."
    )
    parser.add_argument("input", help="input AmbiX WAV file")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="seconds to play before stopping (default: 10)")
    parser.add_argument("--volume", type=float, default=0.8,
                        help="playback volume in [0, 1] (default: 0.8)")
    parser.add_argument("--chunk-size", type=int, default=2048,
                        help="streaming block size in samples (default: 2048)")
    parser.add_argument("--preprocess", choices=["MagLS", "LS"],
                        default="MagLS",
                        help="HRTF preprocessing (default: MagLS)")
    args = parser.parse_args()

    ambi = AmbisonicsFile(args.input, chunk_size=args.chunk_size)
    hrtf = HRTF()
    sh = SphericalHarmonics(
        hrtf=hrtf,
        sampling_rate=ambi.get_samplerate(),
        ambi_order=ambi.order,
        preprocess=args.preprocess,
    )
    player = AudioPlayer(ambi, sh)
    # AudioPlayer.__init__ leaves is_loaded False (only the load_* methods
    # set it), so play()/get_duration() would silently no-op without this.
    player.set_loaded(True)

    try:
        player.set_volume(args.volume)
        duration = min(args.duration, player.get_duration())
        print(f"Playing {args.input} for {duration:.1f} s "
              f"(order {ambi.order}, {ambi.get_samplerate()} Hz) ...")
        player.play()
        time.sleep(duration)
    finally:
        player.stop()
        player.close()  # also closes sh internally
        ambi.close()

    print("Done.")


if __name__ == "__main__":
    main()
