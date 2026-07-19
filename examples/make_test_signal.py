"""
Generate a synthetic AmbiX test file (seeded noise).

Lets you try the other examples without any real Ambisonics material:

    python examples/make_test_signal.py test_ambix.wav --order 1 --duration 5
"""

import argparse

import numpy as np
import soundfile as sf

from playamb import order_to_channel_n

DEFAULT_AMPLITUDE = 0.3  # keeps the decoded output well below clipping


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic AmbiX WAV (seeded noise) for testing."
    )
    parser.add_argument("output", nargs="?", default="test_ambix.wav",
                        help="output WAV path (default: test_ambix.wav)")
    parser.add_argument("--order", type=int, default=1,
                        help="Ambisonics order (default: 1 -> 4 channels)")
    parser.add_argument("--duration", type=float, default=5.0,
                        help="duration in seconds (default: 5.0)")
    parser.add_argument("--samplerate", type=int, default=48000,
                        help="sample rate in Hz (default: 48000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed (default: 42)")
    args = parser.parse_args()

    channels = order_to_channel_n(args.order)
    frames = int(args.duration * args.samplerate)

    rng = np.random.default_rng(args.seed)
    data = (DEFAULT_AMPLITUDE * rng.standard_normal((frames, channels)))
    sf.write(args.output, data.astype(np.float32), args.samplerate,
             subtype="FLOAT")

    print(f"Wrote {args.output}: order {args.order}, {channels} channels, "
          f"{frames} frames @ {args.samplerate} Hz")


if __name__ == "__main__":
    main()
