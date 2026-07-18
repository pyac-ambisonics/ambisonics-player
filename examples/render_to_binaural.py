"""
Render an AmbiX Ambisonics file to a binaural stereo WAV (offline, streamed).

Uses the chunk-based pipeline, so memory stays O(chunk) even for very long
or high-order recordings.  No audio device is needed.

    python examples/render_to_binaural.py input_ambix.wav output_binaural.wav
    python examples/render_to_binaural.py in.wav out.wav --headphone "Sennheiser HD650"
    python examples/render_to_binaural.py in.wav out.wav --hrtf my_hrtf.sofa --order 3
"""

import argparse

from playamb import (
    AmbisonicsFile,
    HRTF,
    SphericalHarmonics,
    render_to_binaural_file,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline binaural rendering of an AmbiX file (streamed)."
    )
    parser.add_argument("input", help="input AmbiX WAV file")
    parser.add_argument("output", help="output binaural stereo WAV file")
    parser.add_argument("--order", type=int, default=None,
                        help="Ambisonics order (default: auto-detect)")
    parser.add_argument("--chunk-size", type=int, default=2048,
                        help="streaming block size in samples (default: 2048)")
    parser.add_argument("--hrtf", default=None, metavar="SOFA",
                        help="custom HRTF SOFA file (default: bundled FABIAN)")
    parser.add_argument("--headphone", default=None, metavar="NAME",
                        help='headphone filter, e.g. "Sennheiser HD650" '
                             "(see src/resources/Headphones)")
    parser.add_argument("--preprocess", choices=["MagLS", "LS"],
                        default="MagLS",
                        help="HRTF preprocessing (default: MagLS)")
    parser.add_argument("--gain", type=float, default=1.0,
                        help="output gain in [0, 1] (default: 1.0)")
    args = parser.parse_args()

    ambi = AmbisonicsFile(args.input, chunk_size=args.chunk_size,
                          order=args.order)
    hrtf = HRTF(path=args.hrtf)
    if args.headphone:
        hrtf.load_hp_filter(args.headphone)

    sh = SphericalHarmonics(
        hrtf=hrtf,
        sampling_rate=ambi.get_samplerate(),
        ambi_order=ambi.order,
        preprocess=args.preprocess,
    )

    def progress(frames_done: int, total_frames: int) -> None:
        percent = 100.0 * frames_done / total_frames if total_frames else 100.0
        print(f"\r  {percent:5.1f}%  ({frames_done}/{total_frames} frames)",
              end="", flush=True)

    try:
        frames = render_to_binaural_file(
            ambi, sh, args.output,
            gain=args.gain,
            progress_callback=progress,
        )
    finally:
        sh.close()
        ambi.close()

    print(f"\nWrote {args.output} ({frames} stereo frames)")


if __name__ == "__main__":
    main()
