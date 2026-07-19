"""
Streaming binaural renderer (GitHub issue #6).

Fully decodes an AmbiX Ambisonics file to a binaural stereo WAV using the
chunk-based streaming pipeline, so memory stays O(chunk) regardless of file
length.  The DSP path is identical to real-time playback: chunks from
``AmbisonicsFile.get_next_chunk()`` are convolved via
``SphericalHarmonics.apply_hrtf_fast`` and stitched with overlap-add.

Overlap-add derivation (each step with its source):

1. Linear convolution length:  len(x * h) = L + M - 1
   Source: definition of discrete linear convolution.
2. Block decomposition: x = sum_k shift(x_k, o_k)  =>
   x * h = sum_k shift(x_k * h, o_k)
   Source: linearity and time-invariance of convolution (superposition).
   Each block's convolution therefore has an M-1 sample tail that must be
   added onto the head of the next block's output (the "overlap-add").
3. FFT size N >= L + M - 1 makes circular convolution equal linear
   convolution.
   Source: circular convolution theorem (zero-padding condition).

The processor below is a direct port of the proven real-time implementation
in ``AudioPlayer._reset_process_variables`` / ``_process_chunk``
(player.py) so that offline rendering and playback share the same math.
"""

from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
import soundfile as sf

from playamb.audio.data.ambifile import AmbisonicsFile
from playamb.audio.engine.spherical import SphericalHarmonics
from playamb.utils.utils import next_power_of_two


class OverlapAddProcessor:
    """
    Chunk-wise binaural decoder with overlap-add state.

    Ported from ``AudioPlayer._reset_process_variables`` (buffer setup) and
    ``AudioPlayer._process_chunk`` (per-block overlap-add) so the offline
    path stays numerically identical to real-time playback.

    Notes
    -----
    Constructing this processor calls ``sh.update_hrirs_fft(N)``, which
    overwrites the FFT cache shared with any ``AudioPlayer`` using the same
    ``SphericalHarmonics`` instance.  Do not render while that player is
    actively playing (a subsequent ``play()`` resets its own state and is
    unaffected).
    """

    def __init__(self, sh: SphericalHarmonics, chunk_size: int):
        self.sh = sh
        # impulse response length M
        self.ir_length = sh.get_IR_length()
        # FFT length N >= M + L - 1 (circular == linear condition)
        self.fft_size = next_power_of_two(self.ir_length + chunk_size - 1)
        sh.update_hrirs_fft(self.fft_size)
        # carry buffer for the M-1 tail samples of each block
        self.overlap_buffer = np.zeros((self.ir_length - 1, 2), dtype=np.float32)

    def process(self, chunk: np.ndarray) -> np.ndarray:
        """
        Decode one Ambisonics chunk and return its clean stereo samples.

        Parameters
        ----------
        chunk : np.ndarray
            Ambisonics block, shape ``(block_size, n_channels)``.  Only the
            last chunk of a stream may be shorter than the nominal size.

        Returns
        -------
        np.ndarray
            Stereo block of shape ``(block_size, 2)``.  The convolution
            tail is kept in the internal overlap buffer.
        """
        block_size = chunk.shape[0]
        stereo = self.sh.apply_hrtf_chunk(chunk, self.fft_size)
        # add the tail carried over from the previous block
        stereo[: self.ir_length - 1] += self.overlap_buffer
        # save the new tail for the next block
        self.overlap_buffer[:] = stereo[block_size : block_size + self.ir_length - 1]
        return stereo[:block_size]

    def flush(self) -> np.ndarray:
        """Return the final M-1 tail samples after the last chunk."""
        return self.overlap_buffer.copy()


def render_to_binaural_file(
    ambi_file: AmbisonicsFile,
    sh: SphericalHarmonics,
    out_path: Union[str, Path],
    *,
    gain: float = 1.0,
    subtype: str = "FLOAT",
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> int:
    """
    Decode an entire AmbiX file to a binaural stereo WAV, streamed in chunks.

    Memory usage is O(chunk): one FFT buffer plus one (M-1, 2) overlap
    buffer, independent of file length.  Output length equals
    ``total_frames + sh.get_IR_length() - 1``, matching a full-file
    ``mode='full'`` convolution (``SphericalHarmonics.apply_hrtf``).

    Parameters
    ----------
    ambi_file : AmbisonicsFile
        Input file.  Its chunk size drives the streaming block length.  The
        read position is saved on entry and restored on exit.
    sh : SphericalHarmonics
        Decoder whose channel count must match the file.  Rendering uses the
        decoder's *current* rotation state; equivalence with ``apply_hrtf``
        (which ignores rotation) holds only for identity rotation.
    out_path : str or Path
        Output WAV path.  Overwritten if it exists.
    gain : float, optional
        Post-decode gain in ``[0., 1.]``, applied per block.  Default 1.0.
    subtype : str, optional
        libsndfile subtype for the output.  Default ``"FLOAT"`` (32-bit
        float) to preserve decode precision; note plain ``soundfile.write``
        would default to ``"PCM_16"`` for WAV.
    progress_callback : callable, optional
        Called once per chunk as ``progress_callback(frames_read,
        total_frames)``.

    Returns
    -------
    int
        Number of stereo frames written (0 for an empty input file).

    Raises
    ------
    ValueError
        If the file channel count does not match the decoder.
    AttributeError
        If ``gain`` is outside ``[0., 1.]``.
    """
    # validate before opening the output file so failures leave no artifact
    # (checks mirror AudioPlayer.__init__)
    _, sh_channels, _ = sh.hrir_nm.shape
    if ambi_file.get_num_channels() != sh_channels:
        raise ValueError("Channel counts must match (e.g. 16 for 3rd order).")
    if gain > 1.0 or gain < 0.0:
        raise AttributeError("The gain must be in range [0., 1.].")

    total_frames = ambi_file.total_frames
    saved_position = ambi_file.get_current_position()
    frames_written = 0

    try:
        ambi_file.reset_position()
        ola = OverlapAddProcessor(sh, ambi_file.get_chunk_size())

        with sf.SoundFile(
            str(out_path),
            mode="w",
            samplerate=ambi_file.get_samplerate(),
            channels=2,
            subtype=subtype,
            format="WAV",
        ) as out:
            frames_read = 0
            while True:
                chunk, end_of_file = ambi_file.get_next_chunk()
                if chunk is None:
                    # empty file: nothing was convolved, so there is no tail
                    break

                block = ola.process(chunk)
                if gain != 1.0:
                    block = block * gain
                out.write(block)
                frames_written += block.shape[0]
                frames_read += chunk.shape[0]

                if progress_callback is not None:
                    progress_callback(frames_read, total_frames)

                if end_of_file:
                    # flush the final convolution tail (matches mode='full')
                    tail = ola.flush()
                    if gain != 1.0:
                        tail = tail * gain
                    out.write(tail)
                    frames_written += tail.shape[0]
                    break
    finally:
        # rendering must not disturb the caller's read position
        ambi_file.seek_to_position(saved_position)

    return frames_written
