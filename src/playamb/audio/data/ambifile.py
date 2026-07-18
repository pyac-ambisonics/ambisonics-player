# -*- coding: utf-8 -*-
"""
Created on Thu May 21 15:55:05 2026

@author: lenovo
"""

"""
Ambisonics file handling with pyfar.Signal format.
Supports automatic empty channel detection and real-time chunked reading.
"""

import logging
import numpy as np
import pyfar as pf
import soundfile as sf
from typing import Tuple, Optional, List

from playamb.utils.utils import ambix_channels_to_order, order_to_channel_n, next_power_of_two

logger = logging.getLogger(__name__)


class AmbisonicsFile:
    """
    Memory-efficient Ambisonics audio loader.

    Features:
    - TRUE chunked streaming
    - Low memory usage
    - float32 processing
    - pyfar compatibility
    """

    _SUPPORTED_FORMATS = ("ambix", "fuma")
    _SUPPORTED_NORMALIZATIONS = ("SN3D", "N3D")

    def __init__(
        self,
        filepath: str,
        chunk_size: int = 2048,
        order: Optional[int] = None,
        trim_extra_channels: bool = True,
        format: str = "ambix",
        normalization: str = "SN3D",
    ):
        """
        Open an Ambisonics WAV file for chunked streaming.

        Parameters
        ----------
        filepath : str
            Path to the Ambisonics WAV file.
        chunk_size : int
            Number of samples per streaming chunk (clamped to [32, 8192],
            rounded up to the nearest power of two).  Default 2048.
        order : int or None
            Ambisonics order to decode.  When ``None`` (default), the order
            is auto-detected from the channel count via
            ``ambix_channels_to_order``.
        trim_extra_channels : bool
            When True (default), trailing channels beyond the expected
            ``(order+1)^2`` are discarded.  Set to False to keep all
            channels (may break decoding if extra channels are present).
        format : str
            Ambisonics format — ``"ambix"`` (ACN/SN3D, default) or
            ``"fuma"`` (WXY/Furse-Malham).
        normalization : str
            Channel normalisation — ``"SN3D"`` (default) or ``"N3D"``.
        """

        if format not in self._SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format '{format}'. "
                f"Supported: {', '.join(self._SUPPORTED_FORMATS)}."
            )
        self.format = format

        if normalization not in self._SUPPORTED_NORMALIZATIONS:
            raise ValueError(
                f"Unsupported normalization '{normalization}'. "
                f"Supported: {', '.join(self._SUPPORTED_NORMALIZATIONS)}."
            )
        self.normalization = normalization

        self.filepath = filepath
        self.chunk_size = self._normalize_chunk_size(chunk_size)

        # ==========================================================
        # Use soundfile for STREAMING instead of loading entire file
        # ==========================================================

        self.file = sf.SoundFile(filepath)

        self.samplerate = self.file.samplerate
        self.num_channels = self.file.channels
        self.total_frames = len(self.file)

        self.duration = self.total_frames / self.samplerate

        # ==========================================================
        # Validate AmbiX format
        # ==========================================================

        self._validate_ambix_format()

        # ==========================================================
        # Determine Ambisonics order
        # ==========================================================

        if order is not None:
            self.order = order
            logger.info("Using user-specified order: %d", self.order)
        else:
            self.order = ambix_channels_to_order(self.num_channels)

        # ==========================================================
        # Normalization (SN3D → N3D conversion scales)
        # ==========================================================

        self._n3d_scales = self._build_normalization_scales()

        # ==========================================================
        # Stream state
        # ==========================================================

        self._current_position = 0

        # ==========================================================
        # Channel management
        # ==========================================================

        self._valid_channels: Optional[List[int]] = None
        self._empty_channels: Optional[List[int]] = None

        self._has_extra_channels = False
        self._channel_warning_printed = False

        # ==========================================================
        # Analyze channels
        # ==========================================================

        self._setup_channel_layout(trim_extra_channels)
        self._print_load_info()

    # ==============================================================
    # AmbiX format validation
    # ==============================================================

    def _validate_ambix_format(self):
        """Validate that the loaded file is a valid Ambisonics audio file.

        Checks performed:
        1. File container is WAV
        2. Channel count >= (n+1)^2 for some valid order n

        The format (ambix ACN/SN3D vs FuMa WXY/Furse-Malham) is set by the
        user via the ``format`` parameter — it cannot be auto-detected.
        """

        # Check 1: Must be a WAV file
        if self.file.format != 'WAV':
            raise ValueError(
                f"Expected WAV container for Ambisonics, got '{self.file.format}'. "
                f"Ambisonics files (ambix / FuMa) use WAV containers."
            )

        # Check 2: Channel count must be >= (n+1)^2 for some valid order n.
        # Extra trailing channels (e.g., spot mics) are trimmed later by
        # _setup_channel_layout — this check only ensures the file has enough
        # channels to cover at least one valid Ambisonics order.
        num_channels = self.num_channels
        order_candidate = int(np.sqrt(num_channels)) - 1
        if order_candidate < 0:
            order_candidate = 0

        if (order_candidate + 1) ** 2 > num_channels:
            raise ValueError(
                f"File has {num_channels} channels, which is fewer than the "
                f"minimum 1 channel required for Ambisonics (order 0). "
                f"This file is not a valid Ambisonics format file."
            )

        # Warm the user when channel count is not an exact (n+1)^2 match.
        # E.g., a regular 2ch stereo WAV would reach here as "order 0 + 1 extra".
        expected = (order_candidate + 1) ** 2
        if num_channels != expected:
            logger.warning(
                "Channel count %d is not an exact (n+1)^2 value (nearest: %d). "
                "Extra channels will be trimmed if trim_extra_channels=True. "
                "If this is a regular multi-channel WAV (not Ambisonics), decoding "
                "results will be incorrect.",
                num_channels, expected,
            )

        logger.info(
            "Ambisonics format validated: %s, %d channels, nearest order %d",
            self.format, num_channels, order_candidate,
        )

    def set_order(self, order: int):
        """
        Change the Ambisonics playback order.

        If the requested *order* exceeds what the current file supports
        (requires ``(order+1)^2`` channels), it is silently clamped to the
        highest valid order.  The channel layout is re-evaluated and extra
        trailing channels are trimmed.

        Parameters
        ----------
        order : int
            Desired Ambisonics order (non-negative).
        """
        if order < 0:
            raise ValueError("Order must be non-negative.")
        
        # make order lower if requested order is too high
        if self.num_channels < order_to_channel_n(order):
            # set to highest possible order instead
            order = ambix_channels_to_order(self.num_channels)
            print(f"Requested order too big for this file. Using highest possible order of {order} instead.")

        self.order = order
        self._validate_ambix_format()
        self._setup_channel_layout(trim_extra_channels=True)

    # ==============================================================
    # Channel setup
    # ==============================================================

    def _setup_channel_layout(self, trim_extra_channels: bool):
        """Determine which source channels map to the current Ambisonics order.

        When the file has more channels than ``(order+1)^2`` and
        *trim_extra_channels* is True, the extra trailing channels are
        flagged as empty and excluded from reads.  When the file has too few
        channels a ``ValueError`` is raised.

        Parameters
        ----------
        trim_extra_channels : bool
            Whether to discard trailing channels beyond the expected count.
        """
        expected_channels = (self.order + 1) ** 2

        if self.num_channels == expected_channels:
            self._valid_channels = list(range(self.num_channels))
            self._empty_channels = []
            return

        logger.info("Channel mismatch: %d vs expected %d", self.num_channels, expected_channels)

        if self.num_channels > expected_channels:
            if trim_extra_channels:
                self._valid_channels = list(range(expected_channels))
                self._empty_channels = list(range(expected_channels, self.num_channels))
                self._has_extra_channels = True
                logger.info("Keeping first %d channels, discarding %d trailing channels",
                            expected_channels, len(self._empty_channels))
            else:
                self._valid_channels = list(range(self.num_channels))
                self._empty_channels = []
                logger.warning("Keeping all channels, decoding may fail")
        else:
            max_order = int(np.sqrt(self.num_channels)) - 1
            raise ValueError(
                f"File has only {self.num_channels} channels, but Ambisonics order "
                f"{self.order} requires {expected_channels} channels. "
                f"The maximum order supported by this file is {max_order}. "
                f"Please reduce the playback order or use a file with more channels.")

    # ==============================================================
    # Frame reading
    # ==============================================================

    def get_frames(
        self,
        start_sample: int,
        num_samples: Optional[int] = None
    ) -> np.ndarray:
        """Read a block of audio frames from the underlying WAV file.

        Only channels flagged as valid (see ``_setup_channel_layout``) are
        returned.  When the normalisation is ``"N3D"``, per-ACN SN3D→N3D
        scaling is applied in-place.

        Parameters
        ----------
        start_sample : int
            Zero-based sample index to seek to before reading.
        num_samples : int or None
            Number of samples to read.  When ``None``, defaults to the
            current chunk size.

        Returns
        -------
        data : ndarray of shape ``(num_samples, num_valid_channels)``, float32
        """
        if self.file is None:
            raise RuntimeError("Cannot read frames: file is closed.")

        if num_samples is None:
            num_samples = self.chunk_size

        if start_sample < 0:
            logger.warning(
                "Negative start_sample %d clamped to 0", start_sample
            )
            start_sample = 0
        elif start_sample > self.total_frames:
            logger.warning(
                "start_sample %d exceeds total_frames %d, clamped",
                start_sample, self.total_frames,
            )
            start_sample = self.total_frames

        self.file.seek(start_sample)

        audio_data = self.file.read(
            frames=num_samples,
            dtype='float32',
            always_2d=True
        )

        # shape:
        # (samples, channels)

        if self._valid_channels is not None:

            audio_data = audio_data[
                :,
                self._valid_channels
            ]

            if (
                self._has_extra_channels
                and not self._channel_warning_printed
            ):

                logger.debug(
                    "Filtering channels: %d -> %d",
                    self.num_channels,
                    audio_data.shape[1]
                )

                self._channel_warning_printed = True

        # Apply SN3D → N3D normalization if requested
        if self.normalization == "N3D":
            num_acn = len(self._n3d_scales)
            audio_data[:, :num_acn] *= self._n3d_scales[np.newaxis, :]

        return audio_data

    # ==============================================================
    # Streaming read
    # ==============================================================

    def get_next_chunk(
        self
    ) -> Tuple[Optional[np.ndarray], bool]:
        """Read the next streaming chunk from the current file position.

        Advances the internal position pointer by the number of samples
        actually read.  If the chunk has more channels than expected for
        the current order, extra channels are trimmed silently.

        Returns
        -------
        chunk : ndarray of shape ``(samples, channels)`` or None
            The next chunk of audio data, or ``None`` when the end of the
            file has been reached.
        end_of_file : bool
            ``True`` if the returned chunk is the last one in the file.
        """
        if self._current_position >= self.total_frames:
            return None, True

        chunk = self.get_frames(
            self._current_position,
            self.chunk_size
        )

        expected_channels = order_to_channel_n(self.order)

        # make sure the chunk has the correct channel count for our order
        if chunk.shape[1] > expected_channels:
            chunk = chunk[:, :expected_channels]

        actual_samples = chunk.shape[0]

        self._current_position += actual_samples

        end_of_file = (
            self._current_position >= self.total_frames
        )

        return chunk, end_of_file

    # ==============================================================
    # pyfar compatibility
    # ==============================================================

    def get_signal_chunk(
        self,
        start_sample: int,
        num_samples: Optional[int] = None
    ) -> pf.Signal:
        """Read a block of audio and return it as a ``pyfar.Signal``.

        Parameters
        ----------
        start_sample : int
            Zero-based sample index to seek to before reading.
        num_samples : int or None
            Number of samples to read.  When ``None``, defaults to the
            current chunk size.

        Returns
        -------
        signal : pyfar.Signal
            Signal in time domain with shape ``(channels, samples)``.
        """

        chunk = self.get_frames(
            start_sample,
            num_samples
        )

        # get_frames() returns (samples, channels)
        # pyfar.Signal expects (channels, samples) — last dim is time
        return pf.Signal(
            chunk.T,
            self.samplerate,
            domain='time',
            fft_norm='none'
        )

    # ==============================================================
    # Seeking
    # ==============================================================

    def seek_to_position(self, position_samples: int):
        """Move the streaming read pointer to an absolute sample index.

        The value is clamped to ``[0, total_frames]``.

        Parameters
        ----------
        position_samples : int
            Target position in samples.
        """

        if position_samples < 0:
            logger.warning("Negative position %d clamped to 0", position_samples)
            position_samples = 0
        elif position_samples > self.total_frames:
            logger.warning(
                "Position %d exceeds total_frames %d, clamped",
                position_samples, self.total_frames,
            )
            position_samples = self.total_frames

        self._current_position = position_samples

    def seek_to_time(self, time_seconds: float):
        """Move the streaming read pointer to a time in seconds.

        Wraps ``seek_to_position`` after converting seconds to samples.

        Parameters
        ----------
        time_seconds : float
            Target time in seconds.
        """

        position = int(
            time_seconds * self.samplerate
        )

        self.seek_to_position(position)

    def reset_position(self):
        """Reset the streaming read pointer to the beginning of the file."""
        self._current_position = 0

    def get_current_position(self) -> int:
        """Return the current stream position in samples."""
        return self._current_position

    def get_current_time(self) -> float:
        """Return the current stream position in seconds."""
        return self._current_position / self.samplerate

    # ==============================================================
    # Info
    # ==============================================================

    def get_duration(self) -> float:
        """Return the total audio duration in seconds."""
        return self.duration

    def get_samplerate(self) -> int:
        """Return the sample rate in Hz."""
        return self.samplerate

    def get_order(self) -> int:
        """Return the current Ambisonics order."""
        return self.order

    def get_format(self) -> str:
        """Return the Ambisonics format: 'ambix' (ACN/SN3D) or 'fuma' (WXY/Furse-Malham)."""
        return self.format

    def get_num_channels(self) -> int:
        """Return the number of valid (non-empty) audio channels.

        When extra channels have been trimmed this is ``(order+1)^2``;
        otherwise it equals ``get_total_channels``.
        """

        if self._valid_channels is not None:
            return len(self._valid_channels)

        return self.num_channels

    def get_total_channels(self) -> int:
        """Return the total number of channels in the source file."""
        return self.num_channels

    def get_chunk_size(self) -> int:
        """Return the current streaming chunk size in samples."""
        return self.chunk_size

    def get_normalization(self) -> str:
        """Return the normalization scheme: 'SN3D' or 'N3D'."""
        return self.normalization

    def _build_normalization_scales(self) -> np.ndarray:
        """Build scale factors for SN3D → N3D conversion.

        For ACN channel c, order n = floor(sqrt(c)), scale = sqrt(2n + 1).
        Returns an array of 1.0 if normalization is SN3D (no conversion needed).
        """
        num_acn = (self.order + 1) ** 2
        scales = np.ones(num_acn, dtype=np.float32)
        if self.normalization == "N3D":
            for c in range(num_acn):
                n = int(np.sqrt(c))
                scales[c] = np.sqrt(2 * n + 1)
        return scales

    @staticmethod
    def _normalize_chunk_size(chunk_size: int) -> int:
        """Clamp *chunk_size* to [32, 8192] and round up to the nearest power of two."""
        # Clamp to valid range
        chunk_size = max(32, min(8192, chunk_size))

        # Round up to nearest power of two: 1 << (n - 1).bit_length()
        if chunk_size > 1:
            chunk_size = next_power_of_two(chunk_size)

        return chunk_size

    def set_chunk_size(self, chunk_size: int):
        """Update the streaming chunk size.

        The value is normalised via ``_normalize_chunk_size``, so it will
        be clamped and rounded to a power of two.

        Parameters
        ----------
        chunk_size : int
            Desired chunk size in samples.
        """
        self.chunk_size = self._normalize_chunk_size(chunk_size)

        logger.info(
            "Chunk size changed to %d",
            self.chunk_size
        )

    # ==============================================================
    # Validation
    # ==============================================================

    def is_valid(self) -> bool:
        """Return True if the file has exactly ``(order+1)^2`` valid channels."""

        effective = (
            len(self._valid_channels)
            if self._valid_channels is not None
            else self.num_channels
        )

        expected = (self.order + 1) ** 2

        return effective == expected

    # ==============================================================
    # Debug
    # ==============================================================

    def get_channel_info(self) -> dict:
        """Return a dictionary summarising the channel layout.

        Keys: total_channels, valid_channels, empty_channels,
        effective_channels, expected_channels, has_extra_channels.
        """
        return {
            'total_channels': self.num_channels,
            'valid_channels': self._valid_channels,
            'empty_channels': self._empty_channels,
            'effective_channels': self.get_num_channels(),
            'expected_channels': (self.order + 1) ** 2,
            'has_extra_channels': self._has_extra_channels,
        }

    # ==============================================================
    # Print info
    # ==============================================================

    def _print_load_info(self):
        """Log a one-line summary of the loaded file (format, order, channels, duration, SR, chunk size)."""
        effective_channels = self.get_num_channels()
        logger.info(
            "Loaded: %s | Format: %s | Norm: %s | Order: %d | Channels: %d -> %d | "
            "Duration: %.2fs | SR: %d | Frames: %d | Chunk: %d",
            self.filepath, self.format, self.normalization, self.order, self.num_channels,
            effective_channels, self.duration, self.samplerate,
            self.total_frames, self.chunk_size
        )

    # ==============================================================
    # Cleanup
    # ==============================================================

    def close(self):
        """Close the underlying soundfile handle and release resources."""
        if self.file:
            self.file.close()
            self.file = None

    def __enter__(self):
        """Context-manager entry — returns self."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context-manager exit — calls ``close()``."""
        self.close()

    def __repr__(self):
        """Return a compact string representation for debugging."""
        return (
            f"AmbisonicsFile(format={self.format}, norm={self.normalization}, order={self.order}, "
            f"channels={self.get_num_channels()}, "
            f"duration={self.duration:.2f}s, "
            f"sr={self.samplerate})"
        )

    def __len__(self):
        """Return the total number of frames in the file."""
        return self.total_frames
