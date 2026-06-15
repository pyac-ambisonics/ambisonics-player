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
from utils import ambix_channels_to_order

logger = logging.getLogger(__name__)


class AmbisonicsFile:


    _SUPPORTED_FORMATS = ("ambix", "fuma")

    def __init__(
        self,
        filepath: str,
        chunk_size: int = 2048,
        order: Optional[int] = None,
        trim_extra_channels: bool = True,
        format: str = "ambix",
    ):

        if format not in self._SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format '{format}'. "
                f"Supported: {', '.join(self._SUPPORTED_FORMATS)}."
            )
        self.format = format

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

    # ==============================================================
    # Channel setup
    # ==============================================================

    def _setup_channel_layout(self, trim_extra_channels: bool):
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

        return audio_data

    # ==============================================================
    # Streaming read
    # ==============================================================

    def get_next_chunk(
        self
    ) -> Tuple[Optional[np.ndarray], bool]:

        if self._current_position >= self.total_frames:
            return None, True

        chunk = self.get_frames(
            self._current_position,
            self.chunk_size
        )

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

        position = int(
            time_seconds * self.samplerate
        )

        self.seek_to_position(position)

    def reset_position(self):
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
        return self.duration

    def get_samplerate(self) -> int:
        return self.samplerate

    def get_order(self) -> int:
        return self.order

    def get_format(self) -> str:
        """Return the Ambisonics format: 'ambix' (ACN/SN3D) or 'fuma' (WXY/Furse-Malham)."""
        return self.format

    def get_num_channels(self) -> int:

        if self._valid_channels is not None:
            return len(self._valid_channels)

        return self.num_channels

    def get_total_channels(self) -> int:
        return self.num_channels

    def get_chunk_size(self) -> int:
        return self.chunk_size

    @staticmethod
    def _normalize_chunk_size(chunk_size: int) -> int:
        """Round chunk_size up to the nearest power of two, clamped to [32, 8192].

        Small chunks → lower latency for real-time playback (VR head-tracking etc.)
        Must be a power of two for FFT and audio processing compatibility.
        """
        # Clamp to valid range
        chunk_size = max(32, min(8192, chunk_size))

        # Round up to nearest power of two: 1 << (n - 1).bit_length()
        if chunk_size > 1:
            chunk_size = 1 << (chunk_size - 1).bit_length()

        return chunk_size

    def set_chunk_size(self, chunk_size: int):
        self.chunk_size = self._normalize_chunk_size(chunk_size)

        logger.info(
            "Chunk size changed to %d",
            self.chunk_size
        )

    # ==============================================================
    # Validation
    # ==============================================================

    def is_valid(self) -> bool:

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
        effective_channels = self.get_num_channels()
        logger.info(
            "Loaded: %s | Format: %s | Order: %d | Channels: %d -> %d | "
            "Duration: %.2fs | SR: %d | Frames: %d | Chunk: %d",
            self.filepath, self.format, self.order, self.num_channels,
            effective_channels, self.duration, self.samplerate,
            self.total_frames, self.chunk_size
        )

    # ==============================================================
    # Cleanup
    # ==============================================================

    def close(self):

        if self.file:
            self.file.close()
            self.file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __repr__(self):
        return (
            f"AmbisonicsFile(format={self.format}, order={self.order}, "
            f"channels={self.get_num_channels()}, "
            f"duration={self.duration:.2f}s, "
            f"sr={self.samplerate})"
        )

    def __len__(self):
        return self.total_frames
