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
    """
    Memory-efficient Ambisonics audio loader.

    Features:
    - TRUE chunked streaming
    - Low memory usage
    - float32 processing
    - pyfar compatibility
    """

    def __init__(
        self,
        filepath: str,
        chunk_size: int = 2048,
        order: Optional[int] = None,
        trim_extra_channels: bool = True,
    ):

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
        """Validate that the loaded file is a valid AmbiX (ACN/SN3D) file.

        Checks performed:
        1. File format is WAV (AmbiX files are WAV containers)
        2. Channel count follows the (n+1)^2 pattern for Ambisonics
        3. WAV format tag is appropriate for multi-channel audio (>2 channels)
        """

        # Check 1: Must be a WAV file
        file_format = self.file.format
        if file_format != 'WAV':
            raise ValueError(
                f"Expected WAV format for AmbiX, got '{file_format}'. "
                f"Only AmbiX (ACN/SN3D) files in WAV containers are supported."
            )

        # Check 2: Channel count must follow (n+1)^2 for some integer n >= 0
        num_channels = self.num_channels
        order_candidate = int(np.sqrt(num_channels)) - 1

        if num_channels < 1 or (order_candidate + 1) ** 2 != num_channels:
            raise ValueError(
                f"File has {num_channels} channels, which does not match any "
                f"Ambisonics order. Expected (n+1)^2 channels (e.g., 1, 4, 9, "
                f"16, 25, 36, 49, 64). "
                f"This file may not be a valid AmbiX format file."
            )

        # Check 3: For multi-channel audio, verify format tag
        if num_channels > 2:
            subtype = self.file.subtype
            logger.debug(
                "WAV format: %s, subtype: %s, channels: %d",
                file_format, subtype, num_channels
            )

        logger.info(
            "AmbiX format validated: order %d candidate, %d channels",
            order_candidate, num_channels
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

        if num_samples is None:
            num_samples = self.chunk_size

        start_sample = max(
            0,
            min(start_sample, self.total_frames)
        )

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

        self._current_position = max(
            0,
            min(position_samples, self.total_frames - 1)
        )

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
            "Loaded: %s | Order: %d | Channels: %d -> %d | "
            "Duration: %.2fs | SR: %d | Frames: %d | Chunk: %d",
            self.filepath, self.order, self.num_channels,
            effective_channels, self.duration, self.samplerate,
            self.total_frames, self.chunk_size
        )

    # ==============================================================
    # Cleanup
    # ==============================================================

    def close(self):

        if self.file:
            self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __repr__(self):
        return (
            f"AmbisonicsFile(order={self.order}, "
            f"channels={self.get_num_channels()}, "
            f"duration={self.duration:.2f}s, "
            f"sr={self.samplerate})"
        )

    def __len__(self):
        return self.total_frames
