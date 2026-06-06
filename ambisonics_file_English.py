# -*- coding: utf-8 -*-
"""
Created on Thu May 21 15:55:05 2026

@author: lenovo
"""

"""
Ambisonics file handling with pyfar.Signal format.
Supports automatic empty channel detection and real-time chunked reading.
"""

import numpy as np
import pyfar as pf
import soundfile as sf

from typing import Tuple, Optional, List
from utils import ambix_channels_to_order


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
        self.chunk_size = max(1024, min(8192, chunk_size))

        # ==========================================================
        # Use soundfile for STREAMING instead of loading entire file
        # ==========================================================

        self.file = sf.SoundFile(filepath)

        self.samplerate = self.file.samplerate
        self.num_channels = self.file.channels
        self.total_frames = len(self.file)

        self.duration = self.total_frames / self.samplerate

        # ==========================================================
        # Determine Ambisonics order
        # ==========================================================

        if order is not None:
            self.order = order
            print(f"Using user-specified order: {self.order}")
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
    # Channel setup
    # ==============================================================

    def _setup_channel_layout(self, trim_extra_channels: bool):
        expected_channels = (self.order + 1) ** 2

        if self.num_channels == expected_channels:
            self._valid_channels = list(range(self.num_channels))
            self._empty_channels = []
            return

        print(f"Channel mismatch: {self.num_channels} vs expected {expected_channels}")

        if self.num_channels > expected_channels:
            if trim_extra_channels:
                self._valid_channels = list(range(expected_channels))
                self._empty_channels = list(range(expected_channels, self.num_channels))
                self._has_extra_channels = True
                print(f"Keeping first {expected_channels} channels, discarding {len(self._empty_channels)} trailing channels")
            else:
                self._valid_channels = list(range(self.num_channels))
                self._empty_channels = []
                print("Warning: Keeping all channels, decoding may fail")
        else:
            raise ValueError(
            f"File has only {self.num_channels} channels, but Ambisonics order {self.order} "
            f"requires at least {expected_channels} channels. The file is incomplete or not "
            f"a valid Ambisonics file of the specified order.")

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

                print(
                    f"Filtering channels: "
                    f"{self.num_channels} -> "
                    f"{audio_data.shape[1]}"
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

        return pf.Signal(
            chunk,
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

        if self._valid_channels:
            return len(self._valid_channels)

        return self.num_channels

    def get_total_channels(self) -> int:
        return self.num_channels

    def get_chunk_size(self) -> int:
        return self.chunk_size

    def set_chunk_size(self, chunk_size: int):

        self.chunk_size = max(
            1024,
            min(8192, chunk_size)
        )

        print(
            f"Chunk size changed to "
            f"{self.chunk_size}"
        )

    # ==============================================================
    # Validation
    # ==============================================================

    def is_valid(self) -> bool:

        effective = (
            len(self._valid_channels)
            if self._valid_channels
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

        print("\n" + "=" * 50)

        print(f"Loaded: {self.filepath}")

        print(f"Order: {self.order}")

        print(
            f"Channels: "
            f"{self.num_channels} -> "
            f"{effective_channels}"
        )

        print(
            f"Duration: "
            f"{self.duration:.2f} sec"
        )

        print(
            f"Sample rate: "
            f"{self.samplerate}"
        )

        print(
            f"Frames: "
            f"{self.total_frames}"
        )

        print(
            f"Chunk size: "
            f"{self.chunk_size}"
        )

        print("=" * 50 + "\n")

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
