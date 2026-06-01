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
    - Empty channel detection
    - float32 processing
    - pyfar compatibility
    """

    def __init__(
        self,
        filepath: str,
        chunk_size: int = 2048,
        auto_detect_empty: bool = True,
        analysis_duration: float = 1.0,
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

        self._setup_channel_layout(
            auto_detect_empty,
            analysis_duration
        )

        self._print_load_info()

    # ==============================================================
    # Channel setup
    # ==============================================================

    def _setup_channel_layout(
        self,
        auto_detect_empty: bool,
        analysis_duration: float
    ):

        expected_channels = (self.order + 1) ** 2

        # Perfect match
        if self.num_channels == expected_channels:

            self._valid_channels = list(range(self.num_channels))
            self._empty_channels = []

            return

        print(
            f"Channel mismatch: "
            f"{self.num_channels} vs expected {expected_channels}"
        )

        # Try auto-detection
        if self.num_channels > expected_channels and auto_detect_empty:

            valid, empty = self._analyze_channels(
                analysis_duration
            )

            self._valid_channels = valid
            self._empty_channels = empty

            if len(valid) == expected_channels:

                print(
                    f"Auto-detected {len(valid)} valid channels"
                )

                print(
                    f"Ignoring {len(empty)} empty channels"
                )

                self._has_extra_channels = True

            else:

               print(
                    f"Detected {len(valid)} active channels "
                    f"but expected {expected_channels}"
                )

        else:

            print(
                f"File has {self.num_channels} channels "
                f"but expected {expected_channels}"
            )

    # ==============================================================
    # Analyze channels
    # ==============================================================

    def _analyze_channels(
        self,
        analysis_duration: float = 1.0
    ) -> Tuple[List[int], List[int]]:

        analysis_samples = min(
            int(analysis_duration * self.samplerate),
            self.total_frames
        )

        # IMPORTANT:
        # only read a small segment
        self.file.seek(0)

        segment = self.file.read(
            frames=analysis_samples,
            dtype='float32',
            always_2d=True
        )

        energies = []

        for ch in range(self.num_channels):

            rms = np.sqrt(
                np.mean(segment[:, ch] ** 2)
            )

            energies.append(rms)

        threshold = 1e-8

        valid_channels = [
            i for i, e in enumerate(energies)
            if e > threshold
        ]

        empty_channels = [
            i for i, e in enumerate(energies)
            if e <= threshold
        ]

        if empty_channels:

            print(
                f"Channel analysis: "
                f"{len(valid_channels)} active, "
                f"{len(empty_channels)} silent"
            )

            print(
                f"Empty channels: {empty_channels}"
            )

        return valid_channels, empty_channels

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
