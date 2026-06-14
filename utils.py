# Utility functions for Ambisonics processing.


import logging
import numpy as np

logger = logging.getLogger(__name__)


def ambix_channels_to_order(num_channels: int) -> int:

    # Ambisonics channels = (order + 1)^2  →  order = sqrt(channels) - 1
    order = int(np.sqrt(num_channels)) - 1

    if order < 0:
        raise ValueError(
            f"Channel count {num_channels} is too low for Ambisonics. "
            f"Minimum required: 1 channel (order 0)."
        )

    if order > 7:
        raise ValueError(
            f"Channel count {num_channels} would require order {order}, "
            f"but maximum supported order is 7 (64 channels)."
        )

    expected = (order + 1) ** 2
    if num_channels != expected:
        raise ValueError(
            f"Channel count {num_channels} does not match any Ambisonics order. "
            f"Expected (n+1)^2 channels (e.g., 1, 4, 9, 16, 25, 36, 49, 64)."
        )

    logger.debug("Detected Ambisonics order %d from %d channels", order, num_channels)
    return order





