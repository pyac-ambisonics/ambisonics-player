# Utility functions for Ambisonics processing.


import logging
import numpy as np

logger = logging.getLogger(__name__)


def ambix_channels_to_order(num_channels: int) -> int:


    if num_channels < 1:
        raise ValueError(
            f"Channel count {num_channels} is too low for Ambisonics. "
            f"Minimum required: 1 channel (order 0)."
        )

    # Find the largest order where (order+1)^2 <= num_channels
    order = int(np.sqrt(num_channels)) - 1

    if order > 7:
        raise ValueError(
            f"Channel count {num_channels} would require order {order}, "
            f"but maximum supported order is 7 (64 channels)."
        )

    logger.debug(
        "Detected Ambisonics order %d from %d channels (min %d)",
        order, num_channels, (order + 1) ** 2,
    )
    return order





