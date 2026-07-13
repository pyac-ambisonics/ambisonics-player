# Utility functions for Ambisonics processing.
import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)
# the src/ folder as project_root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_HRTF_FILE = "resources/default_hrtf/FABIAN_HRIR_measured_HATO_0.sofa"

def resolve_path(path):
        if path is None:
            return None

        try:
            candidate = Path(path)
        except TypeError as error:
            print(f"Couldn't parse path: {path}. {error}")
            return None

        if candidate.is_absolute():
            return candidate
        
        # try to locate from program root
        root_candidate = _PROJECT_ROOT / candidate
        if root_candidate.exists():
            return root_candidate

        # try to locate relative to utils.py
        local_candidate = Path(__file__).resolve().parent / candidate
        if local_candidate.exists():
            return local_candidate

        # couldn't resolve
        return candidate

def ambix_channels_to_order(num_channels: int) -> int:
    """
    Return the ambisonics order corresponding to the number of given channels.

    For a given ambisonic order N the number of channels is (N + 1)^2.

    Parameters
    --------------
    num_channels : int
        Channel count to find ambisonics order for.

    Returns
    -------
    int
    """
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

def next_power_of_two(n: int) -> int:
    """
    Return the smallest power of two greater than or equal to n.

    Parameters
    --------------
    n : int
        Input integer (n >= 1).

    Returns
    ------------
    int
        Smallest power of two >= n.
    """
    # make use of bitshifts to quickly calculate the enxt power of two
    return 1 << (n - 1).bit_length()

    # calculates the nummer of channels corresponding to the ambisonics order
def order_to_channel_n(order: int):
    """
    Return the number of spherical-harmonic channels for the current order.

    For a given ambisonic order N the number of channels is (N + 1)^2.

    Parameters
    --------------
    order : int
        Ambisonics order to find channel count for.

    Returns
    -------
    int
        Number of spherical-harmonic/ambisonic channels.
    """

    return (order + 1)**2





