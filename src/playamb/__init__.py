"""
playamb - Ambisonics player
===========================
Python library and executable program to decode and play ambisonics files as binaural output. Additional features include a GUI and dynamic headtracking.
"""

from playamb.audio.data.ambifile import AmbisonicsFile
from playamb.audio.engine.player import AudioPlayer
from playamb.audio.engine.hrtf import HRTF, Processing
from playamb.audio.engine.spherical import SphericalHarmonics
from playamb.audio.rotation.tracking import Orientation, OrientationState, DemoHeadTracker, HeadTracker
from playamb.audio.rotation.rotation import RotationMatrix
from playamb.gui.gui import AudioPlayerGUI
from playamb.gui.gui import Obj
from playamb.utils.utils import ambix_channels_to_order, next_power_of_two, order_to_channel_n, resolve_path

__version__ = "0.1.0"

__all__ = [
    "AmbisonicsFile",
    "AudioPlayer",
    "HRTF",
    "Processing",
    "SphericalHarmonics",
    "Orientation",
    "OrientationState",
    "DemoHeadTracker",
    "HeadTracker",
    "RotationMatrix",
    "AudioPlayerGUI",
    "Obj",
    "ambix_channels_to_order", 
    "next_power_of_two", 
    "order_to_channel_n", 
    "resolve_path"
    "__version__",
]