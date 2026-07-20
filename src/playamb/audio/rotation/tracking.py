"""Head-orientation helpers for the ambisonics player.

This module exposes a small set of reusable components for reading, storing,
and generating head orientation data. It supports three modes:

* a deterministic demo tracker for testing and presentations, and
* a hardware tracker adapter built around the `pythonheadtracker` package.
* a hardware tracker adapter designed to communicate over OSC.
"""

import math
import threading
import time
import mido
import pyheadtracker as pht
from pythonosc.dispatcher import Dispatcher
from pythonosc.udp_client import SimpleUDPClient
from pythonosc.osc_server import ThreadingOSCUDPServer
from dataclasses import dataclass


@dataclass(frozen=True)
class Orientation:
    """
    Snapshot of a head pose in degrees. Angles are saved in the (yaw, pitch, roll) format,
    and follow the sign conventions of a right-handed coordinate system.

    Attributes
    ----------
        yaw: float, optional
            Rotation around the vertical axis. The default is 0.
        pitch: float, optional
            Rotation around the side-to-side axis. The default is 0.
        roll: float, optional
            Rotation around the forward axis. The default is 0.
        source: {'off', 'demo', 'hardware', 'osc'}
            Origin of the measurement. The default is `'off'`
    """

    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    source: str = "off"


class OrientationState:
    """Thread-safe storage for the latest head orientation."""
    def __init__(self):
        self._lock = threading.Lock()
        self._orientation = Orientation()

    def set(self, yaw=0.0, pitch=0.0, roll=0.0, source="manual") -> Orientation:
        """
        Store a new orientation and return it as an Orientation() instance.
        
        Returns
        -------
        orientation : `':py:class:~playamb.audio.rotation.tracking.Orientation'`
            Orientation object containing the current orientation.
        """

        orientation = Orientation(float(yaw), float(pitch), float(roll), source)
        with self._lock:
            self._orientation = orientation
        return orientation

    def get(self) -> Orientation:
        """Return the most recent orientation snapshot.
        
        Returns
        -------
        orientation : `':py:class:~playamb.audio.rotation.tracking.Orientation'`
            Orientation object containing the orientation that has just been set.
        """

        with self._lock:
            return self._orientation

class DemoHeadTracker:
    """
    Deterministic demo tracker for presentations.

    It generates a smooth yaw motion without requiring external hardware. The GUI
    can sample this object regularly and use the same values for the visualizer
    and the rotation hook.
    """

    def __init__(self, orientation_state: OrientationState, yaw_amplitude=60.0, pitch_amplitude=0.0, period=6.0):
        self.orientation_state = orientation_state
        self.yaw_amplitude = float(yaw_amplitude)
        self.pitch_amplitude = float(pitch_amplitude)
        self.period = max(float(period), 0.1)
        self._running = False
        self._thread = None
        self._start_time = None

    def start(self):
        """Start the demo tracker loop.

        The tracker moves smoothly in yaw and can optionally add pitch motion.
        """

        if self._running:
            return
        self._running = True
        self._start_time = time.perf_counter()
        self._thread = threading.Thread(
            target=self._tracking_loop,
            daemon=True
        )
        self._thread.start()

    def stop(self):
        """Stop the demo tracker and the associated thread."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1)

    def is_running(self):
        """Return whether the demo tracker is currently running."""
        return self._running

    def _tracking_loop(self):
        """Continuously update the orientation state with a smooth sinusoid."""

        while self._running:
            elapsed = time.perf_counter() - self._start_time
            phase = (2 * math.pi * elapsed / self.period) * 0.3
            yaw = self.yaw_amplitude * math.sin(phase)
            pitch = self.pitch_amplitude * math.sin(phase * 0.5)
            self.orientation_state.set(
                yaw,
                pitch,
                0.0,
                source="demo"
            )
            time.sleep(0.04)

    def sample(self):
        """Return the current demo orientation without waiting for the loop."""

        if not self._running:
            return self.orientation_state.get()

        elapsed = time.perf_counter() - self._start_time
        phase = (2.0 * math.pi * elapsed / self.period) * 0.3
        yaw = self.yaw_amplitude * math.sin(phase)
        pitch = self.pitch_amplitude * math.sin(phase * 0.5)
        return self.orientation_state.set(yaw, pitch, 0.0, source="demo")

class HeadTracker:
    """Adapter for physical head-tracking hardware using `pythonheadtracker`.

    The class opens a MIDI-based Supperware Headtracker 1 device, continually
    reads orientation data, and writes it into an `OrientationState` instance.
    """

    def __init__(self, orientation_state):
        self.orientation_state = orientation_state
        self._running = False
        self._thread = None
        self.ht = None

    def is_available(self):
        """Return True when matching MIDI input/output tracker devices are present.

        Returns
        -------
        bool
            `:bool:True` when MIDI input/output tracker devices matching "Head Tracker" are available,
            `:bool:False` otherwise.
        """

        return (any("Head Tracker" in MIDIdevice for MIDIdevice in mido.get_input_names()) 
            and any("Head Tracker" in MIDIdevice for MIDIdevice in mido.get_output_names())
        )

    def start(
        self,
        in_device_name: str=None,
        out_device_name: str=None,
        refresh_rate=25,
        chirality="preserve"
    ):
        """
        Start hardware head tracking.

        If device names are omitted, they are resolved automatically from the
        available MIDI input/output devices.
        """

        if self._running:
            return

        if not in_device_name:
            in_device_name=next(MIDIdevice for MIDIdevice in mido.get_input_names() if "Head Tracker" in MIDIdevice)
        if not out_device_name:
            out_device_name=next(MIDIdevice for MIDIdevice in mido.get_output_names() if "Head Tracker" in MIDIdevice)

        self.ht = pht.supperware.HeadTracker1(
            device_name=in_device_name,
            device_name_output=out_device_name,
            refresh_rate=refresh_rate,
            compass_on=True,
            orient_format="ypr",
            gestures_on="off",
            chirality=chirality
        )

        self.ht.open()
        self.ht.zero()

        self._running = True
        self._thread = threading.Thread(
            target=self._tracking_loop,
            daemon=True,
        )
        self._thread.start()

    def _tracking_loop(self):
        """Read device orientation in a background thread and publish it."""
        while self._running:
            try:
                orientation = pht.utils.rad2deg(self.ht.read_orientation())
                
            except EOFError:
                break
            except Exception as e:
                print(e)
                break
                
            self.orientation_state.set(
                orientation[0],
                orientation[1],
                orientation[2],
                source="hardware"
            )

        self._running = False

    def stop(self):
        """Stop the hardware tracker and close the device connection."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1)
        if self.ht is not None:
            try:
                self.ht.close()
            except Exception as error:
                print(error)
            finally:
                self.ht = None

    def zero(self):
        """Re-zero the device coordinate frame if the tracker supports it."""
        if self.ht is not None:
            self.ht.zero()

    def is_running(self):
        """Return whether the hardware tracking thread is active."""
        return self._running

class OSCHeadTracker:
    """
    Provides headtracker support for hardware that supports sending data via OSC.
    This class has only been tested with the Supperware Head Tracker 1 sending
    OSC messages via the proprietary Bridghead App. Use with different hardware
    and OSC interfaces will likely require providing different addresses/ports.
    """

    def __init__(
        self,
        orientation_state,
        listen_host="127.0.0.1",
        listen_port=8000,
        write_host="127.0.0.1",
        write_port=9010,
    ):
        self.orientation_state = orientation_state
        self.listen_host = listen_host
        self.listen_port = listen_port
        self.write_host = write_host
        self.write_port = write_port
        self._server = None
        self._thread = None
        self._client = None
        self._running = False
        self._last_packet_time = None

    def is_available(self, parsemsg="/yaw,pitch,roll"):
        """
        Return 
        
        Returns
        -------
        bool
            :bool:`True` when compatible OSC devices are connected and are sending the requested parsing message,
            :bool:`False` otherwise.
        """

        if False:
            try:
                self.start(parsemsg)
            except Exception as error:
                print(error)
            rtn = self._receiving_packets
            self.stop()
            return rtn
        if self._last_packet_time is None:
            return False

        return time.monotonic() - self._last_packet_time < 0.5

    # start
    def start(self, parsemsg="/yaw,pitch,roll"):
        """
        Start an OSC threading server to receive orientation data from the tracker and set up a client to send zeroing messages.
        Start a new thread to handle the incoming messages.
        """

        if self._running:
            return
        self.zero()
        dispatcher = Dispatcher()
        dispatcher.map(
            parsemsg,
            self._handle_orientation
        )
        dispatcher.set_default_handler(
            self._unknown_packet
        )
        self._server = ThreadingOSCUDPServer(
            (self.listen_host, self.listen_port),
            dispatcher
        )
        self._client = SimpleUDPClient(
            self.write_host,
            self.write_port
        )
        self._running = True
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True
        )
        self._thread.start()

    # receiving
    def _handle_orientation(self, address, yaw, pitch, roll):
        """
        Callback function. Set the current orientation to the values that have just been read.
        Set `'osc'` as the source. This function will likely require modification for the use
        with a different hardware, since the angle data coming from the test hardware isn't natively
        in degrees.
        """
        #self._running = True
        self.orientation_state.set(
            - OSCHeadTracker._osc2deg(yaw),
            - OSCHeadTracker._osc2deg(pitch),
            OSCHeadTracker._osc2deg(roll),
            source="osc"
        )
        self._last_packet_time = time.monotonic()

    # zeroing
    def zero(self):
        if self._client is not None:
            self._client.send_message("/zero", 1)

    # shutdown
    def stop(self):
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=1)
        self._server = None
        self._running = False
        self._thread = None

    def is_running(self) -> bool:
        """Return whether the OSC hardware tracking thread is active."""
        return self._running
    
    def _unknown_packet(self, address: str, *osc_arguments: List[Any]) -> None:
        """Default callback function for unrecognised OSC addresses. Print the address to stdout and set availability flag to false."""
        print(f"Unrecognised OSC address: {address}")
        self._last_packet_time = time.monotonic()

    @staticmethod
    def _osc2deg(x: float):
        return (x - 0.5) * 360