import math
import threading
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class Orientation:
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    source: str = "off"


class OrientationState:
    """Thread-safe storage for the latest head orientation."""

    def __init__(self):
        self._lock = threading.Lock()
        self._orientation = Orientation()

    def set(self, yaw=0.0, pitch=0.0, roll=0.0, source="manual"):
        orientation = Orientation(float(yaw), float(pitch), float(roll), source)
        with self._lock:
            self._orientation = orientation
        return orientation

    def get(self):
        with self._lock:
            return self._orientation


class DemoHeadTracker:
    """
    Deterministic demo tracker for presentations.

    It generates a smooth yaw motion without requiring external hardware. The GUI
    can sample this object regularly and use the same values for the visualizer
    and the rotation hook.
    """

    def __init__(self, orientation_state, yaw_amplitude=60.0, pitch_amplitude=0.0, period=6.0):
        self.orientation_state = orientation_state
        self.yaw_amplitude = float(yaw_amplitude)
        self.pitch_amplitude = float(pitch_amplitude)
        self.period = max(float(period), 0.1)
        self._running = False
        self._start_time = None

    def start(self):
        self._running = True
        self._start_time = time.perf_counter()
        return self.sample()

    def stop(self):
        self._running = False
        return self.orientation_state.set(0.0, 0.0, 0.0, source="off")

    def is_running(self):
        return self._running

    def sample(self):
        if not self._running:
            return self.orientation_state.get()

        elapsed = time.perf_counter() - self._start_time
        phase = 2.0 * math.pi * elapsed / self.period
        yaw = self.yaw_amplitude * math.sin(phase)
        pitch = self.pitch_amplitude * math.sin(phase * 0.5)
        return self.orientation_state.set(yaw, pitch, 0.0, source="demo")


class OscHeadTracker:
    """
    Optional OSC receiver skeleton.

    The demo GUI does not start this class yet, but it documents the intended
    extension path: receive `/headtracker yaw pitch roll` and write it into the
    same OrientationState used by the demo tracker.
    """

    def __init__(self, orientation_state, host="127.0.0.1", port=9000):
        self.orientation_state = orientation_state
        self.host = host
        self.port = int(port)
        self._server = None
        self._thread = None

    def start(self):
        try:
            from pythonosc.dispatcher import Dispatcher
            from pythonosc.osc_server import ThreadingOSCUDPServer
        except ModuleNotFoundError as error:
            raise RuntimeError("python-osc is required for OSC head tracking.") from error

        dispatcher = Dispatcher()
        dispatcher.map("/headtracker", self._handle_orientation)
        self._server = ThreadingOSCUDPServer((self.host, self.port), dispatcher)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self):
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        self._thread = None

    def _handle_orientation(self, _address, yaw, pitch=0.0, roll=0.0):
        self.orientation_state.set(yaw, pitch, roll, source="osc")
