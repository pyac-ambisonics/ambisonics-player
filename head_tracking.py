import math
import threading
import time
from dataclasses import dataclass
from pythonosc.udp_client import SimpleUDPClient
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import ThreadingOSCUDPServer
from shroom.utils.rotation_utils import wigner_d_matrix
from scipy.spatial.transform import Rotation


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
        phase = (2.0 * math.pi * elapsed / self.period) * 0.3
        yaw = self.yaw_amplitude * math.sin(phase)
        pitch = self.pitch_amplitude * math.sin(phase * 0.5)
        return self.orientation_state.set(yaw, pitch, 0.0, source="demo")


class OSCHeadTracker:
    """
    Provides headtracker support for hardware that supports sending data via OSC
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
        self.connected = False

    # start 
    def start(self):
        dispatcher = Dispatcher()
        dispatcher.map(
            "/[yaw,pitch,roll]",
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
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True
        )
        self._thread.start()

    # receiving
    def _handle_orientation(self, address, *args):#yaw, pitch, roll instead of *args
        print(address)
        print(args)
        """self.connected = True
        self.orientation_state.set(
            yaw,
            pitch,
            roll,
            source="osc"
        )"""

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
        self._thread = None