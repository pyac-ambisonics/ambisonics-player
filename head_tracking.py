import math
import threading
import time
from dataclasses import dataclass
import pyheadtracker as pht
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
        self._thread = None
        self._start_time = None

    def start(self):
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
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1)

    def is_running(self):
        return self._running

    def _tracking_loop(self):
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
        if not self._running:
            return self.orientation_state.get()

        elapsed = time.perf_counter() - self._start_time
        phase = (2.0 * math.pi * elapsed / self.period) * 0.3
        yaw = self.yaw_amplitude * math.sin(phase)
        pitch = self.pitch_amplitude * math.sin(phase * 0.5)
        return self.orientation_state.set(yaw, pitch, 0.0, source="demo")

class HeadTracker:
    """
    Provides headtracker support for Supperware Headtracker 1 and a few other head-tracking solutions via the package pythonheadtracker.
    More info on supported devices at https://pyheadtracker.readthedocs.io/en/latest/index.html 
    """

    def __init__(self, orientation_state):
        self.orientation_state = orientation_state
        self._running = False
        self._thread = None
        self.ht = None

    def is_available(self):
        return pht is not None

    # parameters are specific for the Supperware Headtracker 1 and should be changed for use with a different hardware
    def start(self):
        if self._running:
            return
        self.ht = pht.supperware.HeadTracker1(
            device_name="Head Tracker 1",
            device_name_output="Head Tracker 2",
            refresh_rate=25,
            compass_on=True,
            orient_format="ypr",
            gestures_on="off",
            chirality="preserve" # Change this later with UI switch!
        )
        self.ht.open()
        self.ht.zero()
        self._running = True
        self._thread = threading.Thread(
            target=self._tracking_loop,
            daemon=True
        )
        self._thread.start()

    def _tracking_loop(self):
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
        if self.ht is not None:
            self.ht.zero()

    def is_running(self):
        return self._running

if False:
    class OSCHeadTracker:
        """
        Provides headtracker support for hardware that supports sending data via OSC. Baseline for a potential future implementation
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
