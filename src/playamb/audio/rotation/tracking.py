"""Head-orientation helpers for the ambisonics player.

This module exposes a small set of reusable components for reading, storing,
and generating head orientation data. It supports three modes:

- a deterministic demo tracker for testing and presentations
- a hardware tracker adapter built around the ``pythonheadtracker`` package
- an OSC-based tracker adapter for devices that send orientation via OSC

The public API is intentionally small and thread-safe: create an
``OrientationState`` instance and pass it to a tracker implementation
(``DemoHeadTracker``, ``HeadTracker`` or ``OSCHeadTracker``). Trackers will
write ``Orientation`` snapshots into the shared state which the GUI or audio
pipeline can read from concurrently.
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
    """Thread-safe storage for the latest head orientation.

    This helper provides a lock-protected container for the most recent
    ``Orientation`` snapshot. It is safe to call from multiple threads: one
    thread can update the state while another reads it.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._orientation = Orientation()

    def set(self, yaw=0.0, pitch=0.0, roll=0.0, source="manual") -> Orientation:
        """Store a new orientation and return it.

        The provided angle values are cast to ``float`` and stored as an
        ``Orientation`` dataclass. A thread lock ensures updates are atomic.

        Parameters
        ----------
        yaw : float, optional
            Yaw angle in degrees (rotation around the vertical axis). The
            default is ``0.0``.
        pitch : float, optional
            Pitch angle in degrees (rotation around the side-to-side axis).
            The default is ``0.0``.
        roll : float, optional
            Roll angle in degrees (rotation around the forward axis). The
            default is ``0.0``.
        source : str, optional
            Freeform string describing the source of the measurement, e.g.
            ``'hardware'``, ``'demo'`` or ``'osc'``. The default is
            ``'manual'``.

        Returns
        -------
        Orientation
            The ``Orientation`` instance that was stored.
        """

        orientation = Orientation(float(yaw), float(pitch), float(roll), source)
        with self._lock:
            self._orientation = orientation
        return orientation

    def get(self) -> Orientation:
        """Return the most recent orientation snapshot.

        Returns
        -------
        Orientation
            A frozen ``Orientation`` dataclass representing the last value set.
        """

        with self._lock:
            return self._orientation

class DemoHeadTracker:
    """Deterministic demo tracker used for testing and presentations.

    The tracker continuously writes a smooth sinusoidal yaw (and optional
    pitch) motion into an ``OrientationState`` instance. It is intended for
    use when hardware is unavailable or during demos.

    Parameters
    ----------
    orientation_state : OrientationState
        Shared state object where generated orientations are written.
    yaw_amplitude : float, optional
        Peak yaw angle in degrees. Default is ``60.0``.
    pitch_amplitude : float, optional
        Peak pitch angle in degrees. Default is ``0.0``.
    period : float, optional
        Period of the sinusoid in seconds. Values smaller than ``0.1`` are
        clamped. Default is ``6.0``.
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
        """Start the demo tracker's background thread.

        When started the tracker spawns a daemon thread that updates the
        ``OrientationState`` at a regular interval. Calling ``start`` when the
        tracker is already running has no effect.
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
        """Stop the demo tracker and join the background thread.

        The call requests the background thread to stop and waits up to one
        second for it to finish. If no thread is running the method returns
        immediately.
        """
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1)

    def is_running(self):
        """Return whether the demo tracker is currently running.

        Returns
        -------
        bool
            ``True`` when the background thread is active, otherwise ``False``.
        """
        return self._running

    def _tracking_loop(self):
        """Background loop that writes smooth orientation values.

        This private method runs inside the daemon thread started by
        ``start`` and periodically sets the current yaw/pitch into the
        ``OrientationState`` with ``source='demo'``.
        """

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
        """Return the current orientation computed by the demo tracker.

        If the tracker is running this computes the instantaneous yaw/pitch
        values based on the elapsed time and writes them into the
        ``OrientationState`` (returning the stored ``Orientation``). If the
        tracker is not running the current value from ``OrientationState`` is
        returned unchanged.

        Returns
        -------
        Orientation
            The most recent orientation snapshot produced by the demo tracker.
        """

        if not self._running:
            return self.orientation_state.get()

        elapsed = time.perf_counter() - self._start_time
        phase = (2.0 * math.pi * elapsed / self.period) * 0.3
        yaw = self.yaw_amplitude * math.sin(phase)
        pitch = self.pitch_amplitude * math.sin(phase * 0.5)
        return self.orientation_state.set(yaw, pitch, 0.0, source="demo")
    
    # zero function to match the API of the other trackers, does not do anything at all
    def zero(self):
        return

class HeadTracker:
    """Adapter for physical head-tracking hardware using ``pythonheadtracker``.

    The class opens a MIDI-based Supperware Head Tracker 1 device (via the
    ``pyheadtracker`` package), continuously reads orientation samples and
    writes them into the provided ``OrientationState`` instance.
    """

    def __init__(self, orientation_state: OrientationState):
        """Create a new hardware head tracker adapter.

        Parameters
        ----------
        orientation_state : OrientationState
            Shared state object where device orientations should be written.
        """
        self.orientation_state = orientation_state
        self._running = False
        self._thread = None
        self.ht = None

    def is_available(self):
        """Return whether compatible MIDI tracker devices are available.

        The method inspects the platform's MIDI device names reported by
        ``mido`` and looks for input and output devices containing the
        substring ``'Head Tracker'``.

        Returns
        -------
        bool
            ``True`` when both a matching input and output device are present,
            ``False`` otherwise.
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
        """Start hardware head tracking and spawn the reader thread.

        Parameters
        ----------
        in_device_name : str, optional
            MIDI input device name to open. If omitted the first matching
            device containing ``'Head Tracker'`` is used.
        out_device_name : str, optional
            MIDI output device name to open. If omitted the first matching
            device containing ``'Head Tracker'`` is used.
        refresh_rate : int, optional
            Tracker refresh rate in Hz passed to the underlying device class.
            Default is ``25``.
        chirality : str, optional
            Chirality option forwarded to the device constructor. Default is
            ``'preserve'``.

        Notes
        -----
        If tracking is already running this method is a no-op.
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
        """Background loop reading the device and publishing orientations.

        The loop calls the hardware adapter to read the orientation (in
        radians), converts the values to degrees and stores them in
        ``OrientationState`` with ``source='hardware'``. The loop exits when
        the tracker is stopped or when a terminal error occurs.
        """
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
        """Stop the hardware tracker and close the device connection.

        The method requests the background thread to stop, joins it with a
        one-second timeout and attempts to close the underlying hardware
        device if it was opened.
        """
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
        """Re-zero the device coordinate frame if supported.

        This calls the device's ``zero`` method when a device instance is
        present; otherwise the call is silently ignored.
        """
        if self.ht is not None:
            self.ht.zero()

    def is_running(self):
        """Return whether the hardware tracking thread is active.

        Returns
        -------
        bool
            ``True`` when the background tracker thread is alive, otherwise
            ``False``.
        """
        return self._running

class OSCHeadTracker:
    """Head tracker adapter for devices that send orientation via OSC.

    The adapter runs a :class:`pythonosc.osc_server.ThreadingOSCUDPServer`
    to receive orientation packets and provides a small UDP client for
    sending calibration/zeroing commands back to the device. It exposes the
    same simple methods as the other trackers: ``start``, ``stop``, ``zero``
    and ``is_available``.

    The implementation expects incoming OSC messages with three numeric
    arguments (yaw, pitch, roll) and maps them into degrees using the
    ``_osc2deg`` helper. Different hardware that sends other message formats
    will require adapting the mapping address used by ``start_server``.
    """

    def __init__(
        self,
        orientation_state,
        listen_host="127.0.0.1",
        listen_port=8000,
        write_host="127.0.0.1",
        write_port=9010,
    ):
        """Create an OSC-based head tracker adapter.

        Parameters
        ----------
        orientation_state : OrientationState
            Shared state object where received orientations are written.
        listen_host : str, optional
            Host/interface the OSC server listens on. Default is ``'127.0.0.1'``.
        listen_port : int, optional
            UDP port the OSC server listens on. Default is ``8000``.
        write_host : str, optional
            Host to which the adapter will send calibration commands. Default
            is ``'127.0.0.1'``.
        write_port : int, optional
            UDP port used for calibration commands. Default is ``9010``.
        """
        self.orientation_state = orientation_state
        self.listen_host = listen_host
        self.listen_port = listen_port
        self.write_host = write_host
        self.write_port = write_port
        self._server = None
        self._thread = None
        self._client = None
        self._server_running = False
        self._running = False
        self._last_packet_time = None

    def is_available(self, parsemsg="/yaw,pitch,roll"):
        """Return whether OSC data has been received recently.

        The method uses the timestamp of the last received packet to infer
        whether an OSC device is actively sending data. If no packet has been
        received yet the method returns ``False``.

        Parameters
        ----------
        parsemsg : str, optional
            The OSC address pattern used for parsing (unused by this simple
            availability check but kept for API compatibility). Default is
            ``'/yaw,pitch,roll'``.

        Returns
        -------
        bool
            ``True`` when a packet was received within the last 0.5 seconds,
            otherwise ``False``.
        """

        if self._last_packet_time is None:
            return False
        # if more than half a second since the last packet was received, assume device disconnected/server un => OSC unavailable
        return time.monotonic() - self._last_packet_time < 0.5

    # start listeing to OSC messages
    def start_server(self, parsemsg="/yaw,pitch,roll"):
        """Start the OSC server and a background thread to handle messages.

        The method registers a handler for the given address pattern
        (``parsemsg``) which expects three numeric arguments (yaw, pitch,
        roll). A small UDP client is also created to send calibration
        messages back to the device.

        Parameters
        ----------
        parsemsg : str, optional
            OSC address pattern to map to the orientation handler. Default is
            ``'/yaw,pitch,roll'``.
        """

        if self._server_running:
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
        self._server_running = True
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True
        )
        self._thread.start()

    # shutdown
    def stop_server(self):
        """Shut down the OSC server and clean up background thread and client.

        This method is safe to call multiple times; it will silently return
        if the server was not running.
        """
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=1)
        self._server = None
        self._running = False
        self._server_running = False
        self._thread = None

    # start tracking: restart server to update parsing address, set running status True
    def start(self, parsemsg="/yaw,pitch,roll"):
        """Restart the server (to update parsing address) and mark running.

        Parameters
        ----------
        parsemsg : str, optional
            OSC address pattern to register for incoming orientation messages.
        """
        self.stop_server()
        self.start_server(parsemsg)
        if self._server_running:
            self._running = True

    # stop tracking = set running status to False
    def stop(self):
        """Stop processing incoming OSC orientation messages.

        This does not necessarily shut down the OSC server socket; it only
        prevents newly received messages from being applied to the
        ``OrientationState`` until ``start`` is called again.
        """
        self._running = False

    # receiving
    def _handle_orientation(self, address, yaw, pitch, roll):
        """OSC callback that converts incoming values and updates the state.

        The handler receives three numeric arguments (yaw, pitch, roll) from
        the OSC message, converts them to degrees via ``_osc2deg`` and writes
        the resulting values into ``OrientationState`` with ``source='osc'``.
        The mapping and sign inversion are chosen to match the Bridghead App's
        output and may need adjustment for other devices.

        Parameters
        ----------
        address : str
            OSC address string for the incoming message.
        yaw, pitch, roll : float
            Raw numeric values extracted from the OSC message (device-specific
            range/format). They will be converted to degrees by
            ``_osc2deg``.
        """
        if self._running:
            self.orientation_state.set(
                - OSCHeadTracker._osc2deg(yaw),
                - OSCHeadTracker._osc2deg(pitch),
                OSCHeadTracker._osc2deg(roll),
                source="osc"
            )
        self._last_packet_time = time.monotonic()
    
    def _unknown_packet(self, address: str, *osc_arguments) -> None:
        """Default handler for unrecognised OSC messages.

        When an unrecognised address is received the adapter marks itself as
        not running and raises a ``LookupError`` to surface the problem when
        processing is active. The handler always updates the last-packet
        timestamp so availability checks reflect recent activity.

        Parameters
        ----------
        address : str
            OSC address string that could not be parsed.
        *osc_arguments : tuple
            Any additional arguments carried by the OSC message.
        """
        if self._running:
            self._running = False
            raise LookupError(f"Could not parse OSC message. Unrecognised OSC address: {address}\n")
        self._last_packet_time = time.monotonic()

    # zeroing
    def zero(self):
        if self._client is not None:
            self._client.send_message("/zero", 1)

    def is_running(self) -> bool:
        """Return whether the OSC hardware tracking is marked as running.

        Returns
        -------
        bool
            ``True`` when the tracker is in running state and will apply
            incoming messages to the orientation state.
        """
        return self._running

    @staticmethod
    def _osc2deg(x: float):
        """Convert a normalized OSC value to degrees.

        Many OSC-based trackers encode angles in a normalized 0..1 range. This
        helper maps that range to -180..+180 degrees using the simple linear
        transform ``(x - 0.5) * 360``.

        Parameters
        ----------
        x : float
            Input value in the unit interval (0..1).

        Returns
        -------
        float
            Angle in degrees in the range ``[-180, 180]``.
        """
        return (x - 0.5) * 360