import math
import os
import threading
import traceback
import tkinter as tk
import mido
from pathlib import Path
from queue import Queue
from tkinter import filedialog, messagebox
from tkinter import ttk

from ambisonics_file_English import AmbisonicsFile
from audio_player import AudioPlayer
from head_tracking import HeadTracker, DemoHeadTracker, OrientationState
from hrtf import HRTF
from spherical import SphericalHarmonics


class AudioPlayerGUI:
    """
    GUI for the Ambisonics Player.

    Main supported workflow:
    AmbiX WAV -> AmbisonicsFile -> HRTF/SphericalHarmonics -> AudioPlayer

    The old standalone binaural WAV path is intentionally disabled in this
    GUI because the current AudioPlayer backend is an AmbiX streaming player.
    """

    DEFAULT_HRTF_FILE = "FABIAN_HRIR_measured_HATO_0.sofa"

    def __init__(self):
        self.app_dir = Path(__file__).resolve().parent
        self.player = None
        self.is_loading = False
        self.is_dragging_progress = False

        self._backend_load_queue = Queue()

        self.root = tk.Tk()
        self.root.title("Ambisonics Player")
        self.root.geometry("1100x820")
        self.root.minsize(900, 560)

        # File / pipeline state
        self.selected_file = tk.StringVar(value="No input loaded")
        self.input_type = tk.StringVar(value="Input type: none")
        self.output_type = tk.StringVar(value="Output: none")
        self.pipeline_status = tk.StringVar(value="Pipeline: waiting for input")
        self.loading_text = tk.StringVar(value="")
        self.playback_status = tk.StringVar(value="Status: No audio loaded")

        # Playback state
        self.time_text = tk.StringVar(value="00:00.00 / 00:00.00")
        self.volume_value = tk.DoubleVar(value=1.0)
        self.volume_display_value = tk.StringVar(value="1.00")
        self.offset_value = tk.StringVar(value="0")
        self.loop_value = tk.BooleanVar(value=False)
        self.progress_value = tk.DoubleVar(value=0.0)

        # Decoder settings
        self.order_value = tk.StringVar(value="Auto")
        self.block_size_value = tk.StringVar(value="1024")
        self.hrtf_path_value = tk.StringVar(value="Default FABIAN HRTF")
        self.headphone_value = tk.StringVar(value="Diffuse Field Equalization")
        self.loaded_settings_text = tk.StringVar(value="Loaded settings: none")
        self.decoder_note = tk.StringVar(
            value="Order, block size, HRTF, and headphone filter are applied when loading a file."
        )
        self.update_event = threading.Event()

        # Rotation and head tracking demo state
        self.orientation_state = OrientationState()
        self.demo_tracker = DemoHeadTracker(self.orientation_state)
        self.head_tracker = HeadTracker(self.orientation_state)
        self.rotation_tracker = self.demo_tracker
        self.head_tracker_devices = []
        self.midi_error = ""
        self.yaw_value = tk.DoubleVar(value=0.0)
        self.pitch_value = tk.DoubleVar(value=0.0)
        self.roll_value = tk.DoubleVar(value=0.0)
        self.rotation_text = tk.StringVar(value="Rotation: yaw 0.0, pitch 0.0, roll 0.0")
        # self.rotation_backend_text = tk.StringVar(value="Rotation backend: not loaded")
        self.rotation_note = tk.StringVar(value="Hardware tracking status: not checked.")
        self.tracking_mode = tk.StringVar(value="Off")
        self.tracking_status = tk.StringVar(value="Tracking: Off")
        self.tracking_angles = tk.StringVar(value="Yaw 0.0 | Pitch 0.0 | Roll 0.0")

        self.setup_style()
        self.create_widgets()
        self.refresh_hardware_tracking_status(show_message=False)
        self.set_controls_enabled(False)
        self.root.after(100, self._process_backend_load_queue)
        self.root.after(250, self.update_gui_loop)
        self.root.after(50, self.update_head_tracking_loop)

    # ==============================================================
    # Styling
    # ==============================================================

    def setup_style(self):
        self.root.configure(bg="#f5f6f8")

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#f5f6f8")
        style.configure("Card.TFrame", background="white", relief="flat")
        style.configure(
            "Title.TLabel",
            background="#f5f6f8",
            foreground="#14395b",
            font=("Arial", 25, "bold"),
        )
        style.configure(
            "Subtitle.TLabel",
            background="#f5f6f8",
            foreground="#666666",
            font=("Arial", 11),
        )
        style.configure(
            "Section.TLabel",
            background="white",
            foreground="#14395b",
            font=("Arial", 13, "bold"),
        )
        style.configure(
            "SmallInfo.TLabel",
            background="white",
            foreground="#444444",
            font=("Arial", 10),
        )
        style.configure(
            "Time.TLabel",
            background="white",
            foreground="#1f1f1f",
            font=("Consolas", 12, "bold"),
        )
        style.configure("TButton", font=("Arial", 10), padding=6)
        style.configure("Accent.TButton", font=("Arial", 10, "bold"), padding=6)

    # ==============================================================
    # Layout
    # ==============================================================

    def create_widgets(self):
        self.scroll_canvas = tk.Canvas(
            self.root,
            bg="#f5f6f8",
            highlightthickness=0,
            borderwidth=0,
        )
        self.page_scrollbar = ttk.Scrollbar(
            self.root,
            orient=tk.VERTICAL,
            command=self.scroll_canvas.yview,
        )
        self.scroll_canvas.configure(yscrollcommand=self.page_scrollbar.set)

        self.page_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.scroll_frame = ttk.Frame(self.scroll_canvas, padding=24)
        self.scroll_window = self.scroll_canvas.create_window(
            (0, 0),
            window=self.scroll_frame,
            anchor="nw",
        )

        self.scroll_frame.bind("<Configure>", self.on_scroll_frame_configure)
        self.scroll_canvas.bind("<Configure>", self.on_scroll_canvas_configure)
        self.root.bind_all("<MouseWheel>", self.on_mousewheel)
        self.root.bind_all("<Button-4>", self.on_mousewheel)
        self.root.bind_all("<Button-5>", self.on_mousewheel)

        main = self.scroll_frame

        ttk.Label(main, text="Ambisonics Player", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            main,
            text="AmbiX file selection, binaural SH-HRTF decoding, and transport control",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(2, 18))

        self.create_input_card(main)
        self.create_playback_card(main)
        self.create_decoder_settings_card(main)
        self.create_rotation_card(main)
        self.create_info_card(main)

    def on_scroll_frame_configure(self, _event=None):
        self.scroll_canvas.configure(scrollregion=self.scroll_canvas.bbox("all"))

    def on_scroll_canvas_configure(self, event):
        self.scroll_canvas.itemconfigure(self.scroll_window, width=event.width)

    def on_mousewheel(self, event):
        if event.widget is getattr(self, "info_text", None):
            return

        if getattr(event, "num", None) == 4:
            self.scroll_canvas.yview_scroll(-1, "units")
        elif getattr(event, "num", None) == 5:
            self.scroll_canvas.yview_scroll(1, "units")
        else:
            self.scroll_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def create_input_card(self, parent):
        card = ttk.Frame(parent, style="Card.TFrame", padding=18)
        card.pack(fill=tk.X, pady=(0, 14))

        ttk.Label(card, text="Input", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        self.playback_status_label = ttk.Label(card, textvariable=self.playback_status, style="SmallInfo.TLabel")
        self.playback_status_label.grid(
            row=0, column=1, sticky="e", padx=(16, 0)
        )

        self.ambix_button = ttk.Button(
            card,
            text="Load AmbiX File",
            command=self.load_ambix,
            style="Accent.TButton",
        )
        self.ambix_button.grid(row=0, column=2, padx=(20, 0), sticky="e")

        ttk.Label(card, textvariable=self.selected_file, style="SmallInfo.TLabel").grid(
            row=1, column=0, columnspan=3, sticky="w", pady=(10, 0)
        )
        ttk.Label(card, textvariable=self.input_type, style="SmallInfo.TLabel").grid(
            row=2, column=0, sticky="w", pady=(6, 0)
        )
        ttk.Label(card, textvariable=self.output_type, style="SmallInfo.TLabel").grid(
            row=2, column=1, sticky="w", padx=(30, 0), pady=(6, 0)
        )
        ttk.Label(card, textvariable=self.pipeline_status, style="SmallInfo.TLabel").grid(
            row=3, column=0, columnspan=2, sticky="w", pady=(6, 0)
        )
        ttk.Label(card, textvariable=self.loading_text, style="SmallInfo.TLabel").grid(
            row=4, column=0, columnspan=3, sticky="w", pady=(6, 0)
        )

        card.columnconfigure(0, weight=1)
        card.columnconfigure(1, weight=1)

    def create_playback_card(self, parent):
        card = ttk.Frame(parent, style="Card.TFrame", padding=18)
        card.pack(fill=tk.X, pady=(0, 14))

        ttk.Label(card, text="Playback Control", style="Section.TLabel").grid(
            row=0, column=0, columnspan=5, sticky="w"
        )

        self.play_button = ttk.Button(card, text="Play / Resume", command=self.play)
        self.play_button.grid(row=1, column=0, padx=(0, 8), pady=(16, 12))

        self.pause_button = ttk.Button(card, text="Pause", command=self.pause)
        self.pause_button.grid(row=1, column=1, padx=8, pady=(16, 12))

        self.stop_button = ttk.Button(card, text="Stop", command=self.stop)
        self.stop_button.grid(row=1, column=2, padx=8, pady=(16, 12))

        self.loop_check = ttk.Checkbutton(
            card,
            text="Loop",
            variable=self.loop_value,
            command=self.set_loop,
        )
        self.loop_check.grid(row=1, column=3, padx=18, pady=(16, 12))

        ttk.Label(card, textvariable=self.time_text, style="Time.TLabel").grid(
            row=1, column=4, sticky="e", pady=(16, 12)
        )

        self.progress_slider = ttk.Scale(
            card,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            variable=self.progress_value,
            command=self.on_progress_drag,
        )
        self.progress_slider.grid(row=2, column=0, columnspan=5, sticky="ew", pady=(6, 4))
        self.progress_slider.bind("<ButtonPress-1>", self.on_progress_press)
        self.progress_slider.bind("<ButtonRelease-1>", self.on_progress_release)

        card.columnconfigure(4, weight=1)

    def create_decoder_settings_card(self, parent):
        card = ttk.Frame(parent, style="Card.TFrame", padding=18)
        card.pack(fill=tk.X, pady=(0, 14))

        ttk.Label(card, text="Decoder Settings", style="Section.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w"
        )
        ttk.Label(card, textvariable=self.decoder_note, style="SmallInfo.TLabel").grid(
            row=0, column=2, columnspan=4, sticky="e", padx=(16, 0)
        )

        ttk.Label(card, text="Order", background="white", font=("Arial", 10, "bold")).grid(
            row=1, column=0, sticky="w", pady=(16, 0)
        )
        self.order_box = ttk.Combobox(
            card,
            textvariable=self.order_value,
            values=["Auto", "0", "1", "2", "3", "4", "5", "6", "7"],
            state="readonly",
            width=10,
        )
        self.order_box.grid(row=1, column=1, sticky="w", padx=(14, 18), pady=(16, 0))

        ttk.Label(card, text="Block Size", background="white", font=("Arial", 10, "bold")).grid(
            row=1, column=2, sticky="w", pady=(16, 0)
        )
        self.block_size_box = ttk.Combobox(
            card,
            textvariable=self.block_size_value,
            values=["512", "1024", "2048", "4096"],
            state="readonly",
            width=10,
        )
        self.block_size_box.grid(row=1, column=3, sticky="w", padx=(14, 18), pady=(16, 0))

        ttk.Label(card, text="Volume", background="white", font=("Arial", 10, "bold")).grid(
            row=2, column=0, sticky="w", pady=(16, 0)
        )
        self.volume_slider = ttk.Scale(
            card,
            from_=0.0,
            to=1.0,
            orient=tk.HORIZONTAL,
            variable=self.volume_value,
            command=self.set_volume,
        )
        self.volume_slider.grid(row=2, column=1, columnspan=3, sticky="ew", padx=(14, 18), pady=(16, 0))
        self.volume_display = ttk.Label(
            card,
            textvariable=self.volume_display_value,
            background="white",
            width=6,
            font=("Consolas", 10),
        )
        self.volume_display.grid(row=2, column=4, sticky="w", pady=(16, 0))

        ttk.Label(
            card,
            text="Start Offset (s)",
            background="white",
            font=("Arial", 10, "bold"),
        ).grid(row=3, column=0, sticky="w", pady=(16, 0))
        self.offset_entry = ttk.Entry(card, textvariable=self.offset_value, width=12)
        self.offset_entry.grid(row=3, column=1, sticky="w", padx=(14, 0), pady=(16, 0))
        self.offset_button = ttk.Button(card, text="Set Offset", command=self.set_offset)
        self.offset_button.grid(row=3, column=2, sticky="w", padx=(12, 0), pady=(16, 0))

        ttk.Label(
            card,
            text="Headphone Filter",
            background="white",
            font=("Arial", 10, "bold"),
        ).grid(row=4, column=0, sticky="w", pady=(16, 0))
        self.headphone_box = ttk.Combobox(
            card,
            textvariable=self.headphone_value,
            values=self.get_headphone_names(),
            state="readonly",
            width=36,
        )
        self.headphone_box.grid(row=4, column=1, columnspan=3, sticky="ew", padx=(14, 18), pady=(16, 0))

        ttk.Label(card, text="HRTF SOFA", background="white", font=("Arial", 10, "bold")).grid(
            row=5, column=0, sticky="w", pady=(16, 0)
        )
        ttk.Label(card, textvariable=self.hrtf_path_value, background="white").grid(
            row=5, column=1, columnspan=3, sticky="ew", padx=(14, 18), pady=(16, 0)
        )
        self.hrtf_button = ttk.Button(card, text="Browse", command=self.select_hrtf_file)
        self.hrtf_button.grid(row=5, column=4, sticky="w", pady=(16, 0))

        self.edit_settings_button = ttk.Button(card, text="Edit Settings", command=self.unlock_decoder_settings)
        self.edit_settings_button.grid(row=5, column=5, sticky="w", padx=(12, 0), pady=(16, 0))

        ttk.Label(card, textvariable=self.loaded_settings_text, style="SmallInfo.TLabel").grid(
            row=6, column=0, columnspan=6, sticky="w", pady=(12, 0)
        )

        card.columnconfigure(3, weight=1)

    def create_rotation_card(self, parent):
        card = ttk.Frame(parent, style="Card.TFrame", padding=18)
        card.pack(fill=tk.X, pady=(0, 14))

        ttk.Label(card, text="Rotation / Head Tracking", style="Section.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w"
        )
        ttk.Label(card, textvariable=self.rotation_note, style="SmallInfo.TLabel").grid(
            row=0, column=2, columnspan=4, sticky="e", padx=(16, 0)
        )

        self.yaw_slider = self.create_rotation_slider(card, "Yaw Z", self.yaw_value, 1)
        self.pitch_slider = self.create_rotation_slider(card, "Pitch Y", self.pitch_value, 2)
        self.roll_slider = self.create_rotation_slider(card, "Roll X", self.roll_value, 3)

        self.rotation_button = ttk.Button(card, text="Apply Rotation", command=self.apply_rotation)
        self.rotation_button.grid(row=4, column=0, sticky="w", pady=(16, 0))

        self.reset_rotation_button = ttk.Button(card, text="Reset Rotation", command=self.reset_rotation)
        self.reset_rotation_button.grid(row=4, column=1, sticky="w", padx=(12, 0), pady=(16, 0))

        ttk.Label(card, textvariable=self.rotation_text, style="SmallInfo.TLabel").grid(
            row=4, column=2, columnspan=2, sticky="w", padx=(18, 0), pady=(16, 0)
        )

        # ttk.Label(card, textvariable=self.rotation_backend_text, style="SmallInfo.TLabel").grid(
        #     row=4, column=4, columnspan=2, sticky="e", pady=(16, 0)
        # )

        ttk.Label(card, text="Tracking", background="white", font=("Arial", 10, "bold")).grid(
            row=5, column=0, sticky="w", pady=(16, 0)
        )
        self.tracking_mode_box = ttk.Combobox(
            card,
            textvariable=self.tracking_mode,
            values=self.get_tracking_modes(),
            state="readonly",
            width=10,
        )
        self.tracking_mode_box.grid(row=5, column=1, sticky="w", padx=(14, 12), pady=(16, 0))

        self.refresh_tracker_button = ttk.Button(
            card,
            text="Refresh Devices",
            command=self.refresh_hardware_tracking_status,
        )
        self.refresh_tracker_button.grid(row=5, column=2, sticky="w", padx=(0, 8), pady=(16, 0))

        self.zero_tracker_button = ttk.Button(card, text="Zero Tracker", command=self.head_tracker.zero)
        self.zero_tracker_button.grid(row=5, column=3, sticky="w", padx=(0, 8), pady=(16, 0))
        self.zero_tracker_button["state"] = "disabled"

        self.start_tracking_button = ttk.Button(card, text="Start Tracking", command=self.start_head_tracking)
        self.start_tracking_button.grid(row=5, column=4, sticky="w", padx=(0, 8), pady=(16, 0))

        self.stop_tracking_button = ttk.Button(card, text="Stop Tracking", command=self.stop_head_tracking)
        self.stop_tracking_button.grid(row=5, column=5, sticky="w", padx=(0, 8), pady=(16, 0))

        ttk.Label(card, textvariable=self.tracking_status, style="SmallInfo.TLabel").grid(
            row=6, column=2, columnspan=2, sticky="w", padx=(18, 0), pady=(14, 0)
        )

        self.tracking_canvas = tk.Canvas(
            card,
            width=170,
            height=130,
            bg="white",
            highlightthickness=1,
            highlightbackground="#d8d8d8",
        )
        self.tracking_canvas.grid(row=6, column=0, columnspan=2, sticky="w", pady=(14, 0))

        ttk.Label(card, textvariable=self.tracking_angles, style="SmallInfo.TLabel").grid(
            row=6, column=4, columnspan=2, sticky="e", padx=(18, 0), pady=(14, 0)
        )

        self.draw_head_tracking_visualizer(self.orientation_state.get())

        card.columnconfigure(1, weight=1)
        card.columnconfigure(2, weight=0)
        card.columnconfigure(3, weight=0)
        card.columnconfigure(4, weight=0)

    def get_tracking_modes(self):
        modes = ["Off"]
        if self.head_tracker.is_available() and self.head_tracker_devices:
            modes.append("Hardware")
        modes.append("Demo")
        return modes

    def refresh_hardware_tracking_status(self, show_message=True):
        self.head_tracker_devices = []
        self.midi_error = ""

        # if not self.head_tracker.is_available():
        #     self.rotation_note.set("Hardware tracking unavailable: pyheadtracker is not installed.")
        # elif mido is None:
        #     self.rotation_note.set("Hardware tracking unavailable: mido is not installed.")
        # else:
        try:
            names = mido.get_input_names()
            self.head_tracker_devices = [name for name in names if "Head Tracker" in name]
            if self.head_tracker_devices:
                self.rotation_note.set(
                    "Hardware tracking available: " + ", ".join(self.head_tracker_devices)
                )
            else:
                self.rotation_note.set("Hardware tracking is not available. Use Demo mode.")
        except Exception as error:
            self.midi_error = str(error)
            self.rotation_note.set(f"Hardware tracking check failed: {self.midi_error}")

        if hasattr(self, "tracking_mode_box"):
            modes = self.get_tracking_modes()
            self.tracking_mode_box.configure(values=modes)
            if self.tracking_mode.get() not in modes:
                self.tracking_mode.set("Off")

        if show_message:
            self.update_info()

    def create_rotation_slider(self, parent, label, variable, row):
        ttk.Label(parent, text=label, background="white", font=("Arial", 10, "bold")).grid(
            row=row, column=0, sticky="ew", pady=(14, 0)
        )
        slider = ttk.Scale(
            parent,
            from_=-180.0,
            to=180.0,
            orient=tk.HORIZONTAL,
            variable=variable,
            command=lambda _value: self.update_rotation_label(),
        )
        slider.grid(row=row, column=1, columnspan=4, sticky="ew", padx=(14, 18), pady=(14, 0))
        return slider

    def create_info_card(self, parent):
        card = ttk.Frame(parent, style="Card.TFrame", padding=18)
        card.pack(fill=tk.BOTH, expand=True)

        ttk.Label(card, text="Signal Information", style="Section.TLabel").pack(anchor="w")

        text_frame = ttk.Frame(card, style="Card.TFrame")
        text_frame.pack(fill=tk.BOTH, expand=True, pady=(12, 0))

        self.info_text = tk.Text(
            text_frame,
            height=10,
            bg="white",
            fg="#1f1f1f",
            font=("Consolas", 10),
            relief="flat",
            wrap="word",
        )
        self.info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.info_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.info_text.configure(yscrollcommand=scrollbar.set)
        self.info_text.configure(state="disabled")
        self.write_info_text("No audio loaded.")

    # ==============================================================
    # State helpers
    # ==============================================================

    def has_player(self):
        return self.player is not None

    def has_loaded_player(self):
        return self.player is not None and self.player.is_loaded

    def set_controls_enabled(self, enabled: bool):
        state = tk.NORMAL if enabled else tk.DISABLED

        for widget in [
            self.play_button,
            self.pause_button,
            self.stop_button,
            self.progress_slider,
            self.volume_slider,
            self.offset_entry,
            self.offset_button,
            self.loop_check,
            # self.yaw_slider,
            # self.pitch_slider,
            # self.roll_slider,
            # self.rotation_button,
            # self.reset_rotation_button,
            # self.tracking_mode_box,
            # self.start_tracking_button,
            # self.stop_tracking_button,
        ]:
            widget.configure(state=state)

        self.edit_settings_button.configure(state=tk.NORMAL if self.has_loaded_player() else tk.DISABLED)

    def set_decoder_settings_enabled(self, enabled: bool):
        self.order_box.configure(state="readonly" if enabled else tk.DISABLED)
        self.block_size_box.configure(state="readonly" if enabled else tk.DISABLED)
        self.headphone_box.configure(state="readonly" if enabled else tk.DISABLED)
        self.hrtf_button.configure(state=tk.NORMAL if enabled else tk.DISABLED)

    def _normalize_hrtf_path(self, path):
        if path in (None, "", "Default FABIAN HRTF"):
            return None

        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = self.app_dir / candidate

        return str(candidate.resolve())

    def unlock_decoder_settings(self):
        self.set_decoder_settings_enabled(True)
        self.decoder_note.set("Settings unlocked. Click Load AmbiX File again to apply changes.")

    def set_loading(self, loading: bool, message=""):
        self.is_loading = loading

        if loading:
            self.loading_text.set(message or "Loading...")
            self.playback_status.set("Status: Loading")
            self.playback_status_label.configure(foreground="red")
            self.pipeline_status.set(f"Pipeline: {message or 'loading'}")
            self.ambix_button.configure(state=tk.DISABLED)
            self.set_decoder_settings_enabled(False)
            self.edit_settings_button.configure(state=tk.DISABLED)
            self.set_controls_enabled(False)
        else:
            self.loading_text.set("")
            self.ambix_button.configure(state=tk.NORMAL)
            self.set_decoder_settings_enabled(self.has_loaded_player())
            self.set_controls_enabled(self.has_loaded_player())

        self.root.update_idletasks()

    def reset_progress_display(self):
        duration = self.player.get_duration() if self.has_loaded_player() else 0.0
        self.progress_slider.configure(to=max(duration, 0.01))
        self.progress_value.set(0.0)
        self.time_text.set(f"{self.format_time(0.0)} / {self.format_time(duration)}")

    # ==============================================================
    # Settings helpers
    # ==============================================================

    def get_order(self):
        value = self.order_value.get()
        return None if value == "Auto" else int(value)

    def get_block_size(self):
        return int(self.block_size_value.get())

    def get_selected_hrtf_path(self):
        value = self.hrtf_path_value.get()
        if value == "Default FABIAN HRTF":
            local_default = self.app_dir / self.DEFAULT_HRTF_FILE
            if local_default.exists():
                return str(local_default)
            return self.DEFAULT_HRTF_FILE
        return value

    def get_headphone_names(self):
        hp_dir = self.app_dir / "resources" / "Headphones"
        if not hp_dir.exists():
            return ["None"]
        return ["None"] + ["Diffuse Field Equalization"] + sorted(path.name for path in hp_dir.iterdir() if path.is_dir())

    def select_hrtf_file(self):
        file_path = filedialog.askopenfilename(
            title="Select HRTF SOFA file",
            filetypes=[("SOFA files", "*.sofa"), ("All files", "*.*")],
        )
        if file_path:
            self.hrtf_path_value.set(file_path)

    def show_wav_disabled_message(self):
        messagebox.showinfo(
            "Disabled",
            "Standalone binaural WAV playback is disabled because the current backend "
            "is an AmbiX streaming decoder. Use Load AmbiX File for the final workflow.",
        )

    def update_loaded_settings(self, result):
        self.loaded_settings_text.set(
            "Loaded settings: "
            f"order {result['order']} | "
            f"{result['channels']} channels | "
            f"{result['sample_rate']} Hz | "
            f"block {self.block_size_value.get()} | "
            f"HRTF {self.hrtf_path_value.get()} | "
            f"headphone {result['headphone']}"
        )

    # def update_rotation_backend_status(self):
    #     if not self.has_loaded_player():
    #         self.rotation_backend_text.set("Rotation backend: not loaded")
    #         return

    #     if hasattr(self.player.sh, "get_rotation_backend_status"):
    #         status = self.player.sh.get_rotation_backend_status()
    #     else:
    #         status = "unknown"
    #     self.rotation_backend_text.set(f"Rotation backend: {status}")

    def _handle_error(self, title, error):
        message = str(error)
        print(f"[{title}] {message}")
        traceback.print_exc()
        self.stop()
        messagebox.showerror(title, message)

    # ==============================================================
    # Loading
    # ==============================================================

    def load_ambix(self):
        if self.is_loading:
            return

        file_path = filedialog.askopenfilename(
            title="Select AmbiX / Ambisonics WAV file",
            filetypes=[("Ambisonics WAV files", "*.wav"), ("All files", "*.*")],
        )

        if not file_path:
            return

        if self.has_player():
            try:
                self.player.stop()
            except Exception as e:
                print(e)
                pass
        self.stop_head_tracking(reset_orientation=False)

        file_name = os.path.basename(file_path)
        self.selected_file.set(f"AmbiX file selected: {file_name}")
        self.input_type.set("Input type: AmbiX / multichannel WAV")
        self.output_type.set("Output: preparing binaural stream")
        self.pipeline_status.set("Pipeline: loading AmbiX and preparing decoder")
        self.playback_status.set("Status: Loading")
        self.playback_status_label.configure(foreground="red")
        self.write_info_text(self.build_info_text())

        order = self.get_order()
        block_size = self.get_block_size()
        gain = float(self.volume_value.get())
        headphone_name = self.headphone_value.get()
        hrtf_path = self.get_selected_hrtf_path()
        app_dir = str(self.app_dir)

        self.set_loading(True, "Loading AmbiX and preprocessing HRTF. Controls are disabled.")

        def worker(path, requested_order, requested_block_size, requested_gain, hrtf_file, headphone, cwd):
            old_cwd = os.getcwd()
            try:
                os.chdir(cwd)

                ambix = AmbisonicsFile(
                    path,
                    chunk_size=requested_block_size,
                    order=requested_order,
                    trim_extra_channels=True,
                    format="ambix",
                    normalization="SN3D",
                )
                if not ambix.is_valid():
                    raise ValueError(
                        "Selected file is not valid for the selected Ambisonics order."
                    )

                hrtf = HRTF(hrtf_file)
                if headphone != "None":
                    hrtf.load_hp_filter(headphone)

                sh = SphericalHarmonics(
                    hrtf=hrtf,
                    sampling_rate=ambix.get_samplerate(),
                    ambi_order=ambix.get_order(),
                )

                player = AudioPlayer(ambi_file=ambix, sh=sh, gain=requested_gain)
                player.source_name = path
                player.set_loaded(True)

                result = {
                    "player": player,
                    "path": path,
                    "order": ambix.get_order(),
                    "channels": ambix.get_num_channels(),
                    "sample_rate": ambix.get_samplerate(),
                    "duration": ambix.get_duration(),
                    "headphone": headphone,
                    "hrtf": hrtf_file,
                }
                self._backend_load_queue.put(("new_load", result))
            except Exception as error:
                traceback.print_exc()
                self._backend_load_queue.put(("error", str(error)))
            finally:
                os.chdir(old_cwd)

        threading.Thread(
            target=worker,
            args=(file_path, order, block_size, gain, hrtf_path, headphone_name, app_dir),
            daemon=True,
        ).start()

    def update_decoder_settings(self):

        # try to find any changes to the decoder settings. if so, update all necessary attributes in player.
        # get current values
        requested_order = self.get_order()
        requested_block_size = self.get_block_size()
        requested_hp = self.headphone_value.get()
        requested_hrtf = self.get_selected_hrtf_path()

        if requested_hp in (None, "", "None"):
            requested_hp = None

        # lock decoder settings and playback buttons
        # put a text that informs the user why everything is disabled
        self.set_loading(True, "Processing new decoder values. Controls are disabled.")

        self.update_event.clear()

        # the worker thread setting up all new variables
        def update_decoder(new_hrtf=None, new_hp=None, new_block_size=None, new_order=None):
            try: 
                player = self.player
                if player is None:
                    raise RuntimeError("No player loaded.")
                
                current_hrtf = self._normalize_hrtf_path(getattr(player.sh.hrtf, "path", None))
                current_hp = getattr(player.sh.hrtf, "current_filter", None)
                current_block_size = player.ambi_file.get_chunk_size()
                current_order = player.ambi_file.get_order()
                
                new_hrtf = self._normalize_hrtf_path(new_hrtf)
                if new_hp in (None, "", "None"):
                    new_hp = None

                change = False
                needs_rebuild = False

                if new_hrtf is not None and new_hrtf != current_hrtf:
                    print("Updating HRTF")
                    # hrtf file changed
                    self.player.sh.hrtf.path = Path(new_hrtf)
                    player.sh.hrtf = HRTF(new_hrtf)
                    change = True
                    needs_rebuild = True

                if new_hp != current_hp:
                    print("Updating HP Filter")
                    # hp filter changed
                    # calculate new hp_filter
                    self.player.sh.hrtf.load_hp_filter(new_hp)
                    change = True
                    needs_rebuild = True

                if new_block_size is not None and new_block_size != current_block_size:
                    print("Updating Block Size")
                    # block size changed
                    # update block size in ambi_file. player gets the updates automatically
                    self.player.ambi_file.set_chunk_size(new_block_size)
                    change = True

                if new_order is not None and new_order != current_order:
                    print("Updating Order")
                    # order changed
                    # set order in ambi file
                    self.player.ambi_file.set_order(new_order)
                    change = True
                    needs_rebuild = True

                if needs_rebuild:
                    # rebuild our spherical harmonics
                    new_sh = SphericalHarmonics(
                        hrtf=player.sh.hrtf,
                        sampling_rate=player.ambi_file.get_samplerate(),
                        ambi_order=player.ambi_file.get_order(),
                    )
                    # cleanup old SH instance
                    player.sh.close()
                    player.sh = new_sh
                    
                if change:
                    player._reset_process_variables()

                result = {
                    "order": player.ambi_file.get_order(),
                    "channels": player.ambi_file.get_num_channels(),
                    "sample_rate": player.ambi_file.get_samplerate(),
                    "duration": player.ambi_file.get_duration(),
                    "headphone": new_hp if new_hp is not None else current_hp,
                    "hrtf": new_hrtf if new_hrtf is not None else current_hrtf,
                }
                self._backend_load_queue.put(("decoder_update", result))

            except Exception as error:
                traceback.print_exc()
                self._backend_load_queue.put(("error", str(error)))
            finally:
                self.update_event.set()

        # starting worker thread
        threading.Thread(
            target=update_decoder,
            args=(requested_hrtf, requested_hp, requested_block_size, requested_order),
            daemon=True,
        ).start()


    def _process_backend_load_queue(self):
        try:
            while not self._backend_load_queue.empty():
                message_type, payload = self._backend_load_queue.get_nowait()

                if message_type == "new_load":
                    result = payload

                    # if already a player was loaded, discard the old one before loading the new one
                    if self.has_loaded_player():
                        print(f"Closing the old player before assigning the new one.")
                        self.player.close()

                    self.player = result["player"]

                    file_name = os.path.basename(result["path"])
                    self.selected_file.set(f"AmbiX: {file_name}")
                    self.input_type.set(
                        f"Input type: AmbiX / order {result['order']} / "
                        f"{result['channels']} channels"
                    )
                    self.output_type.set("Output: binaural streaming")
                    self.pipeline_status.set("Pipeline: AmbiX -> SH-HRTF decoder -> AudioPlayer")
                    self.playback_status.set("Status: Loaded")
                    self.playback_status_label.configure(foreground="black")
                    self.decoder_note.set("Loaded decoder settings are locked. Use Edit Settings to change them for the next load.")

                    self.reset_progress_display()
                    # self.update_rotation_label()
                    self.update_loaded_settings(result)
                    # self.update_rotation_backend_status()
                    self.set_loading(False)
                    self.update_info()

                elif message_type == "decoder_update":
                    result = payload
                    self.update_event.clear()

                    self.input_type.set(
                        f"Input type: AmbiX / order {result['order']} / "
                        f"{result['channels']} channels"
                    )
                    self.output_type.set("Output: binaural streaming")
                    self.pipeline_status.set("Pipeline: AmbiX -> SH-HRTF decoder -> AudioPlayer")
                    self.playback_status.set("Status: Loaded")
                    self.playback_status_label.configure(foreground="black")
                    self.decoder_note.set("Loaded decoder settings are locked. Use Edit Settings to change them for the next load.")

                    self.reset_progress_display()
                    # self.update_rotation_label()
                    self.update_loaded_settings(result)
                    # self.update_rotation_backend_status()
                    self.set_loading(False)
                    # make sure decoder settings are disabled if we only updated the decoder, because this only gets called on a play
                    self.set_decoder_settings_enabled(False)
                    self.update_info()

                else:
                    self.set_loading(False)
                    self.pipeline_status.set("Pipeline: load failed")
                    self.playback_status.set("Status: Load failed")
                    self.playback_status_label.configure(foreground="red")
                    self.set_controls_enabled(False)
                    self.update_info()
                    self._handle_error("load Error", payload)
        finally:
            self.root.after(100, self._process_backend_load_queue)

    # ==============================================================
    # Playback control
    # ==============================================================

    def play(self):
        if not self.has_loaded_player():
            return
        try:
            self.update_decoder_settings()
            # wait until any updates are done
            self.update_event.wait()

            # for safety
            if not self.has_loaded_player():
                return

            self.player.play()
            self.set_decoder_settings_enabled(False)
            self.playback_status.set("Status: Playing")
            self.update_info()
        except Exception as error:
            self._handle_error("Error", error)

    def pause(self):
        if not self.has_loaded_player():
            return
        try:
            self.player.pause()
            self.playback_status.set("Status: Paused")
            self.update_info()
        except Exception as error:
            self._handle_error("Error", error)

    def stop(self):
        if not self.has_loaded_player():
            return
        try:
            self.player.stop()
            self.set_decoder_settings_enabled(True)
            self.playback_status.set("Status: Stopped")
            self.progress_value.set(0.0)
            self.time_text.set(
                f"{self.format_time(0.0)} / {self.format_time(self.player.get_duration())}"
            )
            self.update_info()
        except Exception as error:
            self._handle_error("Error", error)

    def set_volume(self, value):
        volume = max(0.0, min(float(value), 1.0))
        self.volume_display_value.set(f"{volume:.2f}")

        if not self.has_loaded_player():
            return

        try:
            self.player.set_volume(volume**2.41)
            self.update_info()
        except Exception as error:
            self._handle_error("Error", error)

    def set_offset(self):
        if not self.has_loaded_player():
            return

        try:
            seconds = float(self.offset_value.get())
            if seconds < 0:
                raise ValueError("Offset must be non-negative.")

            duration = self.player.get_duration()
            if seconds >= duration:
                raise ValueError(f"Offset must be smaller than {duration:.2f} seconds.")

            self.player.seek_to(seconds)
            self.progress_value.set(self.player.get_current_time())
            self.playback_status.set("Status: Seeked")
            self.update_info()
        except ValueError as error:
            self._handle_error("Invalid offset", error)
        except Exception as error:
            self._handle_error("Error", error)

    def set_loop(self):
        if not self.has_loaded_player():
            return
        try:
            self.player.set_loop(self.loop_value.get())
            self.update_info()
        except Exception as error:
            self._handle_error("Error", error)

    # ==============================================================
    # Progress / seek
    # ==============================================================

    def on_progress_press(self, _event):
        self.is_dragging_progress = True

    def on_progress_drag(self, value):
        if not self.has_loaded_player():
            return

        if self.is_dragging_progress:
            seconds = float(value)
            self.time_text.set(
                f"{self.format_time(seconds)} / {self.format_time(self.player.get_duration())}"
            )

    def on_progress_release(self, _event):
        if not self.has_loaded_player():
            self.is_dragging_progress = False
            return

        try:
            seconds = float(self.progress_value.get())
            self.player.seek_to(seconds)
            self.progress_value.set(self.player.get_current_time())
            self.playback_status.set("Status: Seeked")
            self.update_info()
        except Exception as error:
            self._handle_error("Error", error)
        finally:
            self.is_dragging_progress = False

    # ==============================================================
    # Rotation
    # ==============================================================

    def update_rotation_label(self):
        self.rotation_text.set(
            "Rotation: "
            f"yaw {self.yaw_value.get():.1f}, "
            f"pitch {self.pitch_value.get():.1f}, "
            f"roll {self.roll_value.get():.1f}"
        )
        self.tracking_angles.set(
            f"Yaw {self.yaw_value.get():.1f} | "
            f"Pitch {self.pitch_value.get():.1f} | "
            f"Roll {self.roll_value.get():.1f}"
        )

    def apply_rotation(self):
        if not self.has_loaded_player():
            return

        try:
            yaw = float(self.yaw_value.get())
            pitch = float(self.pitch_value.get())
            roll = float(self.roll_value.get())
            orientation = self.orientation_state.set(yaw, pitch, roll, source="manual")
            self.apply_orientation_to_audio(orientation)
            self.update_rotation_label()
            self.draw_head_tracking_visualizer(orientation)
            self.pipeline_status.set("Pipeline: rotation matrix updated")
            self.playback_status.set("Status: Rotation updated")
            self.update_info()
        except Exception as error:
            self._handle_error("Rotation Error", error)

    def reset_rotation(self):
        self.yaw_value.set(0.0)
        self.pitch_value.set(0.0)
        self.roll_value.set(0.0)
        self.apply_rotation()

    def apply_orientation_to_audio(self, orientation):
        if self.has_loaded_player():
            self.player.sh.set_rotation([orientation.yaw, orientation.pitch, orientation.roll])

    def start_head_tracking(self):
        self.refresh_hardware_tracking_status(show_message=False)

        # if rotation tracker is running, stop it first before startin again
        if self.rotation_tracker.is_running():
            # make sure we don't forget the current tracking mode
            tracking_mode = self.tracking_mode.get()
            self.stop_head_tracking()
            self.tracking_mode.set(tracking_mode)

        match self.tracking_mode.get():
            case "Demo":
                self.rotation_tracker = self.demo_tracker
            case "Hardware":
                if "Hardware" not in self.get_tracking_modes():
                    self.tracking_status.set("Tracking: Hardware unavailable")
                    messagebox.showwarning(
                        "Head Tracking",
                        "No supported hardware tracker is available. Use Demo tracking instead.",
                    )
                    return
                try:
                    self.rotation_tracker = self.head_tracker
                except Exception as e:
                    print(f"Something went wrong: {e}")
            case "Off":
                return

        try:
            self.rotation_tracker.start()
        except Exception as error:
            self.tracking_status.set("Tracking: failed to start")
            print(f"[Head Tracking Error] {error}")
            traceback.print_exc()
            messagebox.showerror("Head Tracking Error", str(error))
            return
        
        if self.tracking_mode.get() == "Hardware" and self.rotation_tracker.is_running():
            # should only do this on successful start ....
            self.zero_tracker_button["state"] = "normal"

        self.tracking_status.set("Tracking: " + self.tracking_mode.get() + " running")
        self.playback_status.set("Status: " + self.tracking_mode.get() + " tracking")

    def stop_head_tracking(self, reset_orientation=True):
        self.rotation_tracker.stop()
        self.zero_tracker_button["state"] = "disabled"
        self.tracking_mode.set("Off")
        self.tracking_status.set("Tracking: Off")

        if reset_orientation:
            orientation = self.orientation_state.set(0.0, 0.0, 0.0, source="off")
            self.yaw_value.set(0.0)
            self.pitch_value.set(0.0)
            self.roll_value.set(0.0)
            self.update_rotation_label()
            self.draw_head_tracking_visualizer(orientation)
            if self.has_loaded_player():
                self.apply_orientation_to_audio(orientation)

    def update_head_tracking_loop(self):
        # !query both trackers for orientation - inefficient code!
        if self.rotation_tracker.is_running():
            orientation = self.rotation_tracker.orientation_state.get()
        
            self.yaw_value.set(orientation.yaw)
            self.pitch_value.set(orientation.pitch)
            self.roll_value.set(orientation.roll)

            self.update_rotation_label()
            self.draw_head_tracking_visualizer(orientation)
            self.apply_orientation_to_audio(orientation)

        self.root.after(50, self.update_head_tracking_loop)

    def draw_head_tracking_visualizer(self, orientation):
        canvas = self.tracking_canvas
        canvas.delete("all")

        width = int(canvas["width"])
        height = int(canvas["height"])
        cx = width // 2
        cy = height // 2 + 8
        radius = 34

        # Tkinter coordinates grow downward. Positive yaw rotates clockwise here.
        angle = math.radians(orientation.yaw - 90.0)
        arrow_length = 48
        ax = cx + arrow_length * math.cos(angle)
        ay = cy + arrow_length * math.sin(angle)

        canvas.create_oval(cx - radius, cy - radius, cx + radius, cy + radius, outline="#14395b", width=2)
        canvas.create_line(cx, cy, ax, ay, fill="#14395b", width=3, arrow=tk.LAST)
        canvas.create_text(cx, 14, text="Head direction", fill="#444444", font=("Arial", 9, "bold"))
        canvas.create_text(cx, height - 12, text=f"yaw {orientation.yaw:.1f} deg", fill="#444444", font=("Arial", 9))

    # ==============================================================
    # Info update
    # ==============================================================

    def build_info_text(self):
        if not self.has_loaded_player():
            return (
                "No playable AmbiX audio loaded.\n\n"
                f"{self.input_type.get()}\n"
                f"{self.output_type.get()}\n"
                f"{self.pipeline_status.get()}\n\n"
                "Current GUI settings:\n"
                f"Selected order: {self.order_value.get()}\n"
                f"Block size: {self.block_size_value.get()} samples\n"
                f"HRTF: {self.hrtf_path_value.get()}\n"
            f"Headphone filter: {self.headphone_value.get()}\n"
            f"{self.rotation_note.get()}\n"
            # f"{self.rotation_backend_text.get()}\n"
            f"Detected head trackers: {self.format_head_tracker_devices()}\n"
            f"{self.loaded_settings_text.get()}\n"
        )

        return (
            f"{self.player.get_info_text()}\n\n"
            "GUI / decoder settings:\n"
            f"{self.input_type.get()}\n"
            f"{self.output_type.get()}\n"
            f"{self.pipeline_status.get()}\n"
            f"Selected order: {self.order_value.get()}\n"
            f"Block size: {self.block_size_value.get()} samples\n"
            f"HRTF: {self.hrtf_path_value.get()}\n"
            f"Headphone filter: {self.headphone_value.get()}\n"
            f"{self.rotation_text.get()}\n"
            f"{self.rotation_note.get()}\n"
            # f"{self.rotation_backend_text.get()}\n"
            f"Detected head trackers: {self.format_head_tracker_devices()}\n"
            f"{self.tracking_status.get()}\n"
            f"{self.tracking_angles.get()}\n"
            f"{self.loaded_settings_text.get()}\n\n"
            "Notes:\n"
            "Manual rotation and Demo tracking are available for presentations. "
            "Hardware tracking is enabled only when a supported tracker is detected."
        )

    def format_head_tracker_devices(self):
        if self.head_tracker_devices:
            return ", ".join(self.head_tracker_devices)
        if self.midi_error:
            return f"none ({self.midi_error})"
        return "none"

    def write_info_text(self, text):
        self.info_text.configure(state="normal")
        self.info_text.delete("1.0", tk.END)
        self.info_text.insert(tk.END, text)
        self.info_text.configure(state="disabled")

    def update_info(self):
        self.write_info_text(self.build_info_text())

    def update_gui_loop(self):
        if self.has_loaded_player():
            current_time = self.player.get_current_time()
            duration = self.player.get_duration()

            if not self.is_dragging_progress:
                self.progress_value.set(current_time)
                self.time_text.set(
                    f"{self.format_time(current_time)} / {self.format_time(duration)}"
                )

            self.refresh_playback_status()
            self.update_info()

        self.root.after(250, self.update_gui_loop)

    @staticmethod
    def format_time(seconds: float):
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes:02d}:{secs:05.2f}"

    def refresh_playback_status(self):
        if not self.has_loaded_player() or self.is_loading:
            return

        if self.player.pause_event.is_set():
            self.playback_status.set("Status: Paused")
        elif self.player.play_event.is_set() and not self.player.stop_event.is_set():
            self.playback_status.set("Status: Playing")

    # ==============================================================
    # Lifecycle
    # ==============================================================

    def on_close(self):
        self.stop_head_tracking(reset_orientation=False)
        if self.has_player():
            try:
                self.player.close()
            except Exception:
                print("Player didn't close cleanly!")
                pass
        self.root.destroy()

    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.mainloop()
