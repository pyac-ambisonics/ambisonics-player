import os
import threading
import tkinter as tk
from pathlib import Path
from queue import Queue
from tkinter import filedialog, messagebox
from tkinter import ttk

from ambisonics_file_English import AmbisonicsFile
from audio_player import AudioPlayer
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
        self.root.geometry("1050x820")
        self.root.minsize(980, 760)

        # File / pipeline state
        self.selected_file = tk.StringVar(value="No input loaded")
        self.input_type = tk.StringVar(value="Input type: none")
        self.output_type = tk.StringVar(value="Output: none")
        self.pipeline_status = tk.StringVar(value="Pipeline: waiting for input")
        self.loading_text = tk.StringVar(value="")

        # Playback state
        self.time_text = tk.StringVar(value="00:00.00 / 00:00.00")
        self.volume_value = tk.DoubleVar(value=1.0)
        self.volume_display_value = tk.StringVar(value="1.00")
        self.offset_value = tk.StringVar(value="0")
        self.loop_value = tk.BooleanVar(value=False)
        self.progress_value = tk.DoubleVar(value=0.0)

        # Decoder settings
        self.order_value = tk.StringVar(value="Auto")
        self.block_size_value = tk.StringVar(value="2048")
        self.hrtf_path_value = tk.StringVar(value="Default FABIAN HRTF")
        self.headphone_value = tk.StringVar(value="None")

        # Manual rotation, maps to SphericalHarmonics.set_rotation([z, y, x])
        self.yaw_value = tk.DoubleVar(value=0.0)
        self.pitch_value = tk.DoubleVar(value=0.0)
        self.roll_value = tk.DoubleVar(value=0.0)
        self.rotation_text = tk.StringVar(value="Rotation: yaw 0.0, pitch 0.0, roll 0.0")

        self.setup_style()
        self.create_widgets()
        self.set_controls_enabled(False)
        self.root.after(100, self._process_backend_load_queue)
        self.root.after(250, self.update_gui_loop)

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
        main = ttk.Frame(self.root, padding=24)
        main.pack(fill=tk.BOTH, expand=True)

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

    def create_input_card(self, parent):
        card = ttk.Frame(parent, style="Card.TFrame", padding=18)
        card.pack(fill=tk.X, pady=(0, 14))

        ttk.Label(card, text="Input", style="Section.TLabel").grid(row=0, column=0, sticky="w")

        self.ambix_button = ttk.Button(
            card,
            text="Load AmbiX File",
            command=self.load_ambix,
            style="Accent.TButton",
        )
        self.ambix_button.grid(row=0, column=2, padx=(20, 0), sticky="e")

        self.wav_button = ttk.Button(
            card,
            text="Load Binaural WAV (disabled)",
            command=self.show_wav_disabled_message,
        )
        self.wav_button.grid(row=1, column=2, padx=(20, 0), pady=(8, 0), sticky="e")
        self.wav_button.configure(state=tk.DISABLED)

        ttk.Label(card, textvariable=self.selected_file, style="SmallInfo.TLabel").grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(10, 0)
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
            row=0, column=0, columnspan=6, sticky="w"
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
            values=["1024", "2048", "4096"],
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

        card.columnconfigure(3, weight=1)

    def create_rotation_card(self, parent):
        card = ttk.Frame(parent, style="Card.TFrame", padding=18)
        card.pack(fill=tk.X, pady=(0, 14))

        ttk.Label(card, text="Manual Rotation", style="Section.TLabel").grid(
            row=0, column=0, columnspan=5, sticky="w"
        )

        self.yaw_slider = self.create_rotation_slider(card, "Yaw Z", self.yaw_value, 1)
        self.pitch_slider = self.create_rotation_slider(card, "Pitch Y", self.pitch_value, 2)
        self.roll_slider = self.create_rotation_slider(card, "Roll X", self.roll_value, 3)

        self.rotation_button = ttk.Button(card, text="Apply Rotation", command=self.apply_rotation)
        self.rotation_button.grid(row=4, column=0, sticky="w", pady=(16, 0))

        self.reset_rotation_button = ttk.Button(card, text="Reset Rotation", command=self.reset_rotation)
        self.reset_rotation_button.grid(row=4, column=1, sticky="w", padx=(12, 0), pady=(16, 0))

        ttk.Label(card, textvariable=self.rotation_text, style="SmallInfo.TLabel").grid(
            row=4, column=2, columnspan=3, sticky="w", padx=(18, 0), pady=(16, 0)
        )

        card.columnconfigure(1, weight=1)

    def create_rotation_slider(self, parent, label, variable, row):
        ttk.Label(parent, text=label, background="white", font=("Arial", 10, "bold")).grid(
            row=row, column=0, sticky="w", pady=(14, 0)
        )
        slider = ttk.Scale(
            parent,
            from_=-180.0,
            to=180.0,
            orient=tk.HORIZONTAL,
            variable=variable,
            command=lambda _value: self.update_rotation_label(),
        )
        slider.grid(row=row, column=1, columnspan=3, sticky="ew", padx=(14, 18), pady=(14, 0))
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
            self.yaw_slider,
            self.pitch_slider,
            self.roll_slider,
            self.rotation_button,
            self.reset_rotation_button,
        ]:
            widget.configure(state=state)

    def set_loading(self, loading: bool, message=""):
        self.is_loading = loading

        if loading:
            self.loading_text.set(message or "Loading...")
            self.pipeline_status.set(f"Pipeline: {message or 'loading'}")
            self.ambix_button.configure(state=tk.DISABLED)
            self.order_box.configure(state=tk.DISABLED)
            self.block_size_box.configure(state=tk.DISABLED)
            self.headphone_box.configure(state=tk.DISABLED)
            self.hrtf_button.configure(state=tk.DISABLED)
            self.set_controls_enabled(False)
        else:
            self.loading_text.set("")
            self.ambix_button.configure(state=tk.NORMAL)
            self.order_box.configure(state="readonly")
            self.block_size_box.configure(state="readonly")
            self.headphone_box.configure(state="readonly")
            self.hrtf_button.configure(state=tk.NORMAL)
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
        return ["None"] + sorted(path.name for path in hp_dir.iterdir() if path.is_dir())

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
            except Exception:
                pass

        file_name = os.path.basename(file_path)
        self.selected_file.set(f"AmbiX file selected: {file_name}")
        self.input_type.set("Input type: AmbiX / multichannel WAV")
        self.output_type.set("Output: preparing binaural stream")
        self.pipeline_status.set("Pipeline: loading AmbiX and preparing decoder")
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
                self._backend_load_queue.put(("success", result))
            except Exception as error:
                self._backend_load_queue.put(("error", str(error)))
            finally:
                os.chdir(old_cwd)

        threading.Thread(
            target=worker,
            args=(file_path, order, block_size, gain, hrtf_path, headphone_name, app_dir),
            daemon=True,
        ).start()

    def _process_backend_load_queue(self):
        try:
            while not self._backend_load_queue.empty():
                message_type, payload = self._backend_load_queue.get_nowait()

                if message_type == "success":
                    result = payload
                    self.player = result["player"]

                    file_name = os.path.basename(result["path"])
                    self.selected_file.set(f"AmbiX: {file_name}")
                    self.input_type.set(
                        f"Input type: AmbiX / order {result['order']} / "
                        f"{result['channels']} channels"
                    )
                    self.output_type.set("Output: binaural streaming")
                    self.pipeline_status.set("Pipeline: AmbiX -> SH-HRTF decoder -> AudioPlayer")

                    self.reset_progress_display()
                    self.update_rotation_label()
                    self.set_loading(False)
                    self.update_info()
                else:
                    self.set_loading(False)
                    self.pipeline_status.set("Pipeline: load failed")
                    self.set_controls_enabled(False)
                    self.update_info()
                    messagebox.showerror("Load error", payload)
        finally:
            self.root.after(100, self._process_backend_load_queue)

    # ==============================================================
    # Playback control
    # ==============================================================

    def play(self):
        if not self.has_loaded_player():
            return
        try:
            self.player.play()
            self.update_info()
        except Exception as error:
            messagebox.showerror("Error", str(error))

    def pause(self):
        if not self.has_loaded_player():
            return
        try:
            self.player.pause()
            self.update_info()
        except Exception as error:
            messagebox.showerror("Error", str(error))

    def stop(self):
        if not self.has_loaded_player():
            return
        try:
            self.player.stop()
            self.progress_value.set(0.0)
            self.time_text.set(
                f"{self.format_time(0.0)} / {self.format_time(self.player.get_duration())}"
            )
            self.update_info()
        except Exception as error:
            messagebox.showerror("Error", str(error))

    def set_volume(self, value):
        volume = max(0.0, min(float(value), 1.0))
        self.volume_display_value.set(f"{volume:.2f}")

        if not self.has_loaded_player():
            return

        try:
            self.player.set_volume(volume)
            self.update_info()
        except Exception as error:
            messagebox.showerror("Error", str(error))

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
            self.update_info()
        except ValueError as error:
            messagebox.showerror("Invalid offset", str(error))
        except Exception as error:
            messagebox.showerror("Error", str(error))

    def set_loop(self):
        if not self.has_loaded_player():
            return
        try:
            self.player.set_loop(self.loop_value.get())
            self.update_info()
        except Exception as error:
            messagebox.showerror("Error", str(error))

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
            self.update_info()
        except Exception as error:
            messagebox.showerror("Error", str(error))
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

    def apply_rotation(self):
        if not self.has_loaded_player():
            return

        try:
            yaw = float(self.yaw_value.get())
            pitch = float(self.pitch_value.get())
            roll = float(self.roll_value.get())
            self.player.sh.set_rotation([yaw, pitch, roll])
            self.update_rotation_label()
            self.pipeline_status.set("Pipeline: rotation matrix updated")
            self.update_info()
        except Exception as error:
            messagebox.showerror("Rotation error", str(error))

    def reset_rotation(self):
        self.yaw_value.set(0.0)
        self.pitch_value.set(0.0)
        self.roll_value.set(0.0)
        self.apply_rotation()

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
            f"{self.rotation_text.get()}\n\n"
            "Notes:\n"
            "Manual rotation updates the SH rotation matrix. Hardware headtracking "
            "still needs a separate tracker thread before it should be enabled for demos.\n"
            "Standalone binaural WAV playback is disabled in this GUI because the current "
            "backend is the AmbiX streaming decoder."
        )

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

            self.update_info()

        self.root.after(250, self.update_gui_loop)

    @staticmethod
    def format_time(seconds: float):
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes:02d}:{secs:05.2f}"

    # ==============================================================
    # Lifecycle
    # ==============================================================

    def on_close(self):
        if self.has_player():
            try:
                self.player.stop()
            except Exception:
                pass
        self.root.destroy()

    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.mainloop()
