import os
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import threading
from queue import Queue
from spherical import SphericalHarmonics
from ambisonics_file_English import AmbisonicsFile
from audio_player import AudioPlayer
from hrtf import HRTF

from audio_player import AudioPlayer


class AudioPlayerGUI:
    """
    GUI prototype for the Ambisonics Player.

    Main input modes:
    1. Load AmbiX File
       Intended final project input. At the moment this button is prepared
       as a placeholder for the future full pipeline:
       AmbiX -> Loader -> Decoder -> pyfar.Signal -> Player.

    2. Load Binaural WAV
       For standalone playback tests and already-decoded binaural WAV files.

    3. Decoder output as pyfar.Signal
       Internal integration interface used by test_integration_decoder_gui.py.
    """

    def __init__(self):
        self.player = None

        self.root = tk.Tk()
        self.root.title("Ambisonics Player - GUI Prototype")
        self.root.geometry("980x780")
        self.root.minsize(900, 720)

        # GUI state variables
        self.selected_file = tk.StringVar(value="No input loaded")
        self.input_type = tk.StringVar(value="Input type: none")
        self.output_type = tk.StringVar(value="Output: none")
        self.pipeline_status = tk.StringVar(value="Pipeline: waiting for input")

        self.status_text = tk.StringVar(value="No audio loaded.")
        self.time_text = tk.StringVar(value="00:00.00 / 00:00.00")
        self.volume_value = tk.DoubleVar(value=1.0)
        self.offset_value = tk.StringVar(value="0")
        self.loop_value = tk.BooleanVar(value=False)
        self.progress_value = tk.DoubleVar(value=0.0)

        self.is_dragging_progress = False

        self.block_size = 512

        self._backend_load_queue = Queue()
        self.root.after(100, self._process_backend_load_queue)

        self.setup_style()
        self.create_widgets()
        self.update_gui_loop()
        self.set_controls_enabled(False)

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
            font=("Arial", 25, "bold")
        )

        style.configure(
            "Subtitle.TLabel",
            background="#f5f6f8",
            foreground="#666666",
            font=("Arial", 11)
        )

        style.configure(
            "Section.TLabel",
            background="white",
            foreground="#14395b",
            font=("Arial", 13, "bold")
        )

        style.configure(
            "SmallInfo.TLabel",
            background="white",
            foreground="#444444",
            font=("Arial", 10)
        )

        style.configure(
            "Time.TLabel",
            background="white",
            foreground="#1f1f1f",
            font=("Consolas", 12, "bold")
        )

        style.configure(
            "TButton",
            font=("Arial", 10),
            padding=6
        )

        style.configure(
            "Accent.TButton",
            font=("Arial", 10, "bold"),
            padding=6
        )

    # ==============================================================
    # Layout
    # ==============================================================

    def create_widgets(self):
        main = ttk.Frame(self.root, padding=24)
        main.pack(fill=tk.BOTH, expand=True)

        # Header
        title = ttk.Label(
            main,
            text="Ambisonics Player",
            style="Title.TLabel"
        )
        title.pack(anchor="w")

        subtitle = ttk.Label(
            main,
            text="Player / GUI prototype for binaural playback and decoder integration",
            style="Subtitle.TLabel"
        )
        subtitle.pack(anchor="w", pady=(2, 18))

        self.create_input_card(main)
        self.create_playback_card(main)
        self.create_settings_card(main)
        self.create_info_card(main)

        self.set_controls_enabled(False)

    def create_input_card(self, parent):
        input_card = ttk.Frame(parent, style="Card.TFrame", padding=18)
        input_card.pack(fill=tk.X, pady=(0, 14))

        input_title = ttk.Label(
            input_card,
            text="Input",
            style="Section.TLabel"
        )
        input_title.grid(row=0, column=0, sticky="w")

        # Main project input: AmbiX
        ambix_button = ttk.Button(
            input_card,
            text="Load AmbiX File",
            command=self.load_ambix,
            style="Accent.TButton"
        )
        ambix_button.grid(row=0, column=2, padx=(20, 0), sticky="e")

        # Testing input: already-decoded binaural WAV
        wav_button = ttk.Button(
            input_card,
            text="Load Binaural WAV",
            command=self.load_binaural_wav_file
        )
        wav_button.grid(row=1, column=2, padx=(20, 0), pady=(8, 0), sticky="e")

        file_label = ttk.Label(
            input_card,
            textvariable=self.selected_file,
            style="SmallInfo.TLabel"
        )
        file_label.grid(row=1, column=0, columnspan=2, sticky="w", pady=(10, 0))

        input_type_label = ttk.Label(
            input_card,
            textvariable=self.input_type,
            style="SmallInfo.TLabel"
        )
        input_type_label.grid(row=2, column=0, sticky="w", pady=(6, 0))

        output_type_label = ttk.Label(
            input_card,
            textvariable=self.output_type,
            style="SmallInfo.TLabel"
        )
        output_type_label.grid(row=2, column=1, sticky="w", padx=(30, 0), pady=(6, 0))

        pipeline_label = ttk.Label(
            input_card,
            textvariable=self.pipeline_status,
            style="SmallInfo.TLabel"
        )
        pipeline_label.grid(row=3, column=0, columnspan=2, sticky="w", pady=(6, 0))

        input_card.columnconfigure(0, weight=1)
        input_card.columnconfigure(1, weight=1)

    def create_playback_card(self, parent):
        playback_card = ttk.Frame(parent, style="Card.TFrame", padding=18)
        playback_card.pack(fill=tk.X, pady=(0, 14))

        playback_title = ttk.Label(
            playback_card,
            text="Playback Control",
            style="Section.TLabel"
        )
        playback_title.grid(row=0, column=0, columnspan=5, sticky="w")

        self.play_button = ttk.Button(
            playback_card,
            text="Play / Resume",
            command=self.play
        )
        self.play_button.grid(row=1, column=0, padx=(0, 8), pady=(16, 12))

        self.pause_button = ttk.Button(
            playback_card,
            text="Pause",
            command=self.pause
        )
        self.pause_button.grid(row=1, column=1, padx=8, pady=(16, 12))

        self.stop_button = ttk.Button(
            playback_card,
            text="Stop",
            command=self.stop
        )
        self.stop_button.grid(row=1, column=2, padx=8, pady=(16, 12))

        self.loop_check = ttk.Checkbutton(
            playback_card,
            text="Loop",
            variable=self.loop_value,
            command=self.set_loop
        )
        self.loop_check.grid(row=1, column=3, padx=18, pady=(16, 12))

        time_label = ttk.Label(
            playback_card,
            textvariable=self.time_text,
            style="Time.TLabel"
        )
        time_label.grid(row=1, column=4, sticky="e", pady=(16, 12))

        playback_card.columnconfigure(4, weight=1)

        self.progress_slider = ttk.Scale(
            playback_card,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            variable=self.progress_value,
            command=self.on_progress_drag
        )
        self.progress_slider.grid(
            row=2,
            column=0,
            columnspan=5,
            sticky="ew",
            pady=(6, 4)
        )

        self.progress_slider.bind("<ButtonPress-1>", self.on_progress_press)
        self.progress_slider.bind("<ButtonRelease-1>", self.on_progress_release)

    def create_settings_card(self, parent):
        settings_card = ttk.Frame(parent, style="Card.TFrame", padding=18)
        settings_card.pack(fill=tk.X, pady=(0, 14))

        settings_title = ttk.Label(
            settings_card,
            text="Settings",
            style="Section.TLabel"
        )
        settings_title.grid(row=0, column=0, columnspan=4, sticky="w")

        volume_label = ttk.Label(
            settings_card,
            text="Volume",
            background="white",
            font=("Arial", 10, "bold")
        )
        volume_label.grid(row=1, column=0, sticky="w", pady=(16, 0))

        self.volume_slider = ttk.Scale(
            settings_card,
            from_=0.0,
            to=1.0,
            orient=tk.HORIZONTAL,
            variable=self.volume_value,
            command=self.set_volume
        )
        self.volume_slider.grid(
            row=1,
            column=1,
            sticky="ew",
            padx=(14, 18),
            pady=(16, 0)
        )

        self.volume_display = ttk.Label(
            settings_card,
            text="1.00",
            background="white",
            width=6,
            font=("Consolas", 10)
        )
        self.volume_display.grid(row=1, column=2, sticky="w", pady=(16, 0))

        offset_label = ttk.Label(
            settings_card,
            text="Start Offset (s)",
            background="white",
            font=("Arial", 10, "bold")
        )
        offset_label.grid(row=2, column=0, sticky="w", pady=(16, 0))

        self.offset_entry = ttk.Entry(
            settings_card,
            textvariable=self.offset_value,
            width=12
        )
        self.offset_entry.grid(row=2, column=1, sticky="w", padx=(14, 0), pady=(16, 0))

        offset_button = ttk.Button(
            settings_card,
            text="Set Offset",
            command=self.set_offset
        )
        offset_button.grid(row=2, column=2, sticky="w", padx=(12, 0), pady=(16, 0))

        settings_card.columnconfigure(1, weight=1)

    def create_info_card(self, parent):
        info_card = ttk.Frame(parent, style="Card.TFrame", padding=18)
        info_card.pack(fill=tk.BOTH, expand=True)

        info_title = ttk.Label(
            info_card,
            text="Signal Information",
            style="Section.TLabel"
        )
        info_title.pack(anchor="w")

        text_frame = ttk.Frame(info_card, style="Card.TFrame")
        text_frame.pack(fill=tk.BOTH, expand=True, pady=(12, 0))

        self.info_text = tk.Text(
            text_frame,
            height=10,
            bg="white",
            fg="#1f1f1f",
            font=("Consolas", 10),
            relief="flat",
            wrap="word"
        )
        self.info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(
            text_frame,
            orient=tk.VERTICAL,
            command=self.info_text.yview
        )
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.info_text.configure(yscrollcommand=scrollbar.set)
        self.info_text.configure(state="disabled")

        self.write_info_text("No audio loaded.")

    # ==============================================================
    # General GUI state
    # ==============================================================

    def set_controls_enabled(self, enabled: bool):
        state = tk.NORMAL if enabled else tk.DISABLED

        for widget in [
            self.play_button,
            self.pause_button,
            self.stop_button,
            self.progress_slider,
        ]:
            widget.configure(state=state)

    def reset_progress_display(self):
        self.progress_slider.configure(to=max(self.player.get_duration(), 0.01))
        self.progress_value.set(0.0)
        self.time_text.set(
            f"{self.format_time(0.0)} / {self.format_time(self.player.get_duration())}"
        )

    # ==============================================================
    # Loading functions
    # ==============================================================

    def load_ambix(self):
        """
        Placeholder for the final project input.

        Final intended chain:
        AmbiX file -> Ambisonics loader -> HRTF/Decoder -> pyfar.Signal -> GUI/Player

        At the current stage, this button only selects the file and displays
        basic information. Full AmbiX decoding should be connected later.
        """

        file_path = filedialog.askopenfilename(
            title="Select AmbiX / Ambisonics WAV file",
            filetypes=[
                ("Ambisonics WAV files", "*.wav"),
                ("All files", "*.*")
            ]
        )

        if not file_path:
            return

        # UI feedback
        file_name = os.path.basename(file_path)
        self.selected_file.set(f"AmbiX file selected: {file_name}")
        self.input_type.set("Input type: AmbiX / multichannel WAV")
        self.output_type.set("Output: not decoded yet")
        self.pipeline_status.set(
            "Pipeline: AmbiX loading is prepared; decoder connection is the next step"
        )

        self.set_controls_enabled(False)

        # read GUI state on main thread
        requested_block_size = self.get_block_size()
        requested_gain = float(self.volume_value.get())

        # a worker thread that handles the heavy processing and audio playback
        def worker(path, block_size, gain):
            try:
                # load the ambisonics file
                ambix = AmbisonicsFile(path, chunk_size=block_size)
                if not ambix.is_valid():
                    raise ValueError("Selected file is not a valid AmbiX file for its channel count.")
                
                # load the hrtf
                hrtf = HRTF("FABIAN_HRIR_measured_HATO_0.sofa")

                # create SH using the file's order and samplerate
                sh = SphericalHarmonics(hrtf=hrtf,
                                        sampling_rate=ambix.get_samplerate(),
                                        ambi_order=ambix.get_order())
                # create binaural streaming player
                binaural = AudioPlayer(ambi_file=ambix, sh=sh, gain=gain)

                # put success result for main threa to consume
                self._backend_load_queue.put(("success", binaural, path))

            except Exception as e:
                # put error for main thread to consume
                self._backend_load_queue.put(("error", str(e), file_path))

        # start the thread
        threading.Thread(target=worker, 
                         args=(file_path, requested_block_size, requested_gain), 
                         daemon=True).start()

    # threadsafe handling of starting the audio player
    def _process_backend_load_queue(self):
        try:
            while not self._backend_load_queue.empty():
                typ, payload, path = self._backend_load_queue.get_nowait()

                # on success
                if typ == "success":
                    binaural = payload
                    try:
                        self.player = binaural
                    except Exception as e:
                        # something went wrong
                        print("Something went wrong assigning the player")
                        print(e)

                    # set loaded status in player
                    self.player.set_loaded(True)

                    # get info
                    file_name = os.path.basename(path)
                    self.selected_file.set(f"AmbiX: {file_name}")
                    self.input_type.set("Input type: AmbiX / multichannel WAV")
                    self.output_type.set("Output: binaural streaming")
                    self.pipeline_status.set("Pipeline: AmbiX -> Decoder -> Streaming player")

                    # print info
                    print(f"Successfully loaded Ambix file {path}")

                    # update gui
                    self.reset_progress_display()
                    self.set_controls_enabled(True)
                    self.update_info()
                else:
                    # something went wrong!
                    err = payload
                    messagebox.showerror("Load error", err)
                    self.pipeline_status.set("Pipeline: load failed")
                    self.set_controls_enabled(False)
                    self.update_info()
                    print("Something went wrong loading the file")
        except Exception as e:
            # something went wrong
            print("Something went wrong in the worker thread:")
            print(e)
        finally:
            self.root.after(100, self._process_backend_load_queue)

    def load_binaural_wav_file(self):
        """
        Load an already-decoded binaural/stereo WAV file.

        This is mainly for standalone playback tests and for checking
        decoder output files such as written_binaural_ls.wav.
        """

        file_path = filedialog.askopenfilename(
            title="Select binaural/stereo WAV file",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )

        if not file_path:
            return

        try:
            self.player.load_wav(file_path)

            file_name = os.path.basename(file_path)
            self.selected_file.set(f"Binaural WAV: {file_name}")
            self.input_type.set("Input type: decoded WAV file")
            self.output_type.set("Output: headphone stereo playback")
            self.pipeline_status.set("Pipeline: WAV -> AudioPlayer -> headphone playback")

            self.reset_progress_display()
            self.set_controls_enabled(True)
            self.update_info()

        except Exception as error:
            print(error)
            messagebox.showerror("Error", str(error))

    def load_pyfar_signal_to_gui(self, signal, name="Decoder output"):
        """
        Load a pyfar.Signal directly into the GUI player.
        This is used for decoder-to-GUI integration.
        """

        try:
            self.player.load_pyfar_signal(signal)

            self.selected_file.set(f"Source: {name}")
            self.input_type.set("Input type: pyfar.Signal from decoder")
            self.output_type.set("Output: binaural stereo signal")
            self.pipeline_status.set(
                "Pipeline: Decoder -> pyfar.Signal -> GUI -> AudioPlayer"
            )

            self.reset_progress_display()
            self.set_controls_enabled(True)
            self.update_info()

        except Exception as error:
            print(error)
            messagebox.showerror("Error", str(error))

    # ==============================================================
    # Playback control
    # ==============================================================

    def play(self):
        try:
            self.player.play()
            self.update_info()
        except Exception as error:
            print(error)
            messagebox.showerror("Error", str(error))

    def pause(self):
        try:
            self.player.pause()
            self.update_info()
        except Exception as error:
            print(error)
            messagebox.showerror("Error", str(error))

    def stop(self):
        try:
            self.player.stop()
            self.progress_value.set(0.0)
            self.update_info()
        except Exception as error:
            print(error)
            messagebox.showerror("Error", str(error))

    def set_volume(self, value):
        try:
            volume = float(value)
            self.player.set_volume(volume)
            self.volume_display.configure(text=f"{volume:.2f}")
            self.update_info()
        except Exception as error:
            print(error)
            messagebox.showerror("Error", str(error))

    def set_offset(self):
        try:
            seconds = float(self.offset_value.get())
            self.player.set_start_offset(seconds)
            self.progress_value.set(self.player.get_current_time())
            self.update_info()
        except ValueError:
            print(error)
            messagebox.showerror("Error", "Please enter a valid number for offset.")
        except Exception as error:
            print(error)
            messagebox.showerror("Error", str(error))

    def set_loop(self):
        try:
            self.player.set_loop(self.loop_value.get())
            self.update_info()
        except Exception as error:
            print(error)
            messagebox.showerror("Error", str(error))

    # ==============================================================
    # Progress bar
    # ==============================================================

    def on_progress_press(self, event):
        self.is_dragging_progress = True

    def on_progress_drag(self, value):
        if not self.player.is_loaded:
            return

        if self.is_dragging_progress:
            seconds = float(value)
            self.time_text.set(
                f"{self.format_time(seconds)} / {self.format_time(self.player.get_duration())}"
            )

    def on_progress_release(self, event):
        if not self.player.is_loaded:
            self.is_dragging_progress = False
            return

        try:
            seconds = float(self.progress_value.get())
            self.player.seek_to(seconds)
            self.update_info()
        except Exception as error:
            print(error)
            messagebox.showerror("Error", str(error))
        finally:
            self.is_dragging_progress = False

    # ==============================================================
    # Info update
    # ==============================================================

    def build_info_text(self):
        if not self.player.is_loaded:
            return (
                "No playable audio loaded.\n\n"
                f"{self.input_type.get()}\n"
                f"{self.output_type.get()}\n"
                f"{self.pipeline_status.get()}"
            )

        return (
            f"{self.player.get_info_text()}\n\n"
            "GUI input mode:\n"
            f"{self.input_type.get()}\n"
            f"{self.output_type.get()}\n"
            f"{self.pipeline_status.get()}\n\n"
            "Integration note:\n"
            "The GUI supports two playback paths:\n"
            "1. Load Binaural WAV for standalone playback tests.\n"
            "2. Receive decoder output directly as a pyfar.Signal.\n\n"
            "The future final path is:\n"
            "AmbiX file -> Ambisonics loader -> HRTF / SphericalHarmonics decoder "
            "-> binaural pyfar.Signal -> AudioPlayer."
        )

    def write_info_text(self, text):
        self.status_text.set(text)

        if hasattr(self, "info_text"):
            self.info_text.configure(state="normal")
            self.info_text.delete("1.0", tk.END)
            self.info_text.insert(tk.END, text)
            self.info_text.configure(state="disabled")

    def update_info(self):
        self.write_info_text(self.build_info_text())

    def update_gui_loop(self):
        """
        Update GUI regularly.
        This keeps current time and progress bar in sync during playback.
        """
        if not self.player is None:
            if self.player.is_loaded:
                current_time = self.player.get_current_time()
                duration = self.player.get_duration()

                self.time_text.set(
                    f"{self.format_time(current_time)} / {self.format_time(duration)}"
                )

                if not self.is_dragging_progress:
                    self.progress_value.set(current_time)

                self.update_info()

        self.root.after(250, self.update_gui_loop)

    @staticmethod
    def format_time(seconds: float):
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes:02d}:{secs:05.2f}"

    # ==============================================================
    # App lifecycle
    # ==============================================================

    def on_close(self):
        if self.player is not None:
            self.player.stop()
        self.root.destroy()

    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.mainloop()

    def get_block_size(self):
        return self.block_size
