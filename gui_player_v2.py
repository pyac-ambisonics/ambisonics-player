import os
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk

from audio_player import AudioPlayer


class AudioPlayerGUI:
    """
    Improved GUI prototype for the Ambisonics Player.

    Features:
    - Load WAV file
    - Play / Resume
    - Pause
    - Stop
    - Real-time volume control
    - Loop playback
    - Start offset
    - Draggable progress bar
    - Current time display
    - Signal information display
    """

    def __init__(self):
        self.player = AudioPlayer()

        self.root = tk.Tk()
        self.root.title("Ambisonics Player - GUI Prototype")
        self.root.geometry("820x560")
        self.root.minsize(760, 520)

        self.selected_file = tk.StringVar(value="No file selected")
        self.status_text = tk.StringVar(value="No audio loaded.")
        self.time_text = tk.StringVar(value="00:00.00 / 00:00.00")
        self.volume_value = tk.DoubleVar(value=1.0)
        self.offset_value = tk.StringVar(value="0")
        self.loop_value = tk.BooleanVar(value=False)
        self.progress_value = tk.DoubleVar(value=0.0)

        self.is_dragging_progress = False

        self.setup_style()
        self.create_widgets()
        self.update_gui_loop()

    def setup_style(self):
        self.root.configure(bg="#f5f6f8")

        style = ttk.Style()
        style.theme_use("clam")

        style.configure(
            "TFrame",
            background="#f5f6f8"
        )

        style.configure(
            "Card.TFrame",
            background="white",
            relief="flat"
        )

        style.configure(
            "Title.TLabel",
            background="#f5f6f8",
            foreground="#14395b",
            font=("Arial", 22, "bold")
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
            font=("Arial", 12, "bold")
        )

        style.configure(
            "Info.TLabel",
            background="white",
            foreground="#222222",
            font=("Consolas", 10)
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
            text="Player / GUI Prototype for binaural playback",
            style="Subtitle.TLabel"
        )
        subtitle.pack(anchor="w", pady=(2, 18))

        # File card
        file_card = ttk.Frame(main, style="Card.TFrame", padding=16)
        file_card.pack(fill=tk.X, pady=(0, 14))

        file_title = ttk.Label(
            file_card,
            text="Input",
            style="Section.TLabel"
        )
        file_title.grid(row=0, column=0, sticky="w")

        self.file_label = ttk.Label(
            file_card,
            textvariable=self.selected_file,
            background="white",
            foreground="#444444",
            font=("Arial", 10)
        )
        self.file_label.grid(row=1, column=0, sticky="w", pady=(8, 0))

        load_button = ttk.Button(
            file_card,
            text="Load WAV",
            command=self.load_wav_file,
            style="Accent.TButton"
        )
        load_button.grid(row=0, column=1, rowspan=2, padx=(20, 0), sticky="e")

        file_card.columnconfigure(0, weight=1)

        # Playback card
        playback_card = ttk.Frame(main, style="Card.TFrame", padding=16)
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
        self.play_button.grid(row=1, column=0, padx=(0, 8), pady=(14, 12))

        self.pause_button = ttk.Button(
            playback_card,
            text="Pause",
            command=self.pause
        )
        self.pause_button.grid(row=1, column=1, padx=8, pady=(14, 12))

        self.stop_button = ttk.Button(
            playback_card,
            text="Stop",
            command=self.stop
        )
        self.stop_button.grid(row=1, column=2, padx=8, pady=(14, 12))

        loop_check = ttk.Checkbutton(
            playback_card,
            text="Loop",
            variable=self.loop_value,
            command=self.set_loop
        )
        loop_check.grid(row=1, column=3, padx=18, pady=(14, 12))

        # Time display
        time_label = ttk.Label(
            playback_card,
            textvariable=self.time_text,
            background="white",
            foreground="#333333",
            font=("Consolas", 11, "bold")
        )
        time_label.grid(row=1, column=4, sticky="e", pady=(14, 12))

        playback_card.columnconfigure(4, weight=1)

        # Progress bar
        self.progress_slider = ttk.Scale(
            playback_card,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            variable=self.progress_value,
            command=self.on_progress_drag
        )
        self.progress_slider.grid(row=2, column=0, columnspan=5, sticky="ew", pady=(6, 4))

        self.progress_slider.bind("<ButtonPress-1>", self.on_progress_press)
        self.progress_slider.bind("<ButtonRelease-1>", self.on_progress_release)

        # Settings card
        settings_card = ttk.Frame(main, style="Card.TFrame", padding=16)
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
        volume_label.grid(row=1, column=0, sticky="w", pady=(14, 0))

        volume_slider = ttk.Scale(
            settings_card,
            from_=0.0,
            to=1.0,
            orient=tk.HORIZONTAL,
            variable=self.volume_value,
            command=self.set_volume
        )
        volume_slider.grid(row=1, column=1, sticky="ew", padx=(12, 18), pady=(14, 0))

        self.volume_display = ttk.Label(
            settings_card,
            text="1.00",
            background="white",
            width=5,
            font=("Consolas", 10)
        )
        self.volume_display.grid(row=1, column=2, sticky="w", pady=(14, 0))

        offset_label = ttk.Label(
            settings_card,
            text="Start Offset (s)",
            background="white",
            font=("Arial", 10, "bold")
        )
        offset_label.grid(row=2, column=0, sticky="w", pady=(14, 0))

        offset_entry = ttk.Entry(
            settings_card,
            textvariable=self.offset_value,
            width=10
        )
        offset_entry.grid(row=2, column=1, sticky="w", padx=(12, 0), pady=(14, 0))

        offset_button = ttk.Button(
            settings_card,
            text="Set Offset",
            command=self.set_offset
        )
        offset_button.grid(row=2, column=2, sticky="w", padx=(12, 0), pady=(14, 0))

        settings_card.columnconfigure(1, weight=1)

        # Info card
        info_card = ttk.Frame(main, style="Card.TFrame", padding=16)
        info_card.pack(fill=tk.BOTH, expand=True)

        info_title = ttk.Label(
            info_card,
            text="Signal Information",
            style="Section.TLabel"
        )
        info_title.pack(anchor="w")

        self.info_label = ttk.Label(
            info_card,
            textvariable=self.status_text,
            style="Info.TLabel",
            justify=tk.LEFT
        )
        self.info_label.pack(anchor="nw", fill=tk.BOTH, expand=True, pady=(10, 0))

        self.set_controls_enabled(False)

    def set_controls_enabled(self, enabled: bool):
        state = tk.NORMAL if enabled else tk.DISABLED

        for widget in [
            self.play_button,
            self.pause_button,
            self.stop_button,
            self.progress_slider,
        ]:
            widget.configure(state=state)

    def load_wav_file(self):
        file_path = filedialog.askopenfilename(
            title="Select WAV file",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )

        if not file_path:
            return

        try:
            self.player.load_wav(file_path)

            file_name = os.path.basename(file_path)
            self.selected_file.set(file_name)

            self.progress_slider.configure(to=max(self.player.get_duration(), 0.01))
            self.progress_value.set(0.0)

            self.set_controls_enabled(True)
            self.update_info()

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def play(self):
        try:
            self.player.play()
            self.update_info()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def pause(self):
        try:
            self.player.pause()
            self.update_info()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def stop(self):
        try:
            self.player.stop()
            self.progress_value.set(0.0)
            self.update_info()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def set_volume(self, value):
        try:
            volume = float(value)
            self.player.set_volume(volume)
            self.volume_display.configure(text=f"{volume:.2f}")
            self.update_info()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def set_offset(self):
        try:
            seconds = float(self.offset_value.get())
            self.player.set_start_offset(seconds)
            self.progress_value.set(self.player.get_current_time())
            self.update_info()
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number for offset.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def set_loop(self):
        try:
            self.player.set_loop(self.loop_value.get())
            self.update_info()
        except Exception as e:
            messagebox.showerror("Error", str(e))

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
        except Exception as e:
            messagebox.showerror("Error", str(e))
        finally:
            self.is_dragging_progress = False

    def update_info(self):
        self.status_text.set(self.player.get_info_text())

    def update_gui_loop(self):
        """
        Update GUI regularly.
        This keeps current time and progress bar in sync during playback.
        """

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

    def on_close(self):
        self.player.stop()
        self.root.destroy()

    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.mainloop()


if __name__ == "__main__":
    app = AudioPlayerGUI()
    app.run()