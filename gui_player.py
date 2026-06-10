import tkinter as tk
from tkinter import filedialog, messagebox

from audio_player import AudioPlayer


class AudioPlayerGUI:
    """
    Simple GUI prototype for the Ambisonics Player.

    Supports:
    - Load WAV
    - Play / Resume
    - Pause
    - Stop
    - Real-time volume control
    - Start offset
    - Loop
    - Info display with current time
    """

    def __init__(self):
        self.player = AudioPlayer()

        self.root = tk.Tk()
        self.root.title("Ambisonics Player - GUI Prototype")
        self.root.geometry("680x500")

        self.selected_file = tk.StringVar(value="No file selected")
        self.volume_value = tk.DoubleVar(value=1.0)
        self.offset_value = tk.StringVar(value="0")
        self.loop_value = tk.BooleanVar(value=False)

        self.create_widgets()
        self.update_info_loop()

    def create_widgets(self):
        title = tk.Label(
            self.root,
            text="Ambisonics Player - GUI Prototype",
            font=("Arial", 18, "bold")
        )
        title.pack(pady=15)

        file_label = tk.Label(
            self.root,
            textvariable=self.selected_file,
            wraplength=600,
            font=("Arial", 10)
        )
        file_label.pack(pady=5)

        load_button = tk.Button(
            self.root,
            text="Load WAV",
            width=15,
            command=self.load_wav_file
        )
        load_button.pack(pady=8)

        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)

        play_button = tk.Button(
            button_frame,
            text="Play / Resume",
            width=14,
            command=self.play
        )
        play_button.grid(row=0, column=0, padx=6)

        pause_button = tk.Button(
            button_frame,
            text="Pause",
            width=12,
            command=self.pause
        )
        pause_button.grid(row=0, column=1, padx=6)

        stop_button = tk.Button(
            button_frame,
            text="Stop",
            width=12,
            command=self.stop
        )
        stop_button.grid(row=0, column=2, padx=6)

        volume_label = tk.Label(
            self.root,
            text="Volume",
            font=("Arial", 11, "bold")
        )
        volume_label.pack(pady=(15, 0))

        volume_slider = tk.Scale(
            self.root,
            from_=0.0,
            to=1.0,
            resolution=0.05,
            orient=tk.HORIZONTAL,
            variable=self.volume_value,
            command=self.set_volume,
            length=380
        )
        volume_slider.pack()

        offset_frame = tk.Frame(self.root)
        offset_frame.pack(pady=12)

        offset_label = tk.Label(offset_frame, text="Start Offset (seconds):")
        offset_label.grid(row=0, column=0, padx=5)

        offset_entry = tk.Entry(
            offset_frame,
            textvariable=self.offset_value,
            width=10
        )
        offset_entry.grid(row=0, column=1, padx=5)

        offset_button = tk.Button(
            offset_frame,
            text="Set Offset",
            command=self.set_offset
        )
        offset_button.grid(row=0, column=2, padx=5)

        loop_checkbox = tk.Checkbutton(
            self.root,
            text="Loop playback",
            variable=self.loop_value,
            command=self.set_loop
        )
        loop_checkbox.pack(pady=5)

        self.info_label = tk.Label(
            self.root,
            text="Status: No audio loaded.",
            justify=tk.LEFT,
            font=("Arial", 10),
            bg="#f2f2f2",
            width=78,
            height=8,
            anchor="nw"
        )
        self.info_label.pack(pady=15)

    def load_wav_file(self):
        file_path = filedialog.askopenfilename(
            title="Select WAV file",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )

        if not file_path:
            return

        try:
            self.player.load_wav(file_path)
            self.selected_file.set(file_path)
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
            self.update_info()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def set_volume(self, value):
        try:
            self.player.set_volume(float(value))
            self.update_info()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def set_offset(self):
        try:
            seconds = float(self.offset_value.get())
            self.player.set_start_offset(seconds)
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

    def update_info(self):
        self.info_label.config(text=self.player.get_info_text())

    def update_info_loop(self):
        """
        Update the info display regularly.
        This allows current playback time to update while playing.
        """

        self.update_info()
        self.root.after(300, self.update_info_loop)

    def on_close(self):
        """
        Stop audio when closing the GUI window.
        """

        self.player.stop()
        self.root.destroy()

    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.mainloop()


if __name__ == "__main__":
    app = AudioPlayerGUI()
    app.run()