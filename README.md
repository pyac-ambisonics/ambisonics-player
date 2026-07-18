# Ambisonics Player

Python GUI prototype for loading AmbiX Ambisonics WAV files, decoding them to binaural stereo with an HRTF data set, and playing them back with basic transport controls.

## Current Features

- Load AmbiX / multichannel WAV files.
- Detect Ambisonics order from channel count, with optional order selection.
- Stream audio in chunks instead of loading the complete file into RAM.
- Decode Ambisonics to binaural stereo using spherical-harmonic HRTF processing.
- Play, pause, stop, seek, loop, and volume control.
- Select block size before loading a file.
- Select headphone compensation filters from `resources/Headphones`.
- Select a custom HRTF SOFA file, or use the default FABIAN HRTF fallback.
- Display live signal and decoder information in the GUI.
- Manual yaw / pitch / roll rotation prototype.
- Demo head-tracking mode with a synchronized head-direction visualizer.

## Current Limitations

- Hardware head tracking is not fully implemented yet.
- The GUI currently provides manual scene rotation and a simulated demo tracker.
- If the special `shroom.utils` backend is unavailable, MagLS preprocessing falls back to LS preprocessing and rotation falls back to identity rotation.
- Standalone binaural WAV playback was removed from the GUI because the current playback backend is focused on AmbiX streaming.

## Project Structure

```text
ambisonics-player/
  pyproject.toml               packaging metadata (makes playamb pip-installable)
  requirements.txt             plain dependency list (alternative to pyproject)
  src/
    run_playamb_gui.py         GUI application entry point
    resources/                 HRTF data, headphone filters, rotation matrices
    playamb/                   installable Python package
      audio/data/ambifile.py   AmbiX WAV loading, validation, streaming, seeking
      audio/engine/player.py   streaming playback, transport controls, overlap-add
      audio/engine/hrtf.py     HRTF loading, preprocessing, headphone filters
      audio/engine/spherical.py  spherical-harmonic binaural decoding, rotation hook
      audio/rotation/          rotation matrices, orientation state, tracker skeleton
      gui/                     Tkinter GUI and head-direction visualizer
      utils/utils.py           channel/order and FFT helper functions
  tests/                       order/filter rendering tests, MagLS validation notebook
  docs/                        generated API documentation
```

## Setup

Open a terminal in the project folder:

```powershell
cd C:\Users\HP\Documents\GitHub\ambisonics-player
```

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run this once in the same terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

Install the project (this pulls in all dependencies and makes the `playamb`
package importable):

```powershell
python -m pip install --upgrade pip
python -m pip install -e .
```

If you encounter an error relative to the installation of ```python-rtmidi``` upon excecuting the last command, you need to install Microsoft Visual C++ 14.x. You can download the installer from here: https://visualstudio.microsoft.com/de/downloads/?q=build+tools. Once done, run ```pip install -r requirements.txt``` again.

## Run

```powershell
python src\run_playamb_gui.py
```

The GUI will open. Choose the Ambisonics order, block size, HRTF/headphone settings, then click `Load AmbiX File`.

## Using playamb as a Library (GUI Bypass)

All decoding functionality lives in the `playamb` package and can be used from
any other Python project or script without starting the GUI.

### Install into your environment

From the environment of your own project, install this repository in editable
mode (editable mode is currently required: the bundled HRTF and headphone data
in `src/resources` is resolved relative to the source tree):

```powershell
python -m pip install -e C:\path\to\ambisonics-player
```

After that, `import playamb` works from any directory.

### Example 1 — offline rendering to a binaural WAV (no audio device needed)

```python
from playamb import AmbisonicsFile, HRTF, SphericalHarmonics, render_to_binaural_file

ambi = AmbisonicsFile("scene_ambix.wav")    # order auto-detected from channel count
hrtf = HRTF()                               # default FABIAN HRTF
hrtf.load_hp_filter("Sennheiser HD650")     # optional headphone compensation
sh = SphericalHarmonics(hrtf=hrtf, sampling_rate=ambi.get_samplerate(),
                        ambi_order=ambi.order, preprocess="MagLS")

# streams chunk-by-chunk: memory stays 0(chunk) even for hours-long files
render_to_binaural_file(ambi, sh, "scene_binaural.wav")
```

### Example 2 — real-time playback with transport control

```python
from playamb import AmbisonicsFile, HRTF, SphericalHarmonics, AudioPlayer

ambi = AmbisonicsFile("scene_ambix.wav", chunk_size=2048)
sh = SphericalHarmonics(hrtf=HRTF(), sampling_rate=ambi.get_samplerate(),
                        ambi_order=ambi.order, preprocess="MagLS")

player = AudioPlayer(ambi, sh)
player.play()               # non-blocking
# ... player.pause() / player.resume() / player.seek_to(seconds)
# ... player.set_volume(0.5) / player.set_loop(True)
player.stop()
player.close()
```

### Key parameters

| Parameter    | Where                | Meaning                                                          |
|--------------|----------------------|------------------------------------------------------------------|
| `chunk_size` | `AmbisonicsFile`     | samples per streaming block (32-8192, rounded to a power of two) |
| `order`      | `AmbisonicsFile`     | Ambisonics order; `None` = auto-detect from channel count        |
| `path`       | `HRTF`               | custom SOFA file path; `None` = bundled FABIAN HRTF              |
| `preprocess` | `SphericalHarmonics` | `"MagLS"` (falls back to `"LS"` if `shroom` is unavailable)      |
| `gain`       | `AudioPlayer`        | initial playback gain in `[0.0, 1.0]`                            |

Note: `import playamb` also imports the GUI module, so `tkinter` must be
available (it ships with the standard CPython installer).

### Runnable examples

Self-contained example scripts live in `examples/` (run from the repository
root after `pip install -e .`):

```powershell
python examples\make_test_signal.py test_ambix.wav                # synthesize test material
python examples\render_to_binaural.py test_ambix.wav out.wav --headphone "Sennheiser HD650"
python examples\playback_demo.py test_ambix.wav --duration 10
```

Each script has `--help` describing all options.

## Head Tracking Status

The current code does not implement full hardware head tracking. The GUI includes:

- Manual yaw / pitch / roll scene rotation.
- A simulated demo tracker that sweeps the yaw angle smoothly.
- A synchronized head-direction visualizer.
- A rotation-backend status display.

If the rotation backend is available, the demo orientation can be passed into the spherical-harmonic rotation hook. If not, the GUI still shows the visualizer and reports that audio rotation is falling back to identity rotation.

For full head tracking, the next implementation step should be:

1. Add a tracker input module, for example OSC or PyHeadTracker.
2. Store the latest yaw / pitch / roll values in a thread-safe orientation state.
3. Update the rotation matrix at audio block boundaries.
4. Add smoothing and latency checks.
5. Add a GUI visualizer that displays the tracked head direction.

## Demo Notes

For a stable project presentation:

- Use a short AmbiX test file with a clear sound direction.
- Load the file through the GUI and demonstrate play, pause, seek, loop, and volume.
- Show the live `Signal Information` panel to confirm order, channel count, duration, current time, HRTF, and headphone filter.
- Present manual rotation and demo tracking as prototypes, not as completed hardware head tracking.
- Use the head-direction visualizer to show how future tracker data would drive yaw / pitch / roll updates.
