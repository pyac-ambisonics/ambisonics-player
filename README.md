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
  resources/                  headphone compensation filters
  ambisonics_file_English.py   AmbiX WAV loading, validation, streaming, seeking
  audio_player.py              streaming playback, transport controls, overlap-add
  gui_player_v2_new.py         current GUI
  head_tracking.py             orientation state, demo tracker, OSC tracker skeleton
  hrtf.py                      HRTF loading and preprocessing
  main.py                      application entry point
  requirements.txt             Python dependencies
  spherical.py                 spherical-harmonic HRTF decoding and rotation hook
  utils.py                     channel/order and FFT helper functions
  mido
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

Install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run

```powershell
python main.py
```

The GUI will open. Choose the Ambisonics order, block size, HRTF/headphone settings, then click `Load AmbiX File`.

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
