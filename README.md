# NeuralSense Biofeedback

## Project Overview

This repository provides Python tools for EEG biofeedback visualization and experimentation. It includes:
- **eeg_bandpass_viewer.py**: A real-time dashboard for visualizing EEG band activity (Calm & Focus metrics).
- **eeg_scroller_game.py**: An interactive game controlled by EEG signals, featuring blink calibration and real-time feedback.
- Additional utilities for data inspection and debugging.

## Getting Started

### Prerequisites
- Windows, macOS or Linux with Python 3.8+
- [pip](https://pip.pypa.io/en/stable/)

### Quick start (Windows PowerShell)

1) Create & activate the virtual environment (this repo uses `.venv` by default):

```powershell
# create venv (if you haven't already)
python -m venv .venv
# activate
. .venv\Scripts\Activate.ps1
```

2) Install dependencies (from `requirements.txt` if present):

```powershell
.venv\Scripts\python -m pip install -r requirements.txt
# or install minimal deps
.venv\Scripts\python -m pip install numpy scipy matplotlib pylsl
```

3) Run the dashboard in demo (software-only) mode:

```powershell
.venv\Scripts\python eeg_bandpass_viewer.py --demo
```

4) Run the EEG scroller game in demo mode:

```powershell
.venv\Scripts\python eeg_scroller_game.py --demo
```

Notes:
- The project assumes Matplotlib's TkAgg backend on Windows by default. If running headless (CI), switch Matplotlib's backend to `Agg`.
- To run with a real device, stream your device to an LSL endpoint (for example OpenSignals) and run without `--demo`, e.g.:

```powershell
.venv\Scripts\python eeg_bandpass_viewer.py --stream-name "OpenSignals" --expect-ch 2
.venv\Scripts\python eeg_scroller_game.py --stream-name "OpenSignals" --expect-ch 2
```

## File Overview
- `eeg_bandpass_viewer.py`: Live Calm & Focus dashboard (Matplotlib UI, LSL or demo mode)
- `eeg_scroller_game.py`: EEG-controlled game with blink calibration and real-time feedback
- `eeg_runner.py`, `neuro_debug_scope.py`, `neurobloom_lite.py`, `debug.py`: Auxiliary tools for data inspection and debugging

## Project Learnings and Closing Notes

1. Blink detection is the most reliable trigger.
2. Theta activity works well too, but is hard to reliably self-direct. Investigate the use of a low pass filter.
3. Graphics are simple and better graphics would definitely improve appeal.
4. Simplistic nature of the game makes it less appealing. Discover ways to add complexity to the game mechanic.

---

Thank you for contributing to this project!
