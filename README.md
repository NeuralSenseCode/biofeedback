# NeuralSense Biofeedback

## Project Overview

This repository provides Python tools for EEG biofeedback visualization and experimentation. It includes:
- **eeg_bandpass_viewer.py**: A real-time dashboard for visualizing EEG band activity (Calm & Focus metrics).
- **eeg_scroller_game.py**: An interactive game controlled by EEG signals, featuring blink calibration and real-time feedback.
- Additional utilities for data inspection and debugging.

## Getting Started

### Prerequisites
- Python 3.8+
- [pip](https://pip.pypa.io/en/stable/)

### Installation
1. Clone this repository:
	```sh
	git clone <your-repo-url>
	cd Biofeedback
	```
2. Install dependencies:
	```sh
	pip install numpy scipy matplotlib pylsl
	```

### Running the Dashboard (Demo Mode)
No EEG hardware required:
```sh
python eeg_bandpass_viewer.py --demo
```

### Running the EEG Scroller Game (Demo Mode)
No EEG hardware required:
```sh
python eeg_scroller_game.py --demo
```

### Running with Real EEG Hardware
- Connect your EEG device and ensure an LSL stream is available (e.g., OpenSignals, 2 channels).
- Example:
  ```sh
  python eeg_bandpass_viewer.py --stream-name "OpenSignals" --expect-ch 2
  python eeg_scroller_game.py --stream-name "OpenSignals" --expect-ch 2
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
