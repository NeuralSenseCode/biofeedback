
# EEG Scroller (Arcade)

A starter template for a polished 2D EEG scroller game built with **Arcade**.

## Quick start
```bash
# 1) Create venv and install
python -m venv .venv && . .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e .

# 2) Run
python main.py
```

EEG is stubbed by a fake generator. Plug in LSL by filling `game/eeg/lsl_inlet.py`.
Hotkeys: F1 toggle EEG, SPACE to jump.
