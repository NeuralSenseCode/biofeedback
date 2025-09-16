# NeuralSense Biofeedback

## Project Overview

This project contains Python utilities for EEG biofeedback visualization and data collection, including a real-time dashboard and a demo game using EEG signals.

## Getting Started

- Install requirements: `pip install numpy scipy pylsl`
- Run in demo mode: `python eeg_bandpass_viewer.py --demo`
- Run the game: `python eeg_scroller_game.py --demo`

## Project Learnings and Closing Notes

1. Blink detection is the most reliable trigger.
2. Theta activity works well too, but is hard to reliably self-direct. Investigate the use of a low pass filter.
3. Graphics are simple and better graphics would definitely improve appeal.
4. Simplistic nature of the game makes it less appealing. Discover ways to add complexity to the game mechanic.

---

Thank you for contributing to this project!
