# Copilot / AI Agent instructions for NeuralSense Biofeedback
These notes give an AI coding agent the minimal, practical knowledge to be productive in this repository. Be conservative: only document patterns that are discoverable from code and scripts in the tree.

- Project intent
  - Small Python utilities for EEG biofeedback visualization and data collection. Main scripts:
    - `eeg_bandpass_viewer.py` — a live "Calm & Focus" dashboard; uses LSL (`pylsl`) when not in `--demo` mode.
    - `eeg_runner.py`, `neuro_debug_scope.py`, `neurobloom_lite.py`, `debug.py` — auxiliary tools (inspect for usage patterns).
- Big-picture architecture
  - Single-process Python scripts (no web server). The dashboard reads streams via Lab Streaming Layer (LSL) using `pylsl.StreamInlet` and updates a Matplotlib UI at ~10 Hz.
  - Data flow: LSL stream -> reader thread `reader_loop` -> `StreamBuffer.append_chunk` circular buffer -> main thread `CalmFocusDashboard.update_once()` snapshots buffer for metrics and plots.
  - Key classes/functions:
    - `StreamBuffer` (circular buffer): stores last N seconds of multi-channel samples. Methods: `append_chunk(arr)`, `snapshot()`.
    - `reader_loop(inlet, sbuf, stop_evt)`: pulls chunks from an LSL inlet and feeds `StreamBuffer`.
  - `CalmFocusDashboard`: constructs inlet (or demo mode), computes bandpowers with `scipy.signal.welch`, uses `Ema` for smoothing, and plots Calm/Focus series.
  - `synthetic_chunk(...)`: demo synthetic signal generator used when `--demo` is passed.

- Run / debug
  - Requirements (inferred from imports): Python 3.x, numpy, matplotlib (TkAgg backend), pylsl, scipy.
  - Quick run examples (powershell):
    - Demo mode (no hardware required): python .\eeg_bandpass_viewer.py --demo
    - Real LSL stream: python .\eeg_bandpass_viewer.py --stream-name "OpenSignals" --expect-ch 2
  - GUI backend: Matplotlib configured to `TkAgg`. On headless machines, tests or CI should avoid launching UI or switch backend.

- Conventions & patterns
  - Fixed sampling rate assumption: many functions assume `fs=200` (see top-level docstring and default args). Make changes carefully if other rates are needed.
  - Channel indexing: code uses channel 1 (index 1) as the signal of interest (F4-F3 difference). `StreamBuffer` stores channels in axis 0.
  - Thread-safety: `StreamBuffer` uses a `threading.Lock()`; prefer using `snapshot()` for consistent reads on the main thread.
  - Small numeric tolerances: bandpower uses 1e-9 denominators when computing ratios to avoid div0.

- Tests, linting, and quality gates
  - No test harness detected in repository root. When adding tests, prefer small unit tests for `bandpower`, `StreamBuffer`, `Ema`, and `compute_metrics` using deterministic synthetic signals (e.g., `synthetic_chunk`).

- Typical edits and low-risk change patterns
  - Add a `--no-gui` or `--headless` flag to run metrics-only mode (useful for CI or automated testing); snapshot + compute_metrics can be exercised without Matplotlib.
  - When changing window length or fs, remember to update `StreamBuffer.n` and time vector `t` and ensure `nperseg` bounds in `welch` calls remain valid.

- Integration points and external dependencies
  - LSL: `pylsl` expected to provide named streams. Code currently scans all resolved streams and matches on `.name()` and `.channel_count()`.
  - GUI: Matplotlib uses `TkAgg` explicitly. On Windows, this typically works, but remote/CI environments may require `Agg`.

- Example code snippets to reference local patterns
  - Circular buffer append loop (prefer this pattern when ingesting sample arrays):
    - See `StreamBuffer.append_chunk` in `eeg_bandpass_viewer.py`.
  - Pulling from LSL in background thread:
    - See `reader_loop` + `StreamInlet.pull_chunk` usage.
  - Bandpower with Welch and masking of frequencies:
    - See `bandpower(sig, fs, fmin, fmax).

- Safety & constraints for AI edits
  - Don't change assumptions about `fs=200` or channel layout without updating docstrings and tests.
  - Avoid making UI-blocking synchronous reads; maintain the reader thread and `snapshot()` pattern when adding new ingestion code.
  - For low-risk additions (new flags, tests, metrics-only mode), keep the existing code paths unchanged and add opt-in behavior.
- Where to look first when asked to modify behavior
  - Add signal-processing or metrics: modify `compute_metrics` in `CalmFocusDashboard`, and unit-test with `synthetic_chunk`.
  - Change stream handling or channel mapping: edit `StreamBuffer.append_chunk` and the `reader_loop` contract.
  - Modify plotting cadence or layout: edit `CalmFocusDashboard.__init__` and `update_once` (look for `dt_update`, `window_s`, `line_*` plotting objects).

If anything here is unclear or you'd like the document to mention additional files (for example `eeg_runner.py`), tell me which files to inspect and I'll iterate.

----

Please review and tell me if you'd like the document to include more examples, CI hooks, or an alternate metrics-only run mode.
# Copilot / AI Agent instructions for NeuralSense Biofeedback

These notes give an AI coding agent the minimal, practical knowledge to be productive in this repository. Be conservative: only document patterns that are discoverable from code and scripts in the tree.

- Project intent
  - Small Python utilities for EEG biofeedback visualization and data collection. Main scripts:
    - `eeg_bandpass_viewer.py` — a live "Calm & Focus" dashboard; uses LSL (pylsl) when not in `--demo` mode.
    - `eeg_runner.py`, `neuro_debug_scope.py`, `neurobloom_lite.py`, `debug.py` — auxiliary tools (inspect for usage patterns).

- Big-picture architecture
  - Single-process Python scripts (no web server). The dashboard reads streams via Lab Streaming Layer (LSL) using `pylsl.StreamInlet` and updates a Matplotlib UI at ~10 Hz.
  - Data flow: LSL stream -> reader thread `reader_loop` -> `StreamBuffer.append_chunk` circular buffer -> main thread `CalmFocusDashboard.update_once()` snapshots buffer for metrics and plots.
  - Key classes/functions:
    - `StreamBuffer` (circular buffer): stores last N seconds of multi-channel samples. Methods: `append_chunk(arr)`, `snapshot()`.
    - `reader_loop(inlet, sbuf, stop_evt)`: pulls chunks from an LSL inlet and feeds `StreamBuffer`.
    - `CalmFocusDashboard`: constructs inlet (or demo mode), computes bandpowers with `scipy.signal.welch`, uses `Ema` for smoothing, and plots Calm/Focus series.
    - `synthetic_chunk(...)`: demo synthetic signal generator used when `--demo` is passed.

- Run / debug
  - Requirements (inferred from imports): Python 3.x, numpy, matplotlib (TkAgg backend), pylsl, scipy.
  - Quick run examples (powershell):
    - Demo mode (no hardware required): python .\eeg_bandpass_viewer.py --demo
    - Real LSL stream: python .\eeg_bandpass_viewer.py --stream-name "OpenSignals" --expect-ch 2
  - GUI backend: Matplotlib configured to `TkAgg`. On headless machines, tests or CI should avoid launching UI or switch backend.

- Conventions & patterns
  - Fixed sampling rate assumption: many functions assume `fs=200` (see top-level docstring and default args). Make changes carefully if other rates are needed.
  - Channel indexing: code uses channel 1 (index 1) as the signal of interest (F4-F3 difference). `StreamBuffer` stores channels in axis 0.
  - Thread-safety: `StreamBuffer` uses a `threading.Lock()`; prefer using `snapshot()` for consistent reads on the main thread.
  - Small numeric tolerances: bandpower uses 1e-9 denominators when computing ratios to avoid div0.

- Tests, linting, and quality gates
  - No test harness detected in repository root. When adding tests, prefer small unit tests for `bandpower`, `StreamBuffer`, `Ema`, and `compute_metrics` using deterministic synthetic signals (e.g., `synthetic_chunk`).

- Typical edits and low-risk change patterns
  - Add a `--no-gui` or `--headless` flag to run metrics-only mode (useful for CI or automated testing); snapshot + compute_metrics can be exercised without Matplotlib.
  - When changing window length or fs, remember to update `StreamBuffer.n` and time vector `t` and ensure `nperseg` bounds in `welch` calls remain valid.

- Integration points and external dependencies
  - LSL: `pylsl` expected to provide named streams. Code currently scans all resolved streams and matches on `.name()` and `.channel_count()`.
  - GUI: Matplotlib uses `TkAgg` explicitly. On Windows, this typically works, but remote/CI environments may require `Agg`.

- Example code snippets to reference local patterns
  - Circular buffer append loop (prefer this pattern when ingesting sample arrays):
    - See `StreamBuffer.append_chunk` in `eeg_bandpass_viewer.py`.
  - Pulling from LSL in background thread:
    - See `reader_loop` + `StreamInlet.pull_chunk` usage.
  - Bandpower with Welch and masking of frequencies:
    - See `bandpower(sig, fs, fmin, fmax)`.

- Safety & constraints for AI edits
  - Don't change assumptions about `fs=200` or channel layout without updating docstrings and tests.
  - Avoid making UI-blocking synchronous reads; maintain the reader thread and `snapshot()` pattern when adding new ingestion code.
  - For low-risk additions (new flags, tests, metrics-only mode), keep the existing code paths unchanged and add opt-in behavior.

- Where to look first when asked to modify behavior
  - Add signal-processing or metrics: modify `compute_metrics` in `CalmFocusDashboard`, and unit-test with `synthetic_chunk`.
  - Change stream handling or channel mapping: edit `StreamBuffer.append_chunk` and the `reader_loop` contract.
  - Modify plotting cadence or layout: edit `CalmFocusDashboard.__init__` and `update_once` (look for `dt_update`, `window_s`, `line_*` plotting objects).

If anything here is unclear or you'd like the document to mention additional files (for example `eeg_runner.py`), tell me which files to inspect and I'll iterate.  

----

Please review and tell me if you'd like the document to include more examples, CI hooks, or an alternate metrics-only run mode.