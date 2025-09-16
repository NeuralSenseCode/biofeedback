Setup and run (Windows PowerShell)

1) Create & activate the virtual environment (already created by the assistant into `.venv`):

```powershell
# create venv (if you haven't already)
python -m venv .venv
# activate
. .venv\Scripts\Activate.ps1
```

2) Install dependencies (from `requirements.txt`):

```powershell
.venv\Scripts\python -m pip install -r requirements.txt
```

3) Run the demo (software-only synthetic mode):

```powershell
.venv\Scripts\python eeg_bandpass_viewer.py --demo
```

Notes:
- The project assumes Matplotlib TkAgg backend on Windows. If running headless (CI), change backend to `Agg`.
- If using a Plux device, stream Plux data to OpenSignals/LSL and run without `--demo` while passing `--stream-name` and `--expect-ch` as needed.
