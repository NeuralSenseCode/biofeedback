#!/usr/bin/env python3
"""
EEG Calm & Focus Dashboard

- Assumes fs = 200 Hz (fixed)
- Uses channel 1 (F4-F3 difference)
- Calm = alpha(8–12 Hz) / (alpha + HF(20–35 Hz))
- Focus = beta(15–25 Hz) / (alpha + theta(4–7 Hz))
- Rolling window plot (default 60s)

Usage:
    pip install numpy matplotlib pylsl scipy
    python eeg_calm_focus_dashboard.py --stream-name "OpenSignals" --expect-ch 2
    python eeg_calm_focus_dashboard.py --demo
"""

import argparse
import threading
import time
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from pylsl import StreamInlet, resolve_streams
from scipy.signal import welch


# ----------------------------
# Helpers
# ----------------------------
def bandpower(sig, fs, fmin, fmax):
    """Welch power in band [fmin,fmax]"""
    if len(sig) < int(0.5 * fs):
        return 0.0
    f, Pxx = welch(sig, fs=fs, nperseg=min(512, len(sig)))
    df = f[1] - f[0]
    mask = (f >= fmin) & (f <= fmax)
    return float(np.sum(Pxx[mask]) * df) if np.any(mask) else 0.0


class Ema:
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        self.y = None
    def update(self, v):
        if self.y is None:
            self.y = v
        else:
            self.y = self.alpha * v + (1 - self.alpha) * self.y
        return self.y


# ----------------------------
# Stream Buffer
# ----------------------------
class StreamBuffer:
    def __init__(self, fs=200, ch=2, window_sec=60):
        self.fs = fs
        self.ch = ch
        self.n = int(window_sec * fs)
        self.buf = np.zeros((self.ch, self.n), dtype=np.float32)
        self.t = np.linspace(-window_sec, 0.0, self.n, dtype=np.float32)
        self.idx = 0
        self.lock = threading.Lock()

    def append_chunk(self, arr):
        if arr.size == 0:
            return
        with self.lock:
            for s in arr:
                self.buf[:, self.idx] = s[:self.ch]
                self.idx = (self.idx + 1) % self.n

    def snapshot(self):
        with self.lock:
            if self.idx == 0:
                return self.t.copy(), self.buf.copy()
            rolled = np.roll(self.buf, -self.idx, axis=1)
            return self.t.copy(), rolled.copy()


def reader_loop(inlet, sbuf, stop_evt):
    while not stop_evt.is_set():
        try:
            chunk, ts = inlet.pull_chunk(timeout=0.1)
            if chunk:
                arr = np.array(chunk, dtype=np.float32)
                if arr.ndim == 1:
                    arr = arr[None, :]
                sbuf.append_chunk(arr)
            else:
                time.sleep(0.01)
        except Exception:
            time.sleep(0.05)


# ----------------------------
# Synthetic demo mode
# ----------------------------
def synthetic_chunk(fs=200, ch=2, dur=0.5):
    t = np.arange(int(fs * dur)) / fs
    # alpha dominant
    x = 20*np.sin(2*np.pi*10*t)
    # add beta bursts
    if np.random.rand() < 0.3:
        x += 15*np.sin(2*np.pi*20*t)
    # add noise
    x += 5*np.random.randn(len(t))
    sig = np.stack([np.zeros_like(x), x], axis=1)
    return sig.astype(np.float32)


# ----------------------------
# Dashboard
# ----------------------------
class CalmFocusDashboard:
    def __init__(self, stream_name, expect_ch, demo,
                 window_s=60, fs=200):
        self.fs = fs
        self.demo = demo
        if not demo:
            info = None
            for s in resolve_streams():
                if s.name() == stream_name and s.channel_count() == expect_ch:
                    info = s; break
            if not info:
                raise RuntimeError(f"No stream '{stream_name}' with {expect_ch} channels found.")
            ch = info.channel_count()
            print(f"[LSL] Connected to {info.name()} (Ch={ch}, Fs={info.nominal_srate()})")
            self.inlet = StreamInlet(info, max_buflen=60, max_chunklen=0, recover=True)
        else:
            ch = expect_ch
            self.inlet = None
            print("[DEMO] Synthetic signal mode.")

        self.sbuf = StreamBuffer(fs=fs, ch=ch, window_sec=window_s)
        self.ema_calm = Ema(alpha=0.2)
        self.ema_focus = Ema(alpha=0.2)

        self.stop_evt = threading.Event()
        if self.inlet is not None:
            self.reader_th = threading.Thread(target=reader_loop,
                                              args=(self.inlet, self.sbuf, self.stop_evt),
                                              daemon=True)
            self.reader_th.start()

        # plotting
        plt.ion()
        self.fig, (self.ax_calm, self.ax_focus) = plt.subplots(2, 1, figsize=(10,6), sharex=True)
        self.line_calm, = self.ax_calm.plot([], [], color="tab:green", lw=2, label="Calm")
        self.line_focus, = self.ax_focus.plot([], [], color="tab:purple", lw=2, label="Focus")

        self.ax_calm.set_ylabel("Calm (0–1)")
        self.ax_focus.set_ylabel("Focus (0–1)")
        self.ax_focus.set_xlabel("Time (s)")
        for ax in (self.ax_calm, self.ax_focus):
            ax.set_ylim(0,1)
            ax.grid(True)
            ax.legend(loc="upper right")

        self.window_s = window_s
        self.dt_update = 1.0/10.0  # 10 Hz UI update
        self.calm_series = []
        self.focus_series = []

    def compute_metrics(self, sig):
        a = bandpower(sig, self.fs, 8, 12)
        th = bandpower(sig, self.fs, 4, 7)
        b = bandpower(sig, self.fs, 15, 25)
        hf = bandpower(sig, self.fs, 20, 35)
        calm = a / (a + hf + 1e-9)
        focus = b / (a + th + 1e-9)
        return calm, focus

    def update_once(self):
        if self.demo:
            self.sbuf.append_chunk(synthetic_chunk(fs=self.fs, ch=2, dur=0.5))

        t, data = self.sbuf.snapshot()
        if data.shape[1] < self.fs:  # need at least 1 second of data
            return

        sig = data[1,:]  # channel 1 is valid (F4-F3)
        calm, focus = self.compute_metrics(sig)
        calm_s = self.ema_calm.update(calm)
        focus_s = self.ema_focus.update(focus)

        self.calm_series.append(calm_s)
        self.focus_series.append(focus_s)
        if len(self.calm_series) > len(t):
            self.calm_series = self.calm_series[-len(t):]
            self.focus_series = self.focus_series[-len(t):]

        xs = np.linspace(-self.window_s, 0, len(self.calm_series))
        self.line_calm.set_data(xs, self.calm_series)
        self.line_focus.set_data(xs, self.focus_series)

        for ax in (self.ax_calm, self.ax_focus):
            ax.set_xlim(xs[0], xs[-1])

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def run(self):
        try:
            while not self.stop_evt.is_set():
                self.update_once()
                time.sleep(self.dt_update)
        finally:
            self.stop_evt.set()
            if self.inlet is not None:
                self.reader_th.join(timeout=0.5)


# ----------------------------
# Entrypoint
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream-name", default="OpenSignals")
    ap.add_argument("--expect-ch", type=int, default=2)
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    dash = CalmFocusDashboard(args.stream_name, args.expect_ch, args.demo, fs=200)
    dash.run()

if __name__ == "__main__":
    main()
