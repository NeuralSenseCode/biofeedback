#!/usr/bin/env python3
"""
Neuro Debug Scope — Realtime CALM / FOCUS / BLINK viewer (PLUX via OpenSignals LSL)

Plots last 60 seconds:
  1) CALM  = alpha(8–12) / (alpha + HF(20–35))
  2) FOCUS = beta(15–25) / (alpha + theta(4–7))
  3) BLINK square (0/1) using robust derivative threshold (MAD-based), 300 ms pulse

Hotkeys:
  Q / ESC  : quit
  S        : save last 60s to CSV (timestamp, calm, focus, blink, quality)
"""
import argparse
import time
import threading
from collections import deque
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

# Optional deps so the script still starts if missing (you'll want them installed)
try:
    from pylsl import resolve_byprop, resolve_streams, StreamInlet
    HAVE_LSL = True
except Exception as e:
    print("[INFO] pylsl import failed; install with: pip install pylsl")
    HAVE_LSL = False

try:
    from scipy.signal import butter, lfilter, iirnotch, welch
    HAVE_SCIPY = True
except Exception as e:
    print("[INFO] SciPy import failed; install with: pip install scipy")
    HAVE_SCIPY = False


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def ema(prev, x, alpha=0.2):
    return x if prev is None else alpha * x + (1 - alpha) * prev


def bandpower_welch(sig, fs, fmin, fmax):
    if len(sig) < max(128, int(0.5 * fs)):
        return 0.0
    f, Pxx = welch(sig, fs=fs, nperseg=min(512, len(sig)))
    if len(f) < 2:
        return 0.0
    df = f[1] - f[0]
    mask = (f >= fmin) & (f <= fmax)
    return float(np.sum(Pxx[mask]) * df)


class FilterBank:
    """1–40 Hz band-pass + 50 Hz notch (SA mains)"""
    def __init__(self):
        self.fs = None
        self.bp_b = self.bp_a = None
        self.notch_b = self.notch_a = None

    def design(self, fs):
        if not HAVE_SCIPY:
            return
        if self.fs == fs and self.bp_b is not None:
            return
        self.fs = fs
        nyq = fs / 2.0
        self.bp_b, self.bp_a = butter(4, [1.0 / nyq, 40.0 / nyq], btype="band")
        w0 = 50.0 / nyq
        try:
            self.notch_b, self.notch_a = iirnotch(w0, Q=30.0)
        except Exception:
            self.notch_b = self.notch_a = None

    def apply(self, x):
        if not HAVE_SCIPY or self.bp_b is None:
            return x
        y = lfilter(self.bp_b, self.bp_a, x)
        if self.notch_b is not None:
            y = lfilter(self.notch_b, self.notch_a, y)
        return y


class EEGProcessor:
    """
    Maintains rolling buffers, computes CALM/FOCUS/quality every ~100 ms,
    and detects blinks on the best channel.
    """
    def __init__(self, window_secs=10.0):
        self.fs = None
        self.filter = FilterBank()
        self.buffers = []  # deque per channel
        self.lock = threading.Lock()
        self.max_secs = window_secs
        self.max_samples = 256 * window_secs  # provisional until fs known

        # Metrics
        self.calm = 0.0
        self.focus = 0.0
        self.quality = 0.0

        # Smoothing + history for normalization (if needed)
        self.calm_ema = None
        self.focus_ema = None

        # Blink
        self.last_blink = 0.0
        self.blinked = False
        self.blink_hold_s = 0.30  # pulse width for square signal

    def set_fs(self, fs):
        if self.fs != fs:
            self.fs = fs
            self.filter.design(fs)
            self.max_samples = int(self.fs * self.max_secs)
            with self.lock:
                if self.buffers:
                    self.buffers = [deque(list(b)[-self.max_samples:], maxlen=self.max_samples) for b in self.buffers]

    def set_channels(self, n):
        with self.lock:
            self.buffers = [deque(maxlen=self.max_samples) for _ in range(n)]

    def push(self, samples, fs):
        """samples: (n_samples, n_channels)"""
        if self.fs is None:
            self.set_fs(fs)
            self.set_channels(samples.shape[1])
        with self.lock:
            for c in range(samples.shape[1]):
                self.buffers[c].extend(samples[:, c])

    def compute(self):
        """Return (calm_smooth, focus_smooth, blink_bool, quality) or None if not ready."""
        if self.fs is None or not HAVE_SCIPY:
            return None

        # Copy buffers
        with self.lock:
            chans = [np.asarray(list(buf), dtype=np.float32) if len(buf) >= int(self.fs * 1.0) else None
                     for buf in self.buffers]

        if not any(ch is not None for ch in chans):
            return None

        calm_vals, focus_vals, q_vals = [], [], []
        best_q = -1.0
        y_best = None

        for ch in chans:
            if ch is None:
                calm_vals.append(0.0); focus_vals.append(0.0); q_vals.append(0.0)
                continue

            y = self.filter.apply(ch - np.mean(ch))

            # Band powers
            a  = bandpower_welch(y, self.fs, 8.0, 12.0)    # alpha
            th = bandpower_welch(y, self.fs, 4.0, 7.0)     # theta
            b  = bandpower_welch(y, self.fs, 15.0, 25.0)   # beta
            hf = bandpower_welch(y, self.fs, 20.0, 35.0)   # high freq (EMG-ish)
            ln = bandpower_welch(y, self.fs, 49.0, 51.0)   # 50 Hz
            tot= bandpower_welch(y, self.fs, 1.0, 40.0) + 1e-9

            # Quality = low line noise + not dominated by HF + (clip penalty)
            q_line = clamp(1.0 - (ln / tot), 0.0, 1.0)
            q_hf   = clamp(1.0 - (hf / tot), 0.0, 1.0)
            # Clip penalty (samples near rail)
            y_abs = np.abs(y)
            p99 = np.percentile(y_abs, 99.9)
            clip_frac = float(np.mean(y_abs >= p99))
            q_clip = clamp(1.0 - 5.0 * clip_frac, 0.0, 1.0)

            q = 0.4*q_line + 0.4*q_hf + 0.2*q_clip

            # Frontal-friendly indices
            calm  = a / (a + hf + 1e-9)
            focus = b / (a + th + 1e-9)

            calm_vals.append(calm)
            focus_vals.append(focus)
            q_vals.append(q)

            if q > best_q:
                best_q = q
                y_best = y

        qarr = np.array(q_vals); calm_arr = np.array(calm_vals); focus_arr = np.array(focus_vals)

        if np.sum(qarr > 0.4) >= 2:
            calm  = float(np.mean(calm_arr[qarr > 0.4]))
            focus = float(np.mean(focus_arr[qarr > 0.4]))
            quality = float(np.mean(qarr[qarr > 0.4]))
        else:
            idx = int(np.argmax(qarr))
            calm  = float(calm_arr[idx])
            focus = float(focus_arr[idx])
            quality = float(qarr[idx])

        # Blink detection on best channel
        blink = False
        if y_best is not None:
            N = max(int(self.fs * 0.5), 50)
            yw = y_best[-N:]
            dy = np.abs(np.diff(yw))
            if dy.size:
                med = float(np.median(dy))
                mad = float(np.median(np.abs(dy - med))) + 1e-9
                thr = med + 4.0 * mad
                now = time.time()
                if dy.max() > thr and now - self.last_blink > 0.6:
                    blink = True
                    self.last_blink = now

        # Smooth
        self.calm_ema  = ema(self.calm_ema, calm, alpha=0.2)
        self.focus_ema = ema(self.focus_ema, focus, alpha=0.2)

        self.calm    = float(self.calm_ema if self.calm_ema is not None else calm)
        self.focus   = float(self.focus_ema if self.focus_ema is not None else focus)
        self.quality = float(quality)

        return self.calm, self.focus, blink, self.quality


class LSLReader(threading.Thread):
    def __init__(self, processor, stream_name=None, stream_type='EEG'):
        super().__init__(daemon=True)
        self.proc = processor
        self.stream_name = stream_name
        self.stream_type = stream_type
        self.stop_flag = threading.Event()
        self.connected = False
        self.fs = None
        self.inlet = None

    def connect(self):
        if not HAVE_LSL:
            return False
        try:
            streams = []
            if self.stream_name:
                streams = resolve_byprop('name', self.stream_name, 2.0)
            if not streams:
                streams = resolve_byprop('type', self.stream_type or 'EEG', 2.0)
            if not streams:
                try:
                    streams = resolve_streams()  # some pylsl builds don't take a timeout arg
                except TypeError:
                    streams = resolve_streams()

            if not streams:
                print("[LSL] No streams found.")
                return False

            # Pick the most EEG-like
            def score(info):
                s = 0
                if (info.type() or '').upper() == 'EEG': s += 3
                if info.nominal_srate() > 0: s += 2
                if info.channel_count() >= 1: s += 1
                if 'opensignals' in (info.name() or '').lower(): s += 2
                return s
            info = sorted(streams, key=score, reverse=True)[0]

            self.inlet = StreamInlet(info, max_buflen=60)
            ch = info.channel_count()
            fs = int(info.nominal_srate()) or 256
            if fs <= 0 or ch <= 0:
                print(f"[LSL] Rejected stream: Fs={fs}, Ch={ch}")
                return False

            self.fs = fs
            self.proc.set_fs(fs)
            self.proc.set_channels(ch)
            self.connected = True
            print(f"[LSL] Connected: Name='{info.name()}', Type='{info.type()}', Ch={ch}, Fs={fs}")
            return True
        except Exception as e:
            print("[LSL] Connect failed:", e)
            self.connected = False
            return False

    def run(self):
        while not self.stop_flag.is_set():
            if not self.connected:
                if not self.connect():
                    time.sleep(0.5); continue
            try:
                chunk, ts = self.inlet.pull_chunk(timeout=0.05, max_samples=128)
                if ts and len(chunk) > 0:
                    arr = np.asarray(chunk, dtype=np.float32)
                    # Keep first two channels (F3/F4)
                    if arr.shape[1] > 2:
                        arr = arr[:, :2]
                    push_fs = self.fs
                    # Decimate if very high rate
                    if self.fs >= 800:
                        arr = arr[::4]
                        push_fs = int(self.fs / 4)
                    self.proc.push(arr, push_fs)
            except Exception as e:
                print("[LSL] Read error; reconnecting:", e)
                self.connected = False
                time.sleep(0.5)

    def stop(self):
        self.stop_flag.set()


# ---------------------- Plotter / App ----------------------

class DebugScope:
    def __init__(self, stream_name=None, stream_type='EEG', window_s=60.0, update_hz=10.0):
        if not HAVE_LSL or not HAVE_SCIPY:
            raise RuntimeError("This tool needs pylsl and scipy installed.")

        self.proc = EEGProcessor(window_secs=10.0)
        self.reader = LSLReader(self.proc, stream_name=stream_name, stream_type=stream_type)
        self.reader.start()

        self.window_s = float(window_s)
        self.dt_update = 1.0 / float(update_hz)

        # ring buffers for 60s display at update rate
        self.times = deque(maxlen=int(self.window_s / self.dt_update) + 5)
        self.calm_series = deque(maxlen=int(self.window_s / self.dt_update) + 5)
        self.focus_series = deque(maxlen=int(self.window_s / self.dt_update) + 5)
        self.blink_series = deque(maxlen=int(self.window_s / self.dt_update) + 5)

        self.last_save = None
        self.running = True

        # Matplotlib setup
        plt.ion()
        self.fig = plt.figure(figsize=(10, 7))
        self.ax1 = self.fig.add_subplot(3,1,1)
        self.ax2 = self.fig.add_subplot(3,1,2)
        self.ax3 = self.fig.add_subplot(3,1,3)

        self.line_calm,  = self.ax1.plot([], [], lw=2)
        self.line_focus, = self.ax2.plot([], [], lw=2)
        self.line_blink, = self.ax3.plot([], [], lw=2, drawstyle='steps-post')

        self.ax1.set_ylim(0,1); self.ax2.set_ylim(0,1); self.ax3.set_ylim(-0.1, 1.1)
        self.ax1.set_title("CALM (alpha / (alpha + HF)) — smoothed")
        self.ax2.set_title("FOCUS (beta / (alpha + theta)) — smoothed")
        self.ax3.set_title("BLINK (0/1)")

        for ax in (self.ax1, self.ax2, self.ax3):
            ax.grid(True)
            ax.set_xlim(-self.window_s, 0)
            ax.set_xlabel("Time (s)")

        self.text_quality = self.ax1.text(0.99, 0.05, "Quality: --%", transform=self.ax1.transAxes,
                                          ha='right', va='bottom', fontsize=10,
                                          bbox=dict(boxstyle="round,pad=0.3", fc="w", ec="0.7", alpha=0.8))

        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

    def on_key(self, ev):
        if ev.key in ('q', 'escape'):
            self.running = False
        elif ev.key.lower() == 's':
            self.save_csv()

    def save_csv(self):
        if not self.times:
            return
        ts0 = time.time()
        outdir = f"DebugSessions/{datetime.now().strftime('%Y-%m-%d')}"
        os.makedirs(outdir := outdir, exist_ok=True)
        path = f"{outdir}/scope_{datetime.now().strftime('%H%M%S')}.csv"
        with open(path, "w", encoding="utf-8") as f:
            f.write("rel_time_s,calm,focus,blink,quality\n")
            # We don't store per-sample quality; write latest for all rows
            q = self.proc.quality
            for t, c, fo, b in zip(self.times, self.calm_series, self.focus_series, self.blink_series):
                f.write(f"{t:.3f},{c:.6f},{fo:.6f},{int(b)},{q:.3f}\n")
        print(f"[Saved] {path}")

    def update_once(self):
        tnow = time.time()
        res = self.proc.compute()
        if res is None:
            return False
        calm, focus, blink, quality = res

        # blink pulse for 0.3 s
        blink_val = 1 if (blink or (tnow - self.proc.last_blink <= self.proc.blink_hold_s)) else 0

        # update series
        rel_t = 0.0  # we'll plot relative time, with newest at 0
        self.times.append(rel_t)
        self.calm_series.append(calm)
        self.focus_series.append(focus)
        self.blink_series.append(blink_val)

        # shift x to keep last self.window_s seconds with newest at 0
        xs = np.linspace(-self.window_s + self.dt_update, 0, num=len(self.times))
        self.line_calm.set_data(xs, list(self.calm_series))
        self.line_focus.set_data(xs, list(self.focus_series))
        self.line_blink.set_data(xs, list(self.blink_series))

        self.text_quality.set_text(f"Quality: {int(100*quality)}%")
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        return True

    def run(self):
        try:
            while self.running:
                t0 = time.time()
                self.update_once()
                dt = time.time() - t0
                time.sleep(max(0.0, self.dt_update - dt))
        finally:
            self.reader.stop()
            self.reader.join(timeout=0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stream-name', type=str, default="OpenSignals", help="Exact LSL stream name (e.g., 'OpenSignals')")
    ap.add_argument('--stream-type', type=str, default='00:07:80:89:80:02', help="LSL stream type (default: EEG)")
    ap.add_argument('--window', type=float, default=60.0, help="Time window to display (seconds)")
    ap.add_argument('--update-hz', type=float, default=10.0, help="UI update rate (Hz)")
    args = ap.parse_args()

    scope = DebugScope(stream_name=args.stream_name, stream_type=args.stream_type,
                       window_s=args.window, update_hz=args.update_hz)
    scope.run()


if __name__ == "__main__":
    import os
    main()
