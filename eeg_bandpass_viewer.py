#!/usr/bin/env python3
"""
EEG Calm & Focus Dashboard

- Assumes fs = 500 Hz (fixed)
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
from scipy.signal import welch, butter, filtfilt
from scipy.signal import hilbert


# ----------------------------
# Helpers
# ----------------------------
def bandpower(sig, fs, fmin, fmax):
    """Welch power in band [fmin,fmax]"""
    # Always compute on available data, even if very short
    if len(sig) < 2:
        return 0.0
    f, Pxx = welch(sig, fs=fs, nperseg=min(512, len(sig)))
    if len(f) < 2:
        return 0.0
    df = f[1] - f[0]
    mask = (f >= fmin) & (f <= fmax)
    return float(np.sum(Pxx[mask]) * df) if np.any(mask) else 0.0


def bandpass_filter(sig, fs, fmin, fmax, order=4):
    """Zero-phase Butterworth bandpass filter (returns filtered signal)."""
    ny = 0.5 * fs
    low = max(1e-6, fmin / ny)
    high = min(0.9999, fmax / ny)
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, sig)


def band_envelope(sig, fs, fmin, fmax):
    """Return mean envelope (via Hilbert) of bandpassed signal."""
    try:
        bp = bandpass_filter(sig, fs, fmin, fmax)
        env = np.abs(hilbert(bp))
        return float(np.mean(env))
    except Exception:
        return 0.0


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
    def __init__(self, fs=500, ch=2, buffer=60):
        self.fs = fs
        self.ch = ch
        self.n = int(buffer * fs)
        self.buf = np.zeros((self.ch, self.n), dtype=np.float32)
        self.t = np.linspace(-buffer, 0.0, self.n, dtype=np.float32)
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
def synthetic_chunk(fs=500, ch=2, dur=0.5):
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
                 buffer=60, x_length=10, fs=500,
                 # tuning parameters (change in main())
                 alpha_band=(8, 12), beta_band=(15, 25), theta_band=(4, 7), hf_band=(20,35),
                 smoothing_alpha=0.2,
                 # asymmetry mapping for single-channel: gain controls sigmoid steepness
                 asymmetry_gain=5.0,
                 # asymmetry_sign flips the proxy direction if your device polarity differs
                 asymmetry_sign=1.0):
        # Set fs dynamically from LSL if not demo, else use provided/default fs
        self.demo = demo
        if not demo:
            info = None
            for s in resolve_streams():
                if s.name() == stream_name and s.channel_count() == expect_ch:
                    info = s; break
            if not info:
                raise RuntimeError(f"No stream '{stream_name}' with {expect_ch} channels found.")
            ch = info.channel_count()
            self.fs = int(info.nominal_srate())
            print(f"[LSL] Connected to {info.name()} (Ch={ch}, Fs={self.fs})")
            self.inlet = StreamInlet(info, max_buflen=60, max_chunklen=0, recover=True)
        else:
            ch = expect_ch
            self.fs = fs
            self.inlet = None
            print("[DEMO] Synthetic signal mode.")

        # exposeable tuning params
        self.alpha_band = alpha_band
        self.beta_band = beta_band
        self.theta_band = theta_band
        self.hf_band = hf_band
        self.smoothing_alpha = smoothing_alpha
        self.asymmetry_gain = asymmetry_gain
        self.asymmetry_sign = asymmetry_sign

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

        self.sbuf = StreamBuffer(fs=self.fs, ch=ch, buffer=buffer)

        # EMAs for each metric
        self.ema_conc = Ema(alpha=self.smoothing_alpha)
        self.ema_asym = Ema(alpha=self.smoothing_alpha)
        self.ema_theta = Ema(alpha=self.smoothing_alpha)

        self.stop_evt = threading.Event()
        if self.inlet is not None:
            self.reader_th = threading.Thread(target=reader_loop,
                                              args=(self.inlet, self.sbuf, self.stop_evt),
                                              daemon=True)
            self.reader_th.start()

        # plotting: three stacked axes
        plt.ion()
        self.fig, (self.ax_conc, self.ax_asym, self.ax_theta) = plt.subplots(3, 1, figsize=(10,8), sharex=True)
        self.line_conc, = self.ax_conc.plot([], [], color="tab:green", lw=2, label="Concentration")
        self.line_asym, = self.ax_asym.plot([], [], color="tab:orange", lw=2, label="Alpha Asymmetry")
        self.line_theta, = self.ax_theta.plot([], [], color="tab:purple", lw=2, label="Theta (load)")

        self.ax_conc.set_ylabel("Concentration (0–1)")
        self.ax_asym.set_ylabel("Alpha Asymmetry (0–1)")
        self.ax_theta.set_ylabel("Theta (0–1)")
        self.ax_theta.set_xlabel("Time (s)")
        for ax in (self.ax_conc, self.ax_asym, self.ax_theta):
            ax.set_ylim(0,1)
            ax.grid(True)
            ax.legend(loc="upper right")

        self.buffer = buffer
        self.x_length = x_length
        self.dt_update = 1.0/10.0  # 10 Hz UI update (increase for snappier UI)

        self.conc_series = []
        self.asym_series = []
        self.theta_series = []

        # handle Escape key to stop
        def _on_key(event):
            if event.key == 'escape':
                print("[UI] Escape pressed — stopping")
                self.stop_evt.set()
                try:
                    plt.close(self.fig)
                except Exception:
                    pass
        self.fig.canvas.mpl_connect('key_press_event', _on_key)

    def compute_metrics(self, data):
        """Compute three metrics: concentration, alpha asymmetry, theta activity.

        `data` shape: (ch, n). If ch >=2 we treat channels as left/right (F3/F4).
        The formulas below are simple bandpower ratios intended for rapid tuning.
        """
        # Use two-channel metrics when possible
        if data.shape[0] >= 2:
            left = data[0, :]
            right = data[1, :]
            alpha_l = bandpower(left, self.fs, *self.alpha_band)
            alpha_r = bandpower(right, self.fs, *self.alpha_band)
            beta_l = bandpower(left, self.fs, *self.beta_band)
            beta_r = bandpower(right, self.fs, *self.beta_band)
            theta_l = bandpower(left, self.fs, *self.theta_band)
            theta_r = bandpower(right, self.fs, *self.theta_band)
            hf_l = bandpower(left, self.fs, *self.hf_band)
            hf_r = bandpower(right, self.fs, *self.hf_band)

            # 1) Concentration: normalized beta vs (alpha + hf)
            b = 0.5 * (beta_l + beta_r)
            a = 0.5 * (alpha_l + alpha_r)
            hf = 0.5 * (hf_l + hf_r)
            concentration = b / (a + b + hf + 1e-9)

            # 2) Alpha asymmetry: right-alpha minus left-alpha normalized -> map to 0..1
            asym_raw = (alpha_r - alpha_l) / (alpha_l + alpha_r + 1e-9)
            asymmetry = 1.0 / (1.0 + np.exp(-5.0 * asym_raw))

            # 3) Theta activity: normalized theta vs (theta + alpha)
            th = 0.5 * (theta_l + theta_r)
            theta_metric = th / (th + a + 1e-9)
        else:
            # single-channel fallback (differential / monopolar)
            sig = data[0, :]
            a = bandpower(sig, self.fs, *self.alpha_band)
            b = bandpower(sig, self.fs, *self.beta_band)
            th = bandpower(sig, self.fs, *self.theta_band)
            hf = bandpower(sig, self.fs, *self.hf_band)
            concentration = b / (a + b + hf + 1e-9)
            # single-channel asymmetry: mean of alpha-bandpassed differential (F4-F3)
            alpha_diff = bandpass_filter(sig, self.fs, *self.alpha_band)
            # mean value reflects direction: >0 = F4>F3, <0 = F3>F4
            asym_raw = np.mean(alpha_diff)
            # map to 0..1 for visualization (sigmoid, tunable gain)
            asymmetry = 1.0 / (1.0 + np.exp(-self.asymmetry_gain * asym_raw))
            theta_metric = th / (th + a + 1e-9)

        return float(concentration), float(asymmetry), float(theta_metric)

    def update_once(self):
        if self.demo:
            self.sbuf.append_chunk(synthetic_chunk(fs=self.fs, ch=2, dur=0.5))

        t, data = self.sbuf.snapshot()
        # require at least half-second of data to compute bandpowers reliably
        # Allow metrics computation with as little as 2 samples
        if data.shape[1] < 2:
            return

        conc, asym, theta = self.compute_metrics(data)
        conc_s = self.ema_conc.update(conc)
        asym_s = self.ema_asym.update(asym)
        theta_s = self.ema_theta.update(theta)

        self.conc_series.append(conc_s)
        self.asym_series.append(asym_s)
        self.theta_series.append(theta_s)

        # trim series to buffer length
        if len(self.conc_series) > len(t):
            self.conc_series = self.conc_series[-len(t):]
            self.asym_series = self.asym_series[-len(t):]
            self.theta_series = self.theta_series[-len(t):]

        # Only plot the available portion of the buffer until full
        n = len(self.conc_series)
        # Only plot the last x_length seconds, right-aligned to x=0, with fixed x-axis [-x_length, 0]
        n_disp = min(n, int(self.x_length * self.fs))
        if n_disp > 1:
            xs = np.linspace(-n_disp / self.fs, 0, n_disp)
            self.line_conc.set_data(xs, self.conc_series[-n_disp:])
            self.line_asym.set_data(xs, self.asym_series[-n_disp:])
            self.line_theta.set_data(xs, self.theta_series[-n_disp:])
        elif n_disp == 1:
            xs = np.array([0.0])
            self.line_conc.set_data(xs, self.conc_series[-1:])
            self.line_asym.set_data(xs, self.asym_series[-1:])
            self.line_theta.set_data(xs, self.theta_series[-1:])
        else:
            xs = np.array([])
            self.line_conc.set_data(xs, [])
            self.line_asym.set_data(xs, [])
            self.line_theta.set_data(xs, [])

        # Fixed xlim: always [-x_length, 0]
        for ax in (self.ax_conc, self.ax_asym, self.ax_theta):
            ax.set_xlim(-self.x_length, 0)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def run(self, duration_s=20):
        start = time.time()
        try:
            while not self.stop_evt.is_set():
                self.update_once()
                time.sleep(self.dt_update)
                if (time.time() - start) >= duration_s:
                    print(f"[INFO] {duration_s} seconds elapsed. Stopping...")
                    self.stop_evt.set()
        finally:
            self.stop_evt.set()
            if getattr(self, 'inlet', None) is not None:
                self.reader_th.join(timeout=0.5)
            self.print_report()

    def print_report(self):
        def snr(arr):
            arr = np.array(arr)
            return np.mean(arr) / (np.std(arr) + 1e-9)
        def minmax(arr):
            arr = np.array(arr)
            return np.min(arr), np.max(arr)
        print("\n==== 20s Signal Quality Report ====")
        for name, series in zip(["Concentration", "Theta"],
                                [self.conc_series, self.asym_series, self.theta_series]):
            if len(series) < 2:
                print(f"{name}: Not enough data.")
                continue
            s = snr(series)
            mn, mx = minmax(series)
            print(f"{name:14s} | SNR: {s:7.3f} | Range: [{mn:7.3f}, {mx:7.3f}] | Delta: {mx - mn:7.3f} |")
        print("===================================\n")


# ----------------------------
# Entrypoint
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream-name", default="OpenSignals")
    ap.add_argument("--expect-ch", type=int, default=2)
    ap.add_argument("--demo", action="store_true", default=False, help="Run in synthetic demo mode (default: off)")
    args = ap.parse_args()
    dash = CalmFocusDashboard(args.stream_name, args.expect_ch, args.demo, fs=500, buffer=2, x_length=0.3,
                              alpha_band=(8, 12), 
                              beta_band=(15, 25), 
                              theta_band=(4, 7), 
                              hf_band=(20,35),
                              smoothing_alpha=0.2,
                              # asymmetry mapping for single-channel: gain controls sigmoid steepness
                              asymmetry_gain=5.0,
                              # asymmetry_sign flips the proxy direction if your device polarity differs
                              asymmetry_sign=1.0)
    dash.run()

if __name__ == "__main__":
    main()
