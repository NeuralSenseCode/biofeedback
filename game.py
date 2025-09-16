#!/usr/bin/env python3
"""
EEG Scroller Demo Game (30s) + Calibration

- Ball X fixed at screen center; user controls Y by brain metric.
- Control metric: Concentration (default) or Theta (toggle with [T] key).
- Map scrolls at constant speed; obstacles spawn every ~5s (one at a time).
- Touching top/bottom is allowed; hitting an obstacle ends the game.
- Win if you reach 30s without a collision. Press [R] to restart.
- Works with LSL stream (e.g., "OpenSignals") or --demo synthetic signal.
- NEW: Calibration button runs a two-phase procedure:
    1) Focus phase: 5 on-screen mental-arithmetic prompts (we record a MAX).
    2) Calm phase: guided relaxation (we record a MIN).
  The selected metric is then linearly mapped so MAX -> top, MIN -> bottom.

Dependencies:
    pip install numpy scipy pylsl

Usage:
    python eeg_scroller_game.py --stream-name OpenSignals --expect-ch 2
    python eeg_scroller_game.py --demo
    python eeg_scroller_game.py --control theta
"""

import argparse
import random
import threading
import time
from dataclasses import dataclass

import numpy as np
from pylsl import StreamInlet, resolve_streams
from scipy.signal import welch, butter, filtfilt, hilbert

try:
    import tkinter as tk
except Exception as e:
    raise SystemExit("Tkinter is required (backend TkAgg). On some systems: sudo apt-get install python3-tk") from e


# ----------------------------
# Helpers (signal processing)
# ----------------------------
def bandpower(sig, fs, fmin, fmax):
    """Welch power in band [fmin,fmax]"""
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
# Streaming buffer (LSL)
# ----------------------------
class StreamBuffer:
    def __init__(self, fs=500, ch=2, buffer=10):
        self.fs = fs
        self.ch = ch
        self.n = int(buffer * fs)
        self.buf = np.zeros((self.ch, self.n), dtype=np.float32)
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
                return np.roll(self.buf, 0, axis=1)
            return np.roll(self.buf, -self.idx, axis=1)


def reader_loop(inlet, sbuf, stop_evt):
    while not stop_evt.is_set():
        try:
            chunk, ts = inlet.pull_chunk(timeout=0.05)
            if chunk:
                arr = np.array(chunk, dtype=np.float32)
                if arr.ndim == 1:
                    arr = arr[None, :]
                sbuf.append_chunk(arr)
            else:
                time.sleep(0.01)
        except Exception:
            time.sleep(0.02)


def synthetic_chunk(fs=500, ch=2, dur=0.2):
    """Alpha-dominant with occasional beta bursts + noise."""
    t = np.arange(int(fs * dur)) / fs
    x = 20 * np.sin(2 * np.pi * 10 * t)
    if np.random.rand() < 0.3:
        x += 15 * np.sin(2 * np.pi * 20 * t)
    x += 5 * np.random.randn(len(t))
    sig = np.stack([np.zeros_like(x), x], axis=0)  # 2ch
    return sig.astype(np.float32)


# ----------------------------
# EEG Metric Source
# ----------------------------
@dataclass
class EEGConfig:
    stream_name: str = "OpenSignals"
    expect_ch: int = 2
    fs: int = 500
    alpha_band: tuple = (8, 12)
    beta_band: tuple = (15, 25)
    theta_band: tuple = (4, 7)
    hf_band: tuple = (20, 35)
    smoothing_alpha: float = 0.5  # EMA smoothing for stable control
    buffer: float = 2.0          # EEG buffer length in seconds
    demo: bool = False


class EEGSource:
    """Computes concentration and theta continuously; exposes latest smoothed values in [0,1]."""

    def __init__(self, cfg: EEGConfig):
        self.cfg = cfg
        self.fs = cfg.fs
        self.stop_evt = threading.Event()

        if not cfg.demo:
            info = None
            for s in resolve_streams():
                if s.name() == cfg.stream_name and s.channel_count() == cfg.expect_ch:
                    info = s
                    break
            if not info:
                raise RuntimeError(f"No stream '{cfg.stream_name}' with {cfg.expect_ch} channels found.")
            self.fs = int(info.nominal_srate()) or self.fs
            self.inlet = StreamInlet(info, max_buflen=60, max_chunklen=0, recover=True)
            ch = info.channel_count()
            print(f"[LSL] Connected to {info.name()} (Ch={ch}, Fs={self.fs})")
        else:
            self.inlet = None
            print("[DEMO] Using synthetic EEG.")

        # Use configured buffer seconds
        self.sbuf = StreamBuffer(fs=self.fs, ch=cfg.expect_ch, buffer=max(1.0, float(cfg.buffer)))
        if self.inlet is not None:
            self.reader_th = threading.Thread(target=reader_loop,
                                              args=(self.inlet, self.sbuf, self.stop_evt),
                                              daemon=True)
            self.reader_th.start()

        self.ema_conc = Ema(alpha=cfg.smoothing_alpha)
        self.ema_theta = Ema(alpha=cfg.smoothing_alpha)
        self.last_update = 0.0
        self.update_hz = 10.0  # compute metrics at 10 Hz
        self.conc = 0.5
        self.theta_m = 0.5

    def close(self):
        self.stop_evt.set()
        if getattr(self, "inlet", None) is not None:
            try:
                self.reader_th.join(timeout=0.3)
            except Exception:
                pass

    def _compute_metrics(self, data):
        """Return (concentration, theta_metric) as floats in [0,1]."""
        fs = self.fs
        ab = self.cfg.alpha_band
        bb = self.cfg.beta_band
        tb = self.cfg.theta_band
        hb = self.cfg.hf_band

        if data.shape[0] >= 2:
            left = data[0, :]
            right = data[1, :]
            alpha_l = bandpower(left, fs, *ab)
            alpha_r = bandpower(right, fs, *ab)
            beta_l = bandpower(left, fs, *bb)
            beta_r = bandpower(right, fs, *bb)
            theta_l = bandpower(left, fs, *tb)
            theta_r = bandpower(right, fs, *tb)
            hf_l = bandpower(left, fs, *hb)
            hf_r = bandpower(right, fs, *hb)
            b = 0.5 * (beta_l + beta_r)
            a = 0.5 * (alpha_l + alpha_r)
            hf = 0.5 * (hf_l + hf_r)
            conc = b / (a + b + hf + 1e-9)            # 0..1-ish
            th = 0.5 * (theta_l + theta_r)
            theta_metric = th / (th + a + 1e-9)       # 0..1-ish
        else:
            sig = data[0, :]
            a = bandpower(sig, fs, *ab)
            b = bandpower(sig, fs, *bb)
            th = bandpower(sig, fs, *tb)
            hf = bandpower(sig, fs, *hb)
            conc = b / (a + b + hf + 1e-9)
            theta_metric = th / (th + a + 1e-9)

        # clamp
        conc = max(0.0, min(1.0, conc))
        theta_metric = max(0.0, min(1.0, theta_metric))
        return conc, theta_metric

    def get_metrics(self):
        """Update (<=10Hz) and return smoothed (concentration, theta)."""
        now = time.time()
        if (now - self.last_update) >= (1.0 / self.update_hz):
            if self.cfg.demo:
                # Feed synthetic data
                self.sbuf.append_chunk(synthetic_chunk(fs=self.fs, ch=2, dur=0.2).T)
            data = self.sbuf.snapshot()
            if data.shape[1] >= int(0.5 * self.fs):  # need at least 0.5s of data
                conc, theta_m = self._compute_metrics(data)
                self.conc = float(self.ema_conc.update(conc))
                self.theta_m = float(self.ema_theta.update(theta_m))
            self.last_update = now
        return self.conc, self.theta_m


# ----------------------------
# Game + Calibration
# ----------------------------
class ScrollerGame:
    def __init__(self, eeg: EEGSource, control="concentration",
                 width=900, height=600, duration=30.0, spawn_interval=5.0):
        self.eeg = eeg
        self.control = control  # 'concentration' or 'theta'
        self.W = width
        self.H = height
        self.duration = float(duration)
        self.spawn_interval = float(spawn_interval)

        # Map area = 75% vertical space (centered)
        self.map_h = int(self.H * 0.75)
        self.map_top = (self.H - self.map_h) // 2
        self.map_bot = self.map_top + self.map_h

        # Ball
        self.ball_d = 40  # diameter
        self.ball_r = self.ball_d // 2
        self.ball_x = self.W // 2
        self.ball_y = (self.map_top + self.map_bot) // 2

        # Obstacles
        self.obstacle = None  # (id, x, y, w, h) tracked via canvas tags
        self.ob_w = 60
        self.ob_h = self.ball_d   # same height as ball
        self.scroll_px_per_s = 220  # constant map speed (pixels/sec)

        # State
        self.start_time = None
        self.last_spawn_t = None
        self.game_over = False
        self.win = False
        our_in_calibration = False
        self.in_calibration = False

        # Per-metric calibration: dict metric -> (min, max)
        self.calib = {"concentration": None, "theta": None}

        # Tk setup
        self.root = tk.Tk()
        self.root.title("EEG Scroller Demo — Control: Concentration")
        self.canvas = tk.Canvas(self.root, width=self.W, height=self.H, bg="#111")
        self.canvas.pack()

        # Static elements: map bounds lines
        self.canvas.create_line(0, self.map_top, self.W, self.map_top, fill="black", width=3)
        self.canvas.create_line(0, self.map_bot, self.W, self.map_bot, fill="black", width=3)

        # Ball
        self.ball_item = self.canvas.create_oval(self.ball_x - self.ball_r, self.ball_y - self.ball_r,
                                                 self.ball_x + self.ball_r, self.ball_y + self.ball_r,
                                                 fill="#4ade80", outline="")

        # HUD
        self.hud_text = self.canvas.create_text(self.W // 2, 20, fill="#ddd", font=("Helvetica", 14),
                                                text="")
        self.msg_text = self.canvas.create_text(self.W // 2, self.H // 2, fill="#fff",
                                                font=("Helvetica", 28, "bold"), text="")

        # Buttons
        self.btn_frame = tk.Frame(self.root, bg="#111")
        self.btn_frame.pack(pady=6)
        self.restart_btn = tk.Button(self.btn_frame, text="Restart (R)", command=self.restart)
        self.toggle_btn = tk.Button(self.btn_frame, text="Toggle Metric (T)", command=self.toggle_metric)
        self.calib_btn = tk.Button(self.btn_frame, text="Calibrate", command=self.start_calibration)
        self.restart_btn.pack(side="left", padx=8)
        self.toggle_btn.pack(side="left", padx=8)
        self.calib_btn.pack(side="left", padx=8)

        # Key bindings
        self.root.bind("<r>", lambda e: self.restart())
        self.root.bind("<R>", lambda e: self.restart())
        self.root.bind("<t>", lambda e: self.toggle_metric())
        self.root.bind("<T>", lambda e: self.toggle_metric())
        self.root.bind("<c>", lambda e: self.start_calibration())
        self.root.bind("<C>", lambda e: self.start_calibration())

    # ------------ Gameplay ------------
    def start(self):
        self.restart(first=True)
        self.root.mainloop()

    def restart(self, first=False):
        # Clear obstacle if any
        if self.obstacle is not None:
            self.canvas.delete(self.obstacle[0])
            self.obstacle = None

        self.game_over = False
        self.win = False
        self.in_calibration = False
        self.start_time = time.time()
        self.last_spawn_t = 0.0
        self.canvas.itemconfigure(self.msg_text, text="")
        self.update_title()
        if not first:
            # small countdown effect
            self.canvas.itemconfigure(self.msg_text, text="Restarting...")
            self.root.after(600, lambda: self.canvas.itemconfigure(self.msg_text, text=""))
        self.schedule_frame()

    def toggle_metric(self):
        self.control = "theta" if self.control == "concentration" else "concentration"
        self.update_title()

    def update_title(self):
        name = "Concentration" if self.control == "concentration" else "Theta"
        self.root.title(f"EEG Scroller Demo — Control: {name}")

    def schedule_frame(self):
        self.root.after(16, self.frame)  # ~60 FPS

    def frame(self):
        # If calibration is running, skip game mechanics (calibration loop handles UI)
        if self.in_calibration:
            self.schedule_frame()
            return

        if self.game_over:
            return

        now = time.time()
        t = now - self.start_time
        remaining = max(0.0, self.duration - t)

        # 1) Update metrics (<=10Hz) & move ball
        conc, theta_m = self.eeg.get_metrics()
        val = conc if self.control == "concentration" else theta_m
        metric = max(0.0, min(1.0, val))  # clamp

        # Normalize with calibration if available
        metric_norm = self.apply_calibration(metric)

        # Map normalized metric (0..1) -> y position (top..bottom)
        target_y = int(self.map_top + (1.0 - metric_norm) * self.map_h)
        # Smooth move (simple lerp for nicer motion at 60fps)
        self.ball_y = int(0.6 * self.ball_y + 0.4 * target_y)
        self.set_ball_pos(self.ball_x, self.ball_y)

        # 2) Spawn obstacle every ~spawn_interval (one at a time)
        if (self.obstacle is None) and (t - self.last_spawn_t >= self.spawn_interval) and (t <= self.duration - 1.0):
            oy = random.randint(self.map_top, self.map_bot - self.ob_h)
            oid = self.canvas.create_rectangle(self.W + self.ob_w, oy, self.W + 2 * self.ob_w, oy + self.ob_h,
                                               fill="#38bdf8", outline="")
            self.obstacle = (oid, self.W + self.ob_w, oy, self.ob_w, self.ob_h)
            self.last_spawn_t = t

        # 3) Move obstacle & collision check
        if self.obstacle is not None:
            oid, ox, oy, ow, oh = self.obstacle
            dx = -self.scroll_px_per_s / 60.0  # pixels per frame
            ox_new = ox + dx
            self.canvas.move(oid, dx, 0)

            # Remove when fully off-screen
            if ox_new + ow < 0:
                self.canvas.delete(oid)
                self.obstacle = None
            else:
                self.obstacle = (oid, ox_new, oy, ow, oh)
                if self.intersects_ball(ox_new, oy, ow, oh):
                    self.end_game(win=False)
                    return

        # 4) Win condition
        if remaining <= 0.0:
            self.end_game(win=True)
            return

        # 5) HUD
        mname = "Conc" if self.control == "concentration" else "Theta"
        cal_state = "Cal:On" if self.calib.get(self.control) else "Cal:Off"
        self.canvas.itemconfigure(self.hud_text,
                                  text=f"{mname}: {metric:0.2f}  ({cal_state})   |   Time left: {remaining:0.1f}s")

        self.schedule_frame()

    # ------------ Calibration ------------
    def start_calibration(self):
        """Run two-phase calibration for the currently selected metric."""
        if self.in_calibration:
            return
        # Remove obstacle & freeze gameplay
        if self.obstacle is not None:
            self.canvas.delete(self.obstacle[0])
            self.obstacle = None
        self.in_calibration = True
        self.canvas.itemconfigure(self.msg_text, text="Calibration starting...")
        self.root.after(800, self._calibrate_focus_start)

    # Focus phase configuration
    CAL_SUMS = 5
    CAL_SUM_DUR = 4.0   # seconds per sum
    CAL_CALM_DUR = 12.0 # seconds of calm

    def _gen_sum(self):
        # Simple mental arithmetic: two numbers 7..19 with + or - or x (weighted towards +)
        ops = ["+", "+", "+", "-", "×"]
        a = random.randint(7, 19)
        b = random.randint(7, 19)
        op = random.choice(ops)
        return a, op, b

    def _calibrate_focus_start(self):
        self.cal_phase = "focus"
        self.cal_metric_name = self.control
        self.cal_focus_max = -1e9
        self.cal_min = 1e9
        self.cal_start_t = time.time()
        self.cal_sum_idx = 0
        self.cal_next_switch = self.cal_start_t + self.CAL_SUM_DUR
        self.cal_sums = [self._gen_sum() for _ in range(self.CAL_SUMS)]
        self._update_calibration_text()
        self._calibration_tick()

    def _calibration_tick(self):
        """Runs at ~10-15Hz, updates metric extremes and phase transitions."""
        if not self.in_calibration:
            return

        # Read current metric
        conc, theta_m = self.eeg.get_metrics()
        val = conc if self.cal_metric_name == "concentration" else theta_m
        val = max(0.0, min(1.0, val))

        # Track extremes
        if self.cal_phase == "focus":
            if val > self.cal_focus_max:
                self.cal_focus_max = val
        elif self.cal_phase == "calm":
            if val < self.cal_min:
                self.cal_min = val

        now = time.time()

        # Phase control
        if self.cal_phase == "focus":
            if now >= self.cal_next_switch:
                self.cal_sum_idx += 1
                if self.cal_sum_idx >= self.CAL_SUMS:
                    self._calibrate_calm_start()
                else:
                    self.cal_next_switch = now + self.CAL_SUM_DUR
                    self._update_calibration_text()
        elif self.cal_phase == "calm":
            if now >= self.cal_next_switch:
                self._calibration_finish()
                return

        # Update text countdown
        self._update_calibration_text()

        # Schedule next tick
        self.root.after(80, self._calibration_tick)  # ~12.5 Hz

    def _calibrate_calm_start(self):
        self.cal_phase = "calm"
        self.cal_next_switch = time.time() + self.CAL_CALM_DUR
        self._update_calibration_text()

    def _update_calibration_text(self):
        if not self.in_calibration:
            return
        if self.cal_phase == "focus":
            a, op, b = self.cal_sums[self.cal_sum_idx]
            remaining = max(0.0, self.cal_next_switch - time.time())
            msg = (f"Calibration — FOCUS ({self.cal_sum_idx+1}/{self.CAL_SUMS})\n\n"
                   f"Solve in your head:\n\n   {a} {op} {b} = ?\n\n"
                   f"Time left on this sum: {remaining:0.1f}s")
        else:
            remaining = max(0.0, self.cal_next_switch - time.time())
            msg = ("Calibration — CALM\n\n"
                   "Soften the shoulders, relax the eyes,\n"
                   "and take a long, gentle exhale.\n\n"
                   f"Measuring calm for {remaining:0.1f}s...")
        self.canvas.itemconfigure(self.msg_text, text=msg)

    def _calibration_finish(self):
        # Fallbacks if no data
        if self.cal_focus_max <= -1e8 or self.cal_min >= 1e8:
            # Use current reading
            conc, theta_m = self.eeg.get_metrics()
            cur = conc if self.cal_metric_name == "concentration" else theta_m
            self.cal_focus_max = max(0.6, float(cur))
            self.cal_min = min(0.4, float(cur))

        # Ensure min < max; widen if too narrow
        if self.cal_focus_max <= self.cal_min:
            mid = 0.5 * (self.cal_focus_max + self.cal_min)
            self.cal_min = max(0.0, mid - 0.1)
            self.cal_focus_max = min(1.0, mid + 0.1)

        # Avoid tiny ranges
        if (self.cal_focus_max - self.cal_min) < 0.05:
            span = 0.05
            mid = 0.5 * (self.cal_focus_max + self.cal_min)
            self.cal_min = max(0.0, mid - span/2)
            self.cal_focus_max = min(1.0, mid + span/2)

        # Store calibration for this metric
        self.calib[self.cal_metric_name] = (self.cal_min, self.cal_focus_max)

        # Done
        self.in_calibration = False
        self.canvas.itemconfigure(
            self.msg_text,
            text=(f"Calibration complete for {self.cal_metric_name}.\n"
                  f"MIN={self.cal_min:0.2f}  MAX={self.cal_focus_max:0.2f}\n\n"
                  "Starting game...")
        )
        # Start a new round shortly
        self.root.after(1200, lambda: (self.canvas.itemconfigure(self.msg_text, text=""), self.restart()))

    # ------------ Drawing / Collision ------------
    def set_ball_pos(self, x, y):
        r = self.ball_r
        self.canvas.coords(self.ball_item, x - r, y - r, x + r, y + r)

    def intersects_ball(self, ox, oy, ow, oh):
        # AABB vs circle approx using AABB of the ball for simplicity (easy mode)
        bx1 = self.ball_x - self.ball_r
        by1 = self.ball_y - self.ball_r
        bx2 = self.ball_x + self.ball_r
        by2 = self.ball_y + self.ball_r
        ox1 = ox
        oy1 = oy
        ox2 = ox + ow
        oy2 = oy + oh
        return not (bx2 < ox1 or bx1 > ox2 or by2 < oy1 or by1 > oy2)

    def apply_calibration(self, metric_value):
        """Return normalized metric in [0,1] using per-mode calibration.

        Mapping: MAX -> top (norm=1), MIN -> bottom (norm=0).
        """
        pair = self.calib.get(self.control)
        if not pair:
            # Uncalibrated -> identity (already 0..1-ish)
            return metric_value
        mmin, mmax = pair
        if mmax <= mmin + 1e-6:
            return metric_value
        # Normalize 0..1, clamp
        norm = (metric_value - mmin) / (mmax - mmin)
        if norm < 0.0: norm = 0.0
        if norm > 1.0: norm = 1.0
        return norm

    # ------------ End & Messages ------------
    def end_game(self, win=False):
        self.game_over = True
        self.win = win
        msg = "You win! 🎉" if win else "Ouch! Collision."
        self.canvas.itemconfigure(self.msg_text, text=f"{msg}\nPress [R] to restart or use the button.")


# ----------------------------
# Entrypoint
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream-name", default="OpenSignals")
    ap.add_argument("--expect-ch", type=int, default=2)
    ap.add_argument("--demo", action="store_true", default=False)
    ap.add_argument("--control", choices=["concentration", "theta"], default="concentration")
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--spawn-interval", type=float, default=5.0)
    ap.add_argument("--fs", type=int, default=500, help="Fallback Fs if LSL provides 0")
    ap.add_argument("--buffer", type=float, default=1.0, help="EEG buffer length in seconds")
    args = ap.parse_args()

    cfg = EEGConfig(stream_name=args.stream_name, expect_ch=args.expect_ch, fs=args.fs,
                    demo=args.demo, smoothing_alpha=0.1, buffer=args.buffer)
    eeg = EEGSource(cfg)
    try:
        game = ScrollerGame(eeg, control=args.control, duration=args.duration,
                            spawn_interval=args.spawn_interval)
        game.start()
    finally:
        eeg.close()


if __name__ == "__main__":
    main()
