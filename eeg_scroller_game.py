#!/usr/bin/env python3
"""
Endless EEG Scroller (Theta-only) + Blink Calibration + Fullscreen

- Uses Theta metric only. Calibrate first (auto): 
  * MAX (blink): "Blink as fast as you can" for 5s  -> records MAX
  * MIN (calm):  "Relax the eyes, soften shoulders, long exhales" for 20s -> records MIN
  Ball is hidden; screen colors change during calibration.
- After calibration: "Get ready..." transition, then endless game starts.
- Difficulty ramps deterministically with elapsed time (same standard for everyone):
  speed↑, spawn interval↓, concurrent obstacles↑, obstacle size↑ (width & height).
- Obstacles are rectangles; touching top/bottom is allowed; collision ends the run.
- HUD shows Theta value, calibration state, and time survived.
- Fullscreen by default. Press ESC to exit.

Run:
    pip install numpy scipy pylsl
    python eeg_scroller_game.py --demo
    python eeg_scroller_game.py --stream-name OpenSignals --expect-ch 2
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
    raise SystemExit("Tkinter required. On Debian/Ubuntu: sudo apt-get install python3-tk") from e


# ----------------------------
# Signal processing helpers
# ----------------------------
def bandpower(sig, fs, fmin, fmax):
    if len(sig) < 2:
        return 0.0
    f, Pxx = welch(sig, fs=fs, nperseg=min(512, len(sig)))
    if len(f) < 2:
        return 0.0
    df = f[1] - f[0]
    mask = (f >= fmin) & (f <= fmax)
    return float(np.sum(Pxx[mask]) * df) if np.any(mask) else 0.0


def bandpass_filter(sig, fs, fmin, fmax, order=4):
    ny = 0.5 * fs
    low = max(1e-6, fmin / ny)
    high = min(0.9999, fmax / ny)
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, sig)


def band_envelope(sig, fs, fmin, fmax):
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
# LSL stream buffer
# ----------------------------
class StreamBuffer:
    def __init__(self, fs=500, ch=2, buffer=2.0):
        self.fs = int(fs)
        self.ch = int(ch)
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
    # Alpha-dominant with occasional beta bursts + noise
    t = np.arange(int(fs * dur)) / fs
    x = 20 * np.sin(2 * np.pi * 10 * t)
    if np.random.rand() < 0.35:
        x += 15 * np.sin(2 * np.pi * 20 * t)
    x += 5 * np.random.randn(len(t))
    sig = np.stack([np.zeros_like(x), x], axis=0)  # 2ch
    return sig.astype(np.float32)


# ----------------------------
# EEG metric source (Theta-only)
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
    smoothing_alpha: float = 0.2
    buffer: float = 2.0
    demo: bool = False


class EEGSource:
    """Computes Theta continuously; exposes smoothed value in [0,1]-ish."""
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
            print(f"[LSL] Connected to {info.name()} (Ch={info.channel_count()}, Fs={self.fs})")
        else:
            self.inlet = None
            print("[DEMO] Using synthetic EEG.")

        self.sbuf = StreamBuffer(fs=self.fs, ch=cfg.expect_ch, buffer=max(1.0, float(cfg.buffer)))
        if self.inlet is not None:
            self.reader_th = threading.Thread(target=reader_loop,
                                              args=(self.inlet, self.sbuf, self.stop_evt),
                                              daemon=True)
            self.reader_th.start()

        self.ema_theta = Ema(alpha=cfg.smoothing_alpha)
        self.last_update = 0.0
        self.update_hz = 10.0
        self.theta_m = 0.5

    def close(self):
        self.stop_evt.set()
        if getattr(self, "inlet", None) is not None:
            try:
                self.reader_th.join(timeout=0.3)
            except Exception:
                pass

    def _compute_theta(self, data):
        fs = self.fs
        ab = self.cfg.alpha_band
        tb = self.cfg.theta_band
        if data.shape[0] >= 2:
            left = data[0, :]
            right = data[1, :]
            a = 0.5 * (bandpower(left, fs, *ab) + bandpower(right, fs, *ab))
            th = 0.5 * (bandpower(left, fs, *tb) + bandpower(right, fs, *tb))
        else:
            sig = data[0, :]
            a = bandpower(sig, fs, *ab)
            th = bandpower(sig, fs, *tb)
        theta_metric = th / (th + a + 1e-9)
        return max(0.0, min(1.0, float(theta_metric)))

    def get_theta(self):
        now = time.time()
        if (now - self.last_update) >= (1.0 / self.update_hz):
            if self.cfg.demo:
                self.sbuf.append_chunk(synthetic_chunk(fs=self.fs, ch=2, dur=0.2).T)
            data = self.sbuf.snapshot()
            if data.shape[1] >= int(0.5 * self.fs):
                theta_m = self._compute_theta(data)
                self.theta_m = float(self.ema_theta.update(theta_m))
            self.last_update = now
        return self.theta_m


# ----------------------------
# Game with calibration & difficulty ramp
# ----------------------------
class ScrollerGame:
    # Colors
    BG_GAME = "#f6f7fa"  # off-white
    BG_BLINK = "#f6f7fa" # off-white for calibration (was red)
    BG_CALM  = "#f6f7fa" # off-white for calibration (was blue)


    def __init__(self, eeg: EEGSource, width=None, height=None, buffer_px=16):
        self.eeg = eeg
        self.buffer_px = buffer_px

        # Tk
        self.root = tk.Tk()
        self.root.title("Endless EEG Scroller — Theta (Blink Calibrated)")
        try:
            self.root.attributes("-fullscreen", True)
        except Exception:
            pass
        self.root.configure(bg=self.BG_GAME)

        # Get actual screen size for fullscreen
        self.root.update_idletasks()
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        self.W = screen_w
        self.H = screen_h

        self.canvas = tk.Canvas(self.root, width=self.W, height=self.H, bg=self.BG_GAME, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        # Map area 75% of screen height
        self.map_h = int(self.H * 0.75)
        self.map_top = (self.H - self.map_h) // 2
        self.map_bot = self.map_top + self.map_h

        # Ball
        self.ball_d = 44
        self.ball_r = self.ball_d // 2
        self.ball_x = self.W // 2
        self.ball_y = (self.map_top + self.map_bot) // 2

        # Obstacles (list of tuples: (id, x, y, w, h))
        self.obstacles = []
        self.last_spawn_t = 0.0

        # State
        self.game_running = False
        self.in_calibration = False
        self.calib_pair = None  # (min, max)
        self.start_time = None  # game start time
        self.run_time = 0.0     # time survived

        # Neumorphic channel background and bezelled ribbons
        # Draw channel shadow (subtle gradient effect)
        # Off-white neumorphic channel and ribbons
        # Main channel background (darkest)
        self.channel_bg = self.canvas.create_rectangle(
            0, self.map_top, self.W, self.map_bot,
            fill="#d1d5db", outline="", tags="channel_bg")

        # Top ribbon: raised with shadow (lightest)
        self.top_ribbon_shadow = self.canvas.create_rectangle(
            0, self.map_top-12, self.W, self.map_top+8,
            fill="#f6f7fa", outline="", tags="top_shadow")
        self.top_ribbon = self.canvas.create_rectangle(
            0, self.map_top-8, self.W, self.map_top+4,
            fill="#f8f9fb", outline="#e0e1e6", width=2, tags="top_ribbon")

        # Bottom ribbon: raised with shadow (lightest)
        self.bot_ribbon_shadow = self.canvas.create_rectangle(
            0, self.map_bot-8, self.W, self.map_bot+12,
            fill="#f6f7fa", outline="", tags="bot_shadow")
        self.bot_ribbon = self.canvas.create_rectangle(
            0, self.map_bot-4, self.W, self.map_bot+8,
            fill="#f8f9fb", outline="#e0e1e6", width=2, tags="bot_ribbon")

        # Channel edges (remain for separation, slightly darker than bg)
        self.channel_inner_shadow = self.canvas.create_line(
            0, self.map_top+2, self.W, self.map_top+2,
            fill="#bfc3c9", width=6, stipple="gray50", tags="inner_shadow")
        self.channel_inner_shadow2 = self.canvas.create_line(
            0, self.map_bot-2, self.W, self.map_bot-2,
            fill="#bfc3c9", width=6, stipple="gray50", tags="inner_shadow2")

        # Ball: 3D look with highlight and shadow
        self.ball_shadow = self.canvas.create_oval(
            self.ball_x - self.ball_r, self.ball_y + self.ball_r * 0.7,
            self.ball_x + self.ball_r, self.ball_y + self.ball_r * 1.25,
            fill="#b0c4de", outline="", tags="ball_shadow")
        self.ball_item = self.canvas.create_oval(
            self.ball_x - self.ball_r, self.ball_y - self.ball_r,
            self.ball_x + self.ball_r, self.ball_y + self.ball_r,
            fill="deepskyblue", outline="#1e90ff", width=2, tags="ball_main")
        # Rolling highlight (will be animated)
        self.ball_highlight = self.canvas.create_oval(
            self.ball_x - self.ball_r * 0.45, self.ball_y - self.ball_r * 0.7,
            self.ball_x - self.ball_r * 0.05, self.ball_y - self.ball_r * 0.25,
            fill="#e0f6ff", outline="", tags="ball_highlight")

            # HUD
        # Use a modern, scientific font (fallback to Helvetica if not available)
        hud_font = ("Segoe UI Semibold", 18, "bold")
        msg_font = ("Segoe UI Semibold", 44, "bold")
        self.hud_text = self.canvas.create_text(self.W // 2, 36, fill="#111111", font=hud_font, text="", state="hidden")
        self.msg_text = self.canvas.create_text(self.W // 2, self.H // 2, fill="#111111", font=msg_font, text="")

        # Neumorphic time box (recessed)
        self.time_box = self.canvas.create_rectangle(self.W//2-90, 10, self.W//2+90, 54,
                    fill="#ececf0", outline="#d1d5db", width=2, tags="hud")
        self.time_box_shadow = self.canvas.create_line(self.W//2-90, 54, self.W//2+90, 54,
                    fill="#c0c2c7", width=4, tags="hud")
        self.time_text = self.canvas.create_text(self.W//2, 32, fill="#111", font=("Segoe UI Semibold", 24, "bold"),
                            text="0.0 s", tags="hud")

        # Buttons (bottom)
        self.btn_frame = tk.Frame(self.root, bg=self.BG_GAME)
        self.btn_frame.pack(pady=18)
        self.btn_frame.lift()  # Ensure button frame is above canvas
        btn_style = {
            'font': ("Segoe UI Semibold", 16),
            'bg': "#f6f7fa",
            'fg': "#111",
            'activebackground': "#ececf0",
            'activeforeground': "#111",
            'relief': tk.RAISED,
            'bd': 2,
            'highlightthickness': 2,
            'highlightbackground': "#d1d5db",
            'padx': 24,
            'pady': 10,
        }
        self.restart_btn = tk.Button(self.btn_frame, text="Restart (R)", command=self.restart, **btn_style)
        self.calib_btn   = tk.Button(self.btn_frame, text="Calibrate", command=self.start_calibration, **btn_style)
        self.restart_btn.pack(side="left", padx=18)
        self.calib_btn.pack(side="left", padx=18)
        self.restart_btn.update_idletasks()
        self.calib_btn.update_idletasks()
        for btn in [self.restart_btn, self.calib_btn]:
            btn.configure(cursor="hand2")

        # Keys
        self.root.bind("<Escape>", lambda e: self.root.destroy())
        self.root.bind("<r>", lambda e: self.restart())
        self.root.bind("<R>", lambda e: self.restart())
        self.root.bind("<c>", lambda e: self.start_calibration())
        self.root.bind("<C>", lambda e: self.start_calibration())

    # ---------- Public ----------
    def start(self):
        # Auto-calibrate on startup
        self.start_calibration(auto=True)
        self._schedule_frame()
        self.root.mainloop()

    def restart(self):
        # Clear obstacles
        for oid, *_ in self.obstacles:
            self.canvas.delete(oid)
        self.obstacles.clear()
        self.last_spawn_t = 0.0
        self.run_time = 0.0
        self.game_running = False
        self.canvas.itemconfigure(self.msg_text, text="")
        self.set_ball_visible(True)
        self.canvas.config(bg=self.BG_GAME)
        # Start the game directly, skip calibration
        self._begin_run()

    # ---------- Calibration ----------
    CAL_MAX_BLINK_DUR = 5.0   # seconds
    CAL_MIN_RELAX_DUR = 20.0  # seconds

    def set_ball_visible(self, visible: bool):
        self.canvas.itemconfigure(self.ball_item, state=("normal" if visible else "hidden"))

    def start_calibration(self, auto=False):
        if self.in_calibration:
            return
        self.in_calibration = True
        self.game_running = False
        self.set_ball_visible(False)
        self.canvas.config(bg=self.BG_BLINK)
        phase = "Auto " if auto else ""
        self.canvas.itemconfigure(self.msg_text, text=f"{phase}Calibration — BLINK\n\nBlink as fast as you can!")
        self.cal_max = -1e9
        self.cal_min = 1e9
        self._calm_thetas = []  # Store theta values during CALM phase
        self.cal_phase = "blink"
        self.cal_switch_t = time.time() + self.CAL_MAX_BLINK_DUR
        self._calibration_tick()

    def _calibration_tick(self):
        if not self.in_calibration:
            return
        theta = self.eeg.get_theta()
        if self.cal_phase == "blink":
            if theta > self.cal_max: self.cal_max = theta
            if time.time() >= self.cal_switch_t:
                # Calm phase
                self.cal_phase = "calm"
                self.canvas.config(bg=self.BG_CALM)
                self.canvas.itemconfigure(self.msg_text,
                                          text="Calibration — CALM\n\nRelax the eyes.\nSoften shoulders.\nLong, gentle exhales...")
                self.cal_switch_t = time.time() + self.CAL_MIN_RELAX_DUR
        else:
            self._calm_thetas.append(theta)
            if time.time() >= self.cal_switch_t:
                self._calibration_finish()
                return

        # Tick at ~12.5 Hz
        self.root.after(80, self._calibration_tick)

    def _calibration_finish(self):

        # Use average of calm thetas for cal_min
        if hasattr(self, '_calm_thetas') and self._calm_thetas:
            self.cal_min = float(np.mean(self._calm_thetas))
        # Safety fallbacks
        if self.cal_max <= -1e8 or self.cal_min >= 1e8:
            cur = self.eeg.get_theta()
            self.cal_max = max(0.6, float(cur))
            self.cal_min = min(0.4, float(cur))

        # Ensure min < max with minimum span
        if self.cal_max <= self.cal_min:
            mid = 0.5 * (self.cal_max + self.cal_min)
            self.cal_min = max(0.0, mid - 0.1)
            self.cal_max = min(1.0, mid + 0.1)
        if (self.cal_max - self.cal_min) < 0.05:
            span = 0.05
            mid = 0.5 * (self.cal_max + self.cal_min)
            self.cal_min = max(0.0, mid - span/2)
            self.cal_max = min(1.0, mid + span/2)

        self.calib_pair = (self.cal_min, self.cal_max)
        self.in_calibration = False

        # Transition: get ready
        self.canvas.config(bg=self.BG_GAME)
        self.canvas.itemconfigure(self.msg_text,
                                  text=(f"Calibration complete.\n"
                                        f"MIN={self.cal_min:0.2f}  MAX={self.cal_max:0.2f}\n\n"
                                        "Get ready..."))
        self.root.after(1500, self._begin_run)

    def apply_calibration(self, theta_value):
        if not self.calib_pair:
            return theta_value  # identity if not calibrated
        mmin, mmax = self.calib_pair
        if mmax <= mmin + 1e-6:
            return theta_value
        norm = (theta_value - mmin) / (mmax - mmin)
        return 0.0 if norm < 0 else (1.0 if norm > 1 else norm)

    # ---------- Run loop ----------
    def _begin_run(self):
        self.canvas.itemconfigure(self.msg_text, text="")
        self.set_ball_visible(True)
        self.game_running = True
        self.start_time = time.time()
        self.last_spawn_t = 0.0

    def _schedule_frame(self):
        self.root.after(16, self._frame)  # ~60 FPS

    def _frame(self):
        # Keep scheduling frames
        self._schedule_frame()

        # If calibrating or not running yet, just update HUD
        if self.in_calibration or not self.game_running:
            theta = self.eeg.get_theta()
            theta_n = self.apply_calibration(theta)
            cal_state = "On" if self.calib_pair else "Off"
            self.canvas.itemconfigure(self.hud_text, text=f"Theta: {theta_n:0.2f}  (Cal:{cal_state})")
            # Always show time box and text, but keep at 0.0
            self.canvas.coords(self.time_box, self.W//2-90, 10, self.W//2+90, 54)
            self.canvas.coords(self.time_box_shadow, self.W//2-90, 54, self.W//2+90, 54)
            self.canvas.itemconfigure(self.time_text, text="0.0 s")
            self.canvas.tag_raise("hud")
            return

        # Time
        now = time.time()
        self.run_time = now - self.start_time

        # Difficulty schedule (deterministic by time)
        # Speed (px/s): base + linear ramp
        speed = self._speed(self.run_time)
        # Spawn interval (s): decreasing toward min
        spawn_interval = self._spawn_interval(self.run_time)
        # Max concurrent obstacles
        max_obs = self._max_obstacles(self.run_time)

        # Update Theta -> ball position
        theta = self.eeg.get_theta()
        theta_n = self.apply_calibration(theta)
        target_y = int(self.map_top + (1.0 - theta_n) * self.map_h)
        self.ball_y = int(0.6 * self.ball_y + 0.4 * target_y)
        self._set_ball(self.ball_x, self.ball_y)

        # Spawn obstacles respecting concurrency & interval
        if (len(self.obstacles) < max_obs) and ((now - self.last_spawn_t) >= spawn_interval):
            self._spawn_obstacle(self.run_time)
            self.last_spawn_t = now

        # Move obstacles & detect collisions
        dx = -speed / 60.0  # per frame
        to_delete = []
        bx1 = self.ball_x - self.ball_r
        by1 = self.ball_y - self.ball_r
        bx2 = self.ball_x + self.ball_r
        by2 = self.ball_y + self.ball_r

        for i, (oids, ox, oy, ow, oh) in enumerate(self.obstacles):
            ox_new = ox + dx
            for oid in (oids if isinstance(oids, (list, tuple)) else [oids]):
                self.canvas.move(oid, dx, 0)
            if ox_new + ow < 0:
                to_delete.append(i)
            else:
                # AABB check
                ox1 = ox_new
                oy1 = oy
                ox2 = ox_new + ow
                oy2 = oy + oh
                hit = not (bx2 < ox1 or bx1 > ox2 or by2 < oy1 or by1 > oy2)
                if hit:
                    self._end_run()
                    return
                self.obstacles[i] = (oids, ox_new, oy, ow, oh)

        # Remove off-screen obstacles
        for i in reversed(to_delete):
            oids, *_ = self.obstacles[i]
            for oid in (oids if isinstance(oids, (list, tuple)) else [oids]):
                self.canvas.delete(oid)
            del self.obstacles[i]

        # HUD
        # Always show the time box and time value, high contrast
        self.canvas.coords(self.time_box, self.W//2-90, 10, self.W//2+90, 54)
        self.canvas.coords(self.time_box_shadow, self.W//2-90, 54, self.W//2+90, 54)
        self.canvas.itemconfigure(self.time_text, text=f"{self.run_time:0.1f} s")
        self.canvas.tag_raise("hud")

    # ---------- Difficulty functions (deterministic) ----------
    def _speed(self, t):
        base = 220.0
        growth = 3.8   # px/s^2
        return base + growth * t

    def _spawn_interval(self, t):
        base = 1.6    # s
        min_iv = 0.45 # s
        k = 0.018     # decay rate
        iv = base / (1.0 + k * t)
        return min_iv if iv < min_iv else iv

    def _max_obstacles(self, t):
        # Start at 1, add 1 every 20s up to 4
        return int(min(1 + (t // 20), 4))

    def _size_scales(self, t):
        # Grow sizes slowly over time
        h_scale = 1.0 + 0.015 * min(t, 60)   # cap at +90% by 60s
        w_scale = 1.0 + 0.010 * min(t, 60)   # cap at +60% by 60s
        return h_scale, w_scale

    # ---------- Obstacles ----------
    def _spawn_obstacle(self, t):
        h_scale, w_scale = self._size_scales(t)
        base_h = self.ball_d
        base_w = 70
        oh = int(base_h * h_scale)
        ow = int(base_w * w_scale)
        oy = random.randint(self.map_top, max(self.map_top, self.map_bot - oh))
        ox = self.W + ow
        # Alternate seagreen and purple
        color1 = "#2ec4b6" if random.random() < 0.5 else "#7c3aed"
        color2 = "#a0f0e0" if color1 == "#2ec4b6" else "#d1b3ff"
        # Shadow (offset, blurred look)
        shadow = self.canvas.create_rectangle(
            ox+6, oy+8, ox+ow+6, oy+oh+8,
            fill="#d1d5db", outline="", tags="obs_shadow")
        # Main obstacle
        oid = self.canvas.create_rectangle(
            ox, oy, ox+ow, oy+oh,
            fill=color1, outline=color1, width=0, tags="obs_main")
        # Highlight (top half, lighter color)
        highlight = self.canvas.create_rectangle(
            ox, oy, ox+ow, oy+oh//2,
            fill=color2, outline="", stipple="gray25", tags="obs_highlight")
        self.obstacles.append(((shadow, oid, highlight), ox, oy, ow, oh))

    # ---------- Drawing ----------
    def _set_ball(self, x, y):
        r = self.ball_r
        # Ball shadow
        self.canvas.coords(self.ball_shadow,
            x - r, y + r * 0.7,
            x + r, y + r * 1.25)
        # Ball main
        self.canvas.coords(self.ball_item,
            x - r, y - r, x + r, y + r)
        # Animate highlight to simulate rolling
        # The highlight moves horizontally based on ball position (or time)
        import math
        t = time.time()
        angle = (t * 2.5) % (2 * math.pi)
        # Reverse direction by negating angle
        hl_x = x + r * 0.4 * math.cos(-angle)
        hl_y = y - r * 0.4 * math.sin(-angle)
        self.canvas.coords(self.ball_highlight,
            hl_x - r * 0.25, hl_y - r * 0.25,
            hl_x + r * 0.15, hl_y + r * 0.15)

    # ---------- End / scoring ----------
    def _end_run(self):
        self.game_running = False
        for oid, *_ in self.obstacles:
            self.canvas.delete(oid)
        self.obstacles.clear()
        self.canvas.itemconfigure(self.msg_text,
                                  text=f"Collision!\nTime survived: {self.run_time:0.1f}s\n\nPress [R] to try again.\nPress [C] to recalibrate.")

# ----------------------------
# Entrypoint
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream-name", default="OpenSignals")
    ap.add_argument("--expect-ch", type=int, default=2)
    ap.add_argument("--demo", action="store_true", default=False)
    ap.add_argument("--fs", type=int, default=500)
    ap.add_argument("--buffer", type=float, default=2.0)
    args = ap.parse_args()

    cfg = EEGConfig(stream_name=args.stream_name, expect_ch=args.expect_ch, fs=args.fs,
                    demo=args.demo, smoothing_alpha=0.15, buffer=args.buffer)
    eeg = EEGSource(cfg)
    try:
        game = ScrollerGame(eeg)
        game.start()
    finally:
        eeg.close()


if __name__ == "__main__":
    main()
