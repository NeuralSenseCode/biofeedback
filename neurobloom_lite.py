#!/usr/bin/env python3
"""
NeuroBloom Lite — 3-minute EEG biofeedback demo (Python + Pygame + LSL)
F3/F4-friendly version with blink “seed” bursts and guided fallback.

Controls:
  F11  = fullscreen toggle
  R    = reset 3-minute session
  D    = toggle guided/demo mode
  S    = save PNG snapshot immediately
  ESC  = quit

Electrodes (3-lead device):
  Active 1 -> F3   | Active 2 -> F4   | Ground/Ref -> mastoid (A1/A2)

Run:
  pip install numpy scipy pygame pylsl
  python neurobloom_lite.py --mode lsl --brand "Neural Sense — Applied Neuroscience"
  (or) python neurobloom_lite.py --mode demo
"""

import argparse
import math
import os
import time
import threading
from collections import deque
from datetime import datetime

import numpy as np
import pygame

# --- Optional imports so demo still runs without them ---
try:
    from pylsl import resolve_byprop, StreamInlet
    HAVE_LSL = True
except Exception as e:
    print("[INFO] pylsl not available or failed to import. LSL mode will be unavailable:", e)
    HAVE_LSL = False

try:
    from scipy.signal import butter, lfilter, iirnotch, welch
    HAVE_SCIPY = True
except Exception as e:
    print("[INFO] SciPy not available or failed to import. Realtime filtering disabled:", e)
    HAVE_SCIPY = False


def ema_update(prev, x, alpha=0.2):
    """Exponential moving average update."""
    return x if prev is None else alpha * x + (1 - alpha) * prev


def bandpower_welch(sig, fs, fmin, fmax):
    """Band power via Welch PSD integration."""
    if len(sig) < max(128, int(0.5 * fs)):  # need some data
        return 0.0
    f, Pxx = welch(sig, fs=fs, nperseg=min(512, len(sig)))
    if len(f) < 2:
        return 0.0
    df = f[1] - f[0]
    mask = (f >= fmin) & (f <= fmax)
    return float(np.sum(Pxx[mask]) * df)


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


# --------------------------- Filters ---------------------------

class FilterBank:
    """1–40 Hz band-pass + 50 Hz notch (South Africa mains)"""
    def __init__(self):
        self.fs = None
        self.bp_b = self.bp_a = None
        self.notch_b = self.notch_a = None

    def design(self, fs: int):
        if not HAVE_SCIPY:
            return
        if self.fs == fs and self.bp_b is not None:
            return
        self.fs = fs
        nyq = fs / 2.0
        # 1–40 Hz band-pass
        self.bp_b, self.bp_a = butter(4, [1.0 / nyq, 40.0 / nyq], btype='band')
        # 50 Hz notch
        w0 = 50.0 / nyq
        try:
            self.notch_b, self.notch_a = iirnotch(w0, Q=30.0)
        except Exception:
            self.notch_b = self.notch_a = None

    def apply(self, x: np.ndarray) -> np.ndarray:
        if not HAVE_SCIPY or self.bp_b is None:
            return x
        y = lfilter(self.bp_b, self.bp_a, x)
        if self.notch_b is not None:
            y = lfilter(self.notch_b, self.notch_a, y)
        return y


# --------------------------- EEG Processor ---------------------------

class EEGProcessor:
    """
    Realtime metrics tuned for frontal F3/F4:
      - CALM index = alpha / (alpha + high-frequency)  (alpha: 8–12 Hz; HF: 20–35 Hz)
      - FOCUS index = beta / (alpha + theta)           (beta: 15–25 Hz; theta: 4–7 Hz)
      - Blink detector on best channel -> visual "seed" burst
    """
    def __init__(self, max_secs=10.0):
        self.buffers = []       # deque per channel
        self.fs = None
        self.filter = FilterBank()
        self.lock = threading.Lock()
        self.max_samples = int(256 * max_secs)  # provisional until fs known

        # Metrics (self.alpha represents CALM for compatibility with render code)
        self.alpha = 0.0       # CALM index
        self.focus = 0.0       # FOCUS index
        self.quality = 1.0

        # Smoothing + history for normalization
        self.alpha_ema = None
        self.focus_ema = None
        self.alpha_hist = deque(maxlen=120)

        # Blink state
        self.last_blink = 0.0
        self.blinked = False

    def set_fs(self, fs: int):
        if self.fs != fs:
            self.fs = fs
            self.filter.design(fs)
            # resize based on seconds
            with self.lock:
                self.max_samples = int(self.fs * 10.0)
                if self.buffers:
                    n = len(self.buffers)
                    self.buffers = [deque(list(b)[-self.max_samples:], maxlen=self.max_samples) for b in self.buffers]

    def set_channels(self, n: int):
        with self.lock:
            self.buffers = [deque(maxlen=self.max_samples) for _ in range(n)]

    def push(self, samples: np.ndarray, fs: int):
        """
        samples: shape (n_samples, n_channels)
        """
        if self.fs is None:
            self.set_fs(fs)
            self.set_channels(samples.shape[1])
        with self.lock:
            for c in range(samples.shape[1]):
                self.buffers[c].extend(samples[:, c])

    def compute(self) -> bool:
        """Update CALM/FOCUS/quality + blink detection. Call ~10–20 Hz."""
        if self.fs is None or not HAVE_SCIPY:
            return False

        # Copy buffers
        with self.lock:
            chans = [np.asarray(list(buf), dtype=np.float32) if len(buf) >= int(self.fs * 2) else None
                     for buf in self.buffers]

        if not any(ch is not None for ch in chans):
            return False

        calm_vals, focus_vals, q_vals = [], [], []
        best_q = -1.0
        y_best = None

        for ch in chans:
            if ch is None:
                calm_vals.append(0.0); focus_vals.append(0.0); q_vals.append(0.0)
                continue

            y = self.filter.apply(ch - np.mean(ch))

            # crude clip detector: if many samples are near extremes, lower quality
            y_abs = np.abs(y)
            p99 = np.percentile(y_abs, 99.9)
            clip_frac = float(np.mean(y_abs >= p99))  # fraction of near-rail points
            q_clip = clamp(1.0 - 5.0 * clip_frac, 0.0, 1.0)  # harsh penalty if lots of clipping


            # Band powers
            a   = bandpower_welch(y, self.fs, 8.0, 12.0)    # alpha
            th  = bandpower_welch(y, self.fs, 4.0, 7.0)     # theta
            b   = bandpower_welch(y, self.fs, 15.0, 25.0)   # beta
            hf  = bandpower_welch(y, self.fs, 20.0, 35.0)   # high freq (EMG-ish)
            ln  = bandpower_welch(y, self.fs, 49.0, 51.0)   # 50 Hz line
            tot = bandpower_welch(y, self.fs, 1.0, 40.0) + 1e-9

            # Quality: low line noise + not dominated by HF
            q_line = clamp(1.0 - (ln / tot), 0.0, 1.0)
            q_hf   = clamp(1.0 - (hf / tot), 0.0, 1.0)
            q      = 0.6 * q_line + 0.4 * q_hf
            q = 0.5*q + 0.5*q_clip


            calm  = a / (a + hf + 1e-9)         # higher when alpha present and EMG low
            focus = b / (a + th + 1e-9)         # higher when beta rises vs alpha+theta

            calm_vals.append(calm)
            focus_vals.append(focus)
            q_vals.append(q)

            if q > best_q:
                best_q = q
                y_best = y

        qarr = np.array(q_vals)
        calm_arr = np.array(calm_vals)
        focus_arr = np.array(focus_vals)

        if np.sum(qarr > 0.4) >= 2:
            calm  = float(np.mean(calm_arr[qarr > 0.4]))
            focus = float(np.mean(focus_arr[qarr > 0.4]))
            quality = float(np.mean(qarr[qarr > 0.4]))
        else:
            idx = int(np.argmax(qarr))
            calm  = float(calm_arr[idx])
            focus = float(focus_arr[idx])
            quality = float(qarr[idx])

        # Blink detection on best channel (frontal-friendly)
        self.blinked = False
        if y_best is not None:
            N = max(int(self.fs * 0.5), 50)       # last 0.5 s
            yw = y_best[-N:]
            dy = np.abs(np.diff(yw))
            if dy.size:
                med = float(np.median(dy))
                mad = float(np.median(np.abs(dy - med))) + 1e-9
                thr = med + 4.0 * mad
                now = time.time()
                if dy.max() > thr and now - getattr(self, "last_blink", 0.0) > 0.6:
                    self.blinked = True
                    self.last_blink = now

        # Smooth & store
        self.alpha_ema = ema_update(self.alpha_ema, calm, 0.2)   # reuse "alpha" slot to mean CALM
        self.focus_ema = ema_update(self.focus_ema, focus, 0.2)

        self.alpha   = float(self.alpha_ema if self.alpha_ema is not None else calm)
        self.focus   = float(self.focus_ema if self.focus_ema is not None else focus)
        self.quality = float(quality)

        self.alpha_hist.append(self.alpha)
        return True

    def alpha_norm(self) -> float:
        """Normalize CALM for visuals using rolling deciles."""
        if len(self.alpha_hist) < 10:
            return 0.3
        arr = np.array(self.alpha_hist)
        lo = np.percentile(arr, 10)
        hi = np.percentile(arr, 90)
        if hi <= lo:
            return 0.3
        return float(clamp((self.alpha - lo) / (hi - lo), 0.0, 1.0))

    def focus_norm(self) -> float:
        """FOCUS is already ~0–1 range by construction; clamp to [0,1]."""
        return float(clamp(self.focus, 0.0, 1.0))


# --------------------------- LSL Reader ---------------------------

class LSLReader(threading.Thread):
    """Reads EEG chunks from LSL and pushes into the EEGProcessor."""
    def __init__(self, processor: EEGProcessor, stream_name=None, stream_type='EEG'):
        super().__init__(daemon=True)
        self.processor = processor
        self.stream_name = stream_name
        self.stream_type = stream_type
        self.stop_flag = threading.Event()
        self.connected = False
        self.fs = None
        self.inlet = None

    def connect(self) -> bool:
        if not HAVE_LSL:
            return False
        try:
            streams = resolve_byprop('name', self.stream_name, timeout=2.0) if self.stream_name \
                      else resolve_byprop('type', self.stream_type, timeout=2.0)
            if not streams:
                return False
            info = streams[0]
            self.inlet = StreamInlet(info, max_buflen=60)
            ch = info.channel_count()
            fs = int(info.nominal_srate()) or 256
            self.fs = fs
            self.processor.set_fs(fs)
            self.processor.set_channels(ch)
            self.connected = True
            print(f"[LSL] Connected: {info.name()} ({ch} ch @ {fs} Hz)")
            return True
        except Exception as e:
            print("[LSL] Connect failed:", e)
            self.connected = False
            return False

    def run(self):
        while not self.stop_flag.is_set():
            if not self.connected:
                if not self.connect():
                    time.sleep(0.5)
                    continue
            try:
                chunk, timestamps = self.inlet.pull_chunk(timeout=0.02, max_samples=64)
                if timestamps and len(chunk) > 0:
                    arr = np.asarray(chunk, dtype=np.float32)
                    # Keep first two channels for simplicity (F3/F4)
                    if arr.shape[1] > 2:
                        arr = arr[:, :2]

                    arr = np.asarray(chunk, dtype=np.float32)
                    if arr.shape[1] > 2:
                        arr = arr[:, :2]

                    push_fs = self.fs
                    if self.fs >= 800:          # 1000 Hz -> ~250 Hz
                        arr = arr[::4]
                        push_fs = int(self.fs / 4)

                    self.processor.push(arr, self.fs)
            except Exception as e:
                print("[LSL] Read error, attempting reconnect:", e)
                self.connected = False
                time.sleep(0.5)

    def stop(self):
        self.stop_flag.set()


# --------------------------- Visuals ---------------------------

class Particles:
    """Simple burst particles for blink seeds and focus sparks."""
    def __init__(self):
        self.particles = []

    def spawn(self, x, y, n=10, speed_lo=60, speed_hi=140):
        for _ in range(n):
            angle = np.random.uniform(0, 2 * math.pi)
            speed = np.random.uniform(speed_lo, speed_hi)
            life = np.random.uniform(0.6, 1.2)
            self.particles.append([x, y, speed * math.cos(angle), speed * math.sin(angle), life])

    def update(self, dt):
        alive = []
        for x, y, vx, vy, life in self.particles:
            life -= dt
            if life > 0:
                x += vx * dt
                y += vy * dt
                vx *= 0.98
                vy *= 0.98
                alive.append([x, y, vx, vy, life])
        self.particles = alive

    def draw(self, surf):
        for x, y, vx, vy, life in self.particles:
            a = int(255 * clamp(life, 0, 1))
            pygame.draw.circle(surf, (255, 255, 255, a), (int(x), int(y)), 3)


class NeuroBloomGame:
    def __init__(self, args):
        pygame.init()
        pygame.display.set_caption("NeuroBloom Lite — EEG Biofeedback (F3/F4)")
        self.screen = pygame.display.set_mode((1280, 720), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()

        self.args = args
        self.processor = EEGProcessor(max_secs=10.0)

        self.reader = None
        self.demo_mode = (args.mode == 'demo') or (not HAVE_LSL) or (not HAVE_SCIPY)
        if args.mode == 'lsl' and HAVE_LSL and HAVE_SCIPY:
            self.reader = LSLReader(self.processor, stream_name=args.stream_name, stream_type=args.stream_type)
            self.reader.start()

        # Session & state
        self.session_seconds = 180.0
        self.elapsed = 0.0
        self.bad_quality_time = 0.0
        self.guided = self.demo_mode
        self.guided_phase = 0.0

        # Visuals
        self.particles = Particles()
        self.title_font = pygame.font.SysFont("Century Gothic", 42, bold=True)
        self.small_font = pygame.font.SysFont("Century Gothic", 22)
        self.brand_text = args.brand or "Neural Sense — Applied Neuroscience"

        # Output folder
        self.outdir = os.path.join(os.getcwd(), "Sessions", datetime.now().strftime("%Y-%m-%d"))
        os.makedirs(self.outdir, exist_ok=True)

    # ---------- helpers ----------
    def toggle_fullscreen(self):
        flags = self.screen.get_flags()
        if flags & pygame.FULLSCREEN:
            self.screen = pygame.display.set_mode((1280, 720), pygame.RESIZABLE)
        else:
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)

    def save_snapshot(self, suffix=""):
        filename = f"NeuroBloom_{datetime.now().strftime('%H%M%S')}{suffix}.png"
        path = os.path.join(self.outdir, filename)
        pygame.image.save(self.screen, path)
        print(f"[Saved] {path}")

    def reset(self):
        self.elapsed = 0.0
        self.bad_quality_time = 0.0
        self.guided = self.demo_mode
        self.guided_phase = 0.0
        self.particles = Particles()
        self.processor = EEGProcessor(max_secs=10.0)
        if self.reader is not None:
            self.reader.stop()
            self.reader.join(timeout=0.3)
            self.reader = LSLReader(self.processor, stream_name=self.args.stream_name, stream_type=self.args.stream_type)
            self.reader.start()

    # ---------- main loop pieces ----------
    def update_logic(self, dt):
        ok = self.processor.compute()
        q = self.processor.quality if ok else 0.0

        # Decide guided/fallback (in demo, guided stays on)
        if self.demo_mode:
            self.guided = True
        else:
            if q < 0.35:
                self.bad_quality_time += dt
            else:
                self.bad_quality_time = max(0.0, self.bad_quality_time - 2 * dt)
            self.guided = self.bad_quality_time > 4.0

        # Compute normalized indices
        if self.guided:
            self.guided_phase += 0.6 * dt
            alpha_norm = 0.35 + 0.25 * (0.5 + 0.5 * math.sin(self.guided_phase))     # breathing-like growth
            focus_norm = 0.20 + 0.20 * (0.5 + 0.5 * math.sin(1.7 * self.guided_phase))
        else:
            alpha_norm = self.processor.alpha_norm()   # CALM
            focus_norm = self.processor.focus_norm()   # FOCUS

        # Blink → seed burst (very visible)
        if getattr(self.processor, "blinked", False):
            w, h = self.screen.get_size()
            self.particles.spawn(w / 2, h / 2, n=18, speed_lo=80, speed_hi=160)
            self.processor.blinked = False

        # Focus → extra sparks
        if focus_norm > 0.6 and np.random.rand() < 0.25:
            w, h = self.screen.get_size()
            self.particles.spawn(w / 2, h / 2, n=int(6 + 10 * focus_norm))

        self.particles.update(dt)
        self.elapsed += dt
        return alpha_norm, focus_norm, q

    def draw(self, alpha_norm, focus_norm, q):
        w, h = self.screen.get_size()
        surf = pygame.Surface((w, h), pygame.SRCALPHA)

        # Background tint scales with CALM
        base = int(30 + 160 * alpha_norm)
        self.screen.fill((10, 10, 10))
        pygame.draw.rect(self.screen, (base // 2, base // 3, base), (0, 0, w, h))

        # Luminous orb
        cx, cy = w // 2, h // 2
        radius = int(80 + 260 * alpha_norm)
        for i in range(6):
            r = radius + 18 * i
            a = int(120 * (1 - i / 6.0))
            pygame.draw.circle(surf, (255, 255, 255, a), (cx, cy), r)
        pygame.draw.circle(surf, (255, 255, 255, 240), (cx, cy), radius // 2)

        # Particles
        self.particles.draw(surf)

        # Circular 3-minute timer
        frac = clamp(1.0 - self.elapsed / self.session_seconds, 0.0, 1.0)
        end_angle = -math.pi / 2 + 2 * math.pi * (1.0 - frac)
        pygame.draw.arc(surf, (255, 255, 255, 220),
                        (cx - 320, cy - 320, 640, 640), -math.pi / 2, end_angle, width=6)

        # Branding + subtle HUD
        brand = self.title_font.render(self.brand_text, True, (255, 255, 255))
        self.screen.blit(brand, (int(w * 0.05), int(h * 0.08)))
        sub = self.small_font.render("Blink to seed • Relax to grow • Focus to spark", True, (240, 240, 240))
        self.screen.blit(sub, (int(w * 0.05), int(h * 0.08) + 56))

        if not self.demo_mode:
            qcol = (180, 255, 180) if q > 0.6 else (255, 220, 120) if q > 0.35 else (255, 160, 160)
            qtxt = self.small_font.render(f"Signal {int(q * 100)}%", True, qcol)
            self.screen.blit(qtxt, (w - qtxt.get_width() - 20, 20))
        if self.guided:
            gtxt = self.small_font.render("Guided mode", True, (255, 200, 140))
            self.screen.blit(gtxt, (w - gtxt.get_width() - 20, 44))

        # End-card overlay in last 5 seconds
        if self.session_seconds - self.elapsed < 5.0:
            f = clamp((5.0 - (self.session_seconds - self.elapsed)) / 5.0, 0, 1)
            overlay = pygame.Surface((w, h), pygame.SRCALPHA)
            pygame.draw.rect(overlay, (0, 0, 0, int(200 * f)), (0, 0, w, h))
            txt = self.title_font.render("Thanks for growing your Mind Garden", True, (255, 255, 255))
            overlay.blit(txt, (cx - txt.get_width() // 2, cy - 40))
            small = self.small_font.render("Press R for next visitor • S to save snapshot", True, (220, 220, 220))
            overlay.blit(small, (cx - small.get_width() // 2, cy + 10))
            self.screen.blit(overlay, (0, 0))

        self.screen.blit(surf, (0, 0))
        pygame.display.flip()

    def run(self):
        snapshot_done = False
        while True:
            dt = self.clock.tick(60) / 1000.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    if self.reader: self.reader.stop(); self.reader.join(timeout=0.5)
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        if self.reader: self.reader.stop(); self.reader.join(timeout=0.5)
                        return
                    elif event.key == pygame.K_F11:
                        self.toggle_fullscreen()
                    elif event.key == pygame.K_r:
                        self.reset()
                    elif event.key == pygame.K_s:
                        self.save_snapshot("_manual")
                    elif event.key == pygame.K_d:
                        # Manual guided/demo toggle (handy if contact is poor)
                        self.guided = not self.guided
                        print("[Guided mode]", "ON" if self.guided else "OFF")

            alpha_norm, focus_norm, q = self.update_logic(dt)
            self.draw(alpha_norm, focus_norm, q)

            # Auto-snapshot near the end
            if not snapshot_done and self.elapsed > self.session_seconds - 2.0:
                self.save_snapshot()
                snapshot_done = True


# --------------------------- Entrypoint ---------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['lsl', 'demo'], default='lsl', help='Data source: LSL (OpenSignals) or demo')
    p.add_argument('--stream-name', type=str, default="OpenSignals", help='Specific LSL stream name (optional)')
    p.add_argument('--stream-type', type=str, default='00:07:80:89:80:02', help='LSL stream type (default: EEG)')
    p.add_argument('--brand', type=str, default="Neural Sense — Applied Neuroscience", help='Brand text overlay')
    args = p.parse_args()

    game = NeuroBloomGame(args)
    game.run()


if __name__ == "__main__":
    main()
