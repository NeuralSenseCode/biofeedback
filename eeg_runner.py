import argparse, math, threading, time, sys, random
from collections import deque

import numpy as np
from scipy.signal import welch, butter, filtfilt
import pygame

# LSL is optional (only needed for --mode lsl)
try:
    from pylsl import resolve_stream, StreamInlet
except Exception:
    resolve_stream = None
    StreamInlet = None

###############################################################################
# EEG SOURCE(S)
###############################################################################

class EEGSourceBase:
    """Yields (left, right) EEG samples at ~fs Hz."""
    def start(self): pass
    def stop(self): pass
    def pull(self, n_samples): raise NotImplementedError
    def fs(self): return 250  # default

class LSLEEGSource(EEGSourceBase):
    def __init__(self, stream_type='EEG', left_idx=0, right_idx=1):
        if resolve_stream is None:
            raise RuntimeError("pylsl not installed")
        self.left_idx = left_idx
        self.right_idx = right_idx
        self.inlet = None
        self._fs = 250

    def start(self):
        # Find an EEG LSL stream (from OpenSignals)
        streams = resolve_stream('type', 'EEG')
        if not streams:
            raise RuntimeError("No LSL stream with type=EEG found. "
                               "In OpenSignals, enable 'Tools -> LSL Stream'.")
        self.inlet = StreamInlet(streams[0], max_buflen=10)
        info = self.inlet.info()
        self._fs = int(info.nominal_srate() or 250)

    def fs(self):
        return self._fs

    def pull(self, n_samples):
        """Return np.array shape (n_samples, 2) for (left,right)."""
        out = []
        for _ in range(n_samples):
            sample, ts = self.inlet.pull_sample(timeout=0.0)
            if sample is None:
                break
            try:
                left = float(sample[self.left_idx])
                right = float(sample[self.right_idx])
            except Exception:
                continue
            out.append((left, right))
        if not out:
            # if nothing arrived, return empty
            return np.empty((0,2), dtype=float)
        return np.asarray(out, dtype=float)

class SimEEGSource(EEGSourceBase):
    """Simple simulator that creates alpha-ish rhythms + noise + occasional 'approach' bursts."""
    def __init__(self, fs=250):
        self._fs = fs
        self.t = 0.0
        self.dt = 1.0 / fs
        self.running = False

    def fs(self): return self._fs

    def start(self):
        self.running = True

    def stop(self):
        self.running = False

    def pull(self, n_samples):
        if not self.running:
            return np.empty((0,2))
        out = []
        # Base alpha oscillations with slow drifts and noise
        for _ in range(n_samples):
            # Base alpha ~10 Hz with independent phase & noise
            alpha_freq = 10.0 + 0.5*np.sin(2*np.pi*0.1*self.t)
            phi_L = 2*np.pi*alpha_freq*self.t + 0.2*np.random.randn()
            phi_R = 2*np.pi*alpha_freq*self.t + 0.2*np.random.randn()

            # Occasionally increase R alpha relative to L -> positive FAA (approach)
            burst = 1.0 + 0.8 * (np.sin(2*np.pi*0.02*self.t) > 0)  # ~25s on/off
            # Left and right amplitudes
            A_L = 20e-6 * (1.0 + 0.2*np.sin(2*np.pi*0.03*self.t))  # ~20 µV peak-ish
            A_R = 20e-6 * (1.0 + 0.2*np.cos(2*np.pi*0.035*self.t)) * burst

            left = A_L * np.sin(phi_L) + 5e-6*np.random.randn()
            right = A_R * np.sin(phi_R) + 5e-6*np.random.randn()
            out.append((left, right))
            self.t += self.dt
        return np.asarray(out, dtype=float)

###############################################################################
# FAA COMPUTATION
###############################################################################

class FAAComputer:
    def __init__(self, fs, win_sec=2.0, band=(8.0, 13.0), ema_alpha=0.2):
        self.fs = fs
        self.win = int(win_sec * fs)
        self.band = band
        self.left_buf = deque(maxlen=self.win*3)   # 3x window for Welch segments
        self.right_buf = deque(maxlen=self.win*3)
        self.ema_alpha = ema_alpha
        self.ema = None
        self.baseline_vals = deque(maxlen=fs*30)  # ~30s baseline history
        self.eps = 1e-20

        # Bandpass to reduce drift & high-freq noise (1–45 Hz)
        self.bp_b, self.bp_a = butter(4, [1.0/(fs/2), 45.0/(fs/2)], btype='band')

    def update(self, left_chunk, right_chunk):
        if left_chunk.size == 0: 
            return None, None, None

        # Bandpass
        l = filtfilt(self.bp_b, self.bp_a, left_chunk, method='gust')
        r = filtfilt(self.bp_b, self.bp_a, right_chunk, method='gust')

        self.left_buf.extend(l.tolist())
        self.right_buf.extend(r.tolist())

        if len(self.left_buf) < self.win:
            return None, None, None

        left = np.asarray(self.left_buf, dtype=float)
        right = np.asarray(self.right_buf, dtype=float)

        # Welch PSD
        nperseg = min(int(self.fs*1.0), len(left))  # ~1s segments
        noverlap = int(nperseg*0.5)
        fL, pL = welch(left, fs=self.fs, nperseg=nperseg, noverlap=noverlap)
        fR, pR = welch(right, fs=self.fs, nperseg=nperseg, noverlap=noverlap)

        # Alpha power 8–13 Hz
        idxL = (fL >= self.band[0]) & (fL <= self.band[1])
        idxR = (fR >= self.band[0]) & (fR <= self.band[1])
        p_alpha_L = np.trapz(pL[idxL], fL[idxL]) + self.eps
        p_alpha_R = np.trapz(pR[idxR], fR[idxR]) + self.eps

        faa = math.log(p_alpha_R) - math.log(p_alpha_L)  # ln(R) - ln(L)

        # EMA smoothing
        self.ema = faa if self.ema is None else (1-self.ema_alpha)*self.ema + self.ema_alpha*faa

        # Baseline collection (first ~10s while running)
        self.baseline_vals.append(self.ema)

        # Robust threshold (median + k*MAD)
        baseline = np.median(self.baseline_vals) if len(self.baseline_vals) > 50 else 0.0
        mad = np.median(np.abs(np.asarray(self.baseline_vals) - baseline)) + 1e-9
        thresh = baseline + 1.5*mad  # tweak in-game with +/- keys if you like

        return self.ema, baseline, thresh

###############################################################################
# GAME
###############################################################################

class RunnerGame:
    WIDTH, HEIGHT = 900, 480
    GROUND_Y = 380
    GRAVITY = 0.9
    JUMP_V = -16
    OBSTACLE_SPEED = 6
    OBSTACLE_GAP = 260

    def __init__(self, eeg_source, faa_comp):
        pygame.init()
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("EEG-Runner (FAA → Jump)")
        self.clock = pygame.time.Clock()

        self.eeg = eeg_source
        self.faa = faa_comp

        # Player
        self.x = 100
        self.y = self.GROUND_Y
        self.vy = 0
        self.on_ground = True

        # Obstacles
        self.obstacles = []
        self.spawn_x = self.WIDTH + 100

        # HUD
        self.font = pygame.font.SysFont(None, 22)
        self.running = True
        self.cooldown = 0  # frames

        # Thread to read EEG continuously
        self.left_cache = deque()
        self.right_cache = deque()
        self.reader_thread = threading.Thread(target=self._reader_loop, daemon=True)

    def _reader_loop(self):
        self.eeg.start()
        fs = self.eeg.fs()
        chunk = max(1, int(fs/20))  # ~50 ms
        while self.running:
            arr = self.eeg.pull(chunk)
            if arr.size:
                self.left_cache.extend(arr[:,0].tolist())
                self.right_cache.extend(arr[:,1].tolist())
            else:
                time.sleep(0.01)

    def _pull_for_analysis(self):
        # Pull ~100 ms worth for analysis
        n = min(len(self.left_cache), len(self.right_cache))
        if n <= 0:
            return np.array([]), np.array([])
        n = min(n, max(1, int(self.eeg.fs()/10)))
        L = [self.left_cache.popleft() for _ in range(n)]
        R = [self.right_cache.popleft() for _ in range(n)]
        return np.asarray(L), np.asarray(R)

    def _spawn_obstacle_if_needed(self):
        if not self.obstacles or self.obstacles[-1]['x'] < self.WIDTH - self.OBSTACLE_GAP:
            height = random.choice([40, 60, 80])
            self.obstacles.append({'x': self.spawn_x, 'y': self.GROUND_Y - height, 'w': 30, 'h': height})

    def _move_obstacles(self):
        for o in self.obstacles:
            o['x'] -= self.OBSTACLE_SPEED
        self.obstacles = [o for o in self.obstacles if o['x'] + o['w'] > 0]

    def _draw(self, ema, baseline, thresh):
        self.screen.fill((235, 240, 255))
        # Ground
        pygame.draw.rect(self.screen, (60,60,60), (0, self.GROUND_Y+20, self.WIDTH, 5))
        # Player
        pygame.draw.rect(self.screen, (30,144,255), (self.x, self.y-40, 30, 40), border_radius=6)
        # Obstacles
        for o in self.obstacles:
            pygame.draw.rect(self.screen, (34,139,34), (o['x'], o['y'], o['w'], o['h']), border_radius=4)

        # HUD
        text1 = self.font.render(f"FAA(EMA): {ema:+.3f}", True, (20,20,20)) if ema is not None else self.font.render("FAA: ...", True, (20,20,20))
        text2 = self.font.render(f"Baseline: {baseline:+.3f}  Thresh: {thresh:+.3f}", True, (20,20,20)) if baseline is not None else self.font.render("Calibrating baseline...", True, (20,20,20))
        self.screen.blit(text1, (12, 10))
        self.screen.blit(text2, (12, 35))
        pygame.display.flip()

    def run(self):
        self.reader_thread.start()
        ema = baseline = thresh = None

        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                # Space = manual jump (debug)
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE and self.on_ground:
                    self.vy = self.JUMP_V
                    self.on_ground = False

            # EEG update
            L, R = self._pull_for_analysis()
            ebt = self.faa.update(L, R)
            if ebt is not None:
                ema, baseline, thresh = ebt

            # Decide to jump
            if ema is not None and baseline is not None and self.on_ground and self.cooldown == 0:
                if ema > thresh:
                    self.vy = self.JUMP_V
                    self.on_ground = False
                    self.cooldown = 10  # prevent double-triggers

            if self.cooldown > 0:
                self.cooldown -= 1

            # Physics
            if not self.on_ground:
                self.vy += self.GRAVITY
                self.y += self.vy
                if self.y >= self.GROUND_Y:
                    self.y = self.GROUND_Y
                    self.vy = 0
                    self.on_ground = True

            # Obstacles
            self._spawn_obstacle_if_needed()
            self._move_obstacles()

            # Draw
            self._draw(ema, baseline, thresh)
            self.clock.tick(60)

        self.eeg.stop()
        pygame.quit()

###############################################################################
# MAIN
###############################################################################

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["lsl", "sim"], default="sim")
    ap.add_argument("--left_idx", type=int, default=0, help="LSL channel index for LEFT (F3-ish)")
    ap.add_argument("--right_idx", type=int, default=1, help="LSL channel index for RIGHT (F4-ish)")
    args = ap.parse_args()

    if args.mode == "lsl":
        src = LSLEEGSource(left_idx=args.left_idx, right_idx=args.right_idx)
    else:
        src = SimEEGSource(fs=250)

    fs = src.fs()
    faa = FAAComputer(fs=fs, win_sec=2.0, band=(8,13), ema_alpha=0.25)
    game = RunnerGame(src, faa)
    game.run()

if __name__ == "__main__":
    main()
