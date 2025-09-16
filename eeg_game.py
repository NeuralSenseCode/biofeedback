SMOOTHING_ALPHA = 0.8  # Default smoothing for EMA
BUFFER_SEC = 5.0       # Default EEG buffer length in seconds

import argparse
import threading
import time
import numpy as np
import pygame
from pylsl import StreamInlet, resolve_streams
from scipy.signal import welch, butter, filtfilt

# --- EEG metric helpers (copy from your viewer) ---
def bandpower(sig, fs, fmin, fmax):
    if len(sig) < 2:
        return 0.0
    from scipy.signal import welch
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

# --- EEG LSL Threaded Buffer ---
class EEGBuffer:
    def __init__(self, fs, ch, window_sec=2):
        self.fs = fs
        self.ch = ch
        self.n = int(window_sec * fs)
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
                return self.buf.copy()
            rolled = np.roll(self.buf, -self.idx, axis=1)
            return rolled.copy()

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

# --- Game constants ---
WIN_W, WIN_H = 800, 600
MAP_TOP = int(WIN_H * 0.125)
MAP_BOT = int(WIN_H * 0.875)
MAP_HEIGHT = MAP_BOT - MAP_TOP
BALL_RADIUS = 20
BALL_X = WIN_W // 2
OBSTACLE_W = 40
OBSTACLE_H = BALL_RADIUS * 2
OBSTACLE_SPEED = 160  # px/sec
FPS = 30

# --- Main game class ---
class EEGGame:
    def __init__(self, metric='concentration', duration=30):
        self.metric = metric
        self.duration = duration
        self.running = True
        self.win = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption('EEG Demo Game')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 36)
        self.reset()
    def reset(self):
        self.t0 = time.time()
        self.ball_y = WIN_H // 2
        self.obstacle_x = WIN_W
        self.obstacle_y = np.random.randint(MAP_TOP, MAP_BOT - OBSTACLE_H)
        self.next_obstacle_time = self.t0 + 5
        self.score = 0
        self.game_over = False
        self.win_msg = ''
    def update_ball(self, metric_val):
        # Map metric [0,1] to y in map
        y = int(MAP_BOT - metric_val * MAP_HEIGHT)
        self.ball_y = np.clip(y, MAP_TOP + BALL_RADIUS, MAP_BOT - BALL_RADIUS)
    def update_obstacle(self, dt):
        self.obstacle_x -= int(OBSTACLE_SPEED * dt)
        if self.obstacle_x + OBSTACLE_W < 0:
            self.obstacle_x = WIN_W
            self.obstacle_y = np.random.randint(MAP_TOP, MAP_BOT - OBSTACLE_H)
            self.score += 1
    def check_collision(self):
        # Ball is always at BALL_X, self.ball_y
        # Obstacle is at self.obstacle_x, self.obstacle_y
        if (self.obstacle_x < BALL_X + BALL_RADIUS < self.obstacle_x + OBSTACLE_W or
            self.obstacle_x < BALL_X - BALL_RADIUS < self.obstacle_x + OBSTACLE_W):
            # Check y overlap
            if (self.obstacle_y < self.ball_y + BALL_RADIUS < self.obstacle_y + OBSTACLE_H or
                self.obstacle_y < self.ball_y - BALL_RADIUS < self.obstacle_y + OBSTACLE_H):
                return True
        return False
    def draw(self):
        self.win.fill((255,255,255))
        # Map boundaries
        pygame.draw.line(self.win, (0,0,0), (0,MAP_TOP), (WIN_W,MAP_TOP), 4)
        pygame.draw.line(self.win, (0,0,0), (0,MAP_BOT), (WIN_W,MAP_BOT), 4)
        # Ball
        pygame.draw.circle(self.win, (0,100,255), (BALL_X, self.ball_y), BALL_RADIUS)
        # Obstacle
        pygame.draw.rect(self.win, (200,0,0), (self.obstacle_x, self.obstacle_y, OBSTACLE_W, OBSTACLE_H))
        # Score/time
        t_rem = max(0, int(self.duration - (time.time() - self.t0)))
        txt = self.font.render(f"Time: {t_rem}s  Score: {self.score}", True, (0,0,0))
        self.win.blit(txt, (20, 20))
        if self.game_over:
            msg = self.font.render(self.win_msg, True, (0,0,0))
            self.win.blit(msg, (WIN_W//2 - msg.get_width()//2, WIN_H//2 - 40))
            msg2 = self.font.render("Press R to restart or Q to quit", True, (0,0,0))
            self.win.blit(msg2, (WIN_W//2 - msg2.get_width()//2, WIN_H//2 + 10))
        pygame.display.flip()
    def run(self, get_metric):
        self.reset()
        while self.running:
            dt = self.clock.tick(FPS) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                if self.game_over:
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_r:
                            self.reset()
                        if event.key == pygame.K_q:
                            self.running = False
            if not self.game_over:
                metric_val = get_metric()
                self.update_ball(metric_val)
                self.update_obstacle(dt)
                if self.check_collision():
                    self.game_over = True
                    self.win_msg = 'Game Over!'
                elif (time.time() - self.t0) > self.duration:
                    self.game_over = True
                    self.win_msg = 'You Win!'
            self.draw()
        pygame.quit()

# --- Main EEG metric extraction ---
def get_metric_func(metric, sbuf, fs, ema):
    def func():
        data = sbuf.snapshot()
        if data.shape[1] < 2:
            return 0.5
        sig = data[1] if data.shape[0] > 1 else data[0]
        a = bandpower(sig, fs, 8, 12)
        b = bandpower(sig, fs, 15, 25)
        th = bandpower(sig, fs, 4, 7)
        hf = bandpower(sig, fs, 20, 35)
        if metric == 'concentration':
            val = b / (a + b + hf + 1e-9)
        else:
            val = th / (th + a + 1e-9)
        val = np.clip(val, 0, 1)
        return ema.update(val)
    return func

# --- Entrypoint ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metric', choices=['concentration','theta'], default='concentration', help='Which metric controls the ball')
    parser.add_argument('--duration', type=int, default=30, help='Game duration in seconds')
    parser.add_argument('--smoothing', type=float, default=SMOOTHING_ALPHA, help='EMA smoothing alpha')
    parser.add_argument('--buffer', type=float, default=BUFFER_SEC, help='EEG buffer in seconds')
    parser.add_argument('--stream-name', default='OpenSignals')
    parser.add_argument('--expect-ch', type=int, default=2)
    args = parser.parse_args()

    # LSL setup
    info = None
    for s in resolve_streams():
        if s.name() == args.stream_name and s.channel_count() == args.expect_ch:
            info = s; break
    if not info:
        print(f"No stream '{args.stream_name}' with {args.expect_ch} channels found.")
        return
    fs = int(info.nominal_srate())
    ch = info.channel_count()
    inlet = StreamInlet(info, max_buflen=60, max_chunklen=0, recover=True)
    sbuf = EEGBuffer(fs=fs, ch=ch, window_sec=args.buffer)
    stop_evt = threading.Event()
    reader_th = threading.Thread(target=reader_loop, args=(inlet, sbuf, stop_evt), daemon=True)
    reader_th.start()
    ema = Ema(alpha=args.smoothing)
    get_metric = get_metric_func(args.metric, sbuf, fs, ema)

    # Start game
    pygame.init()
    game = EEGGame(metric=args.metric, duration=args.duration)
    game.run(get_metric)
    stop_evt.set()
    reader_th.join(timeout=0.5)

if __name__ == '__main__':
    main()
