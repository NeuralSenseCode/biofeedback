
import numpy as np

class RingBuffer:
    def __init__(self, size: int, channels: int = 1, dtype=np.float32):
        self.size = size
        self.channels = channels
        self.data = np.zeros((size, channels), dtype=dtype)
        self.index = 0
        self.full = False

    def push(self, sample):
        self.data[self.index % self.size] = sample
        self.index += 1
        if self.index >= self.size:
            self.full = True

    def window(self, n):
        if not self.full and self.index < n:
            return self.data[:self.index]
        i = self.index % self.size
        if n <= i:
            return self.data[i-n:i]
        else:
            return np.vstack([self.data[self.size-(n-i):], self.data[:i]])
