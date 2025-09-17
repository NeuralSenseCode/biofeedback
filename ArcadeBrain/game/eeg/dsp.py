
import numpy as np
from scipy.signal import butter, sosfiltfilt, welch

def bandpass_sos(low, high, fs, order=4):
    return butter(order, [low, high], btype="band", fs=fs, output="sos")

def filtfilt_sos(sos, x):
    if x.shape[0] < 8 * sos.shape[0]:
        return x  # too short to filter safely
    return sosfiltfilt(sos, x, axis=0)

def bandpower(x, fs, fmin, fmax):
    f, Pxx = welch(x, fs=fs, nperseg=min(256, len(x)))
    band = (f >= fmin) & (f <= fmax)
    return np.trapz(Pxx[..., band], f[band], axis=-1)
