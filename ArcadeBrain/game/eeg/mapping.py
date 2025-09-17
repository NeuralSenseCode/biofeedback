
import numpy as np
from dataclasses import dataclass

@dataclass
class EEGState:
    focus_idx: float = 1.0
    calm_idx: float = 1.0
    focus_active: bool = False

# simple global/shared state for demo
EEGState.shared = EEGState()

def focus_trigger(focus_idx: float, state: EEGState) -> bool:
    if not state.focus_active and focus_idx > 1.25:
        state.focus_active = True
        return True
    if state.focus_active and focus_idx < 1.05:
        state.focus_active = False
    return False

def calm_to_shadow_sigma(calm_idx: float) -> float:
    cmin, cmax = 0.6, 1.6
    lo, hi = 2.0, 10.0
    t = (np.clip(calm_idx, cmin, cmax) - cmin) / (cmax - cmin + 1e-9)
    return lo + t * (hi - lo)

# Fake EEG generator for demo
def fake_eeg_tick(state: EEGState, dt: float):
    import math
    # oscillate indices for visualization
    t = fake_eeg_tick.t = getattr(fake_eeg_tick, "t", 0.0) + dt
    state.focus_idx = 1.1 + 0.3 * (0.5 + 0.5 * math.sin(2*math.pi*0.5*t))
    state.calm_idx  = 1.0 + 0.4 * (0.5 + 0.5 * math.sin(2*math.pi*0.2*t + 1.0))
    EEGState.shared.focus_idx = state.focus_idx
    EEGState.shared.calm_idx = state.calm_idx
