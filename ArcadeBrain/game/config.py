
from pydantic import BaseModel

class EEGConfig(BaseModel):
    sample_rate: int = 200
    window_ms: int = 1000
    hop_ms: int = 50
    alpha: tuple[int,int] = (8,12)
    beta:  tuple[int,int] = (13,30)
    theta: tuple[int,int] = (4,7)
    gamma: tuple[int,int] = (31,45)
    ewma: float = 0.2
    focus_thresh: float = 1.25
    focus_release: float = 1.05   # hysteresis
    calm_minmax: tuple[float,float] = (0.6, 1.6)

class RenderConfig(BaseModel):
    width: int = 1280
    height: int = 720
    title: str = "EEG Scroller"
    base_palette: str = "teal_orange"
    vignette_strength: float = 0.18
    grain_amount: float = 0.02
    shadow_sigma_px: tuple[float,float] = (2.0, 10.0)  # mapped by calm

class Config(BaseModel):
    eeg: EEGConfig = EEGConfig()
    render: RenderConfig = RenderConfig()

cfg = Config()
