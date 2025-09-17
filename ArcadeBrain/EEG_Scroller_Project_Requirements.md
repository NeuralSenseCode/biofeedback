# EEG Scroller Game – Project Requirements (Arcade Approach)

## Overview
This project integrates real-time EEG signals with a polished 2D scroller game built using the **Arcade** Python library. EEG data controls gameplay mechanics (e.g., focus triggers jumps, calm modulates difficulty), while Arcade ensures smooth graphics, animations, and UI.

---

## Tech Stack

- **Programming Language**: Python 3.11+
- **Game Engine**: [Arcade](https://api.arcade.academy/) (OpenGL-powered 2D framework)
- **EEG Interface**: [Lab Streaming Layer (LSL)](https://github.com/sccn/labstreaminglayer) via `pylsl`
- **Signal Processing**: `numpy`, `scipy` (filtering, bandpower, feature extraction)
- **Configuration**: `pydantic`
- **Testing**: `pytest`
- **Dev Tools**: `black`, `ruff`, `mypy`
- **Optional Graphics**: Stable Diffusion / AI-generated textures

---

## Folder Structure

```
eeg-scroller/
├─ pyproject.toml                  # dependencies and configs
├─ README.md                       # project overview
├─ data/
│  └─ test_signals/                # recorded EEG for offline dev
├─ assets/
│  ├─ textures/                    # sprites, background images
│  ├─ fonts/
│  └─ shaders/                     # GLSL shaders (SDF, blur, vignette)
├─ game/
│  ├─ __init__.py
│  ├─ config.py                    # global settings (EEG & rendering)
│  ├─ app.py                       # main Arcade GameWindow
│  ├─ scenes/
│  │  ├─ base_scene.py
│  │  ├─ gameplay_scene.py         # main game (player, parallax, obstacles)
│  │  └─ ui_scene.py               # HUD overlays
│  ├─ render/
│  │  ├─ camera.py                 # world and HUD cameras
│  │  ├─ layers.py                 # parallax backgrounds
│  │  ├─ postfx.py                 # blur, vignette, shadow passes
│  │  └─ shapes.py                 # SDF helpers (rounded rects, capsules)
│  ├─ shaders/
│  │  └─ pipeline.py               # shader manager
│  ├─ assets.py                    # texture/font loader
│  ├─ ecs/
│  │  ├─ components.py             # game entities
│  │  ├─ systems.py                # physics-lite, spawn logic
│  │  └─ events.py                 # jump, calm/focus changes
│  ├─ eeg/
│  │  ├─ lsl_inlet.py              # LSL inlet for EEG signals
│  │  ├─ ringbuffer.py             # lock-free buffer for samples
│  │  ├─ dsp.py                    # filters, bandpower, smoothing
│  │  └─ mapping.py                # EEG → gameplay mapping
│  └─ input/
│     └─ controls.py               # keyboard fallback & debug
└─ main.py                         # entrypoint
```

---

## Game Flow

1. **EEG Acquisition**  
   - EEG sensor → OpenSignals → LSL → Python inlet  
   - Samples stored in ring buffer

2. **Signal Processing**  
   - Bandpass + notch filtering  
   - Compute bandpower (alpha, beta, theta, gamma)  
   - Extract indices:  
     - Focus Index = beta / (alpha + theta)  
     - Calm Index = alpha / (beta + gamma)  

3. **Mapping to Gameplay**  
   - Focus → jump trigger (with hysteresis)  
   - Calm → environment visuals (shadows, parallax, color grading)

4. **Arcade Rendering**  
   - Parallax backgrounds  
   - SDF curved channel with rounded edges  
   - Soft diffuse shadows via blur pass  
   - HUD overlays (bars, text)  
   - Particle effects tied to EEG states

5. **Input System**  
   - EEG as primary control  
   - Keyboard fallback for debug

---

## Graphics Requirements

- Rounded/circular channel edges (SDF shader)
- Diffuse shadows (two-pass blur)
- Parallax background layers (3–5 depth levels)
- HUD cards with rounded edges
- Particle effects responding to EEG indices
- Optional post-processing: vignette, film grain, glow

---

## Performance Targets

- **Frame Rate**: 60–120 FPS
- **EEG Latency**: ≤ 200 ms from brain → action
- **DSP Update**: every 50–100 ms
- **Shadow Buffers**: downsampled for efficiency

---

## Development Workflow

- Offline mode with pre-recorded EEG (`data/test_signals/`)  
- Live mode with headset via LSL  
- Shader hot-reload for fast iteration  
- Config tuning via `config.py`

---

## Future Enhancements

- AI-generated textures and sprites for consistency
- Dynamic difficulty scaling via EEG baseline calibration
- Procedural soundtrack reacting to Calm/Focus indices
- Multiplayer (competitive EEG states)
