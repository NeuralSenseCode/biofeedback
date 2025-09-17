
import arcade
from ..render.layers import Parallax
from ..render.postfx import PostFX
from ..eeg.mapping import EEGState, focus_trigger, fake_eeg_tick, calm_to_shadow_sigma

class Player(arcade.SpriteCircle):
    def __init__(self):
        super().__init__(radius=16, color=arcade.color.CYAN)
        self.center_x, self.center_y = 200, 360
        self.vy = 0.0

    def update(self, dt):
        self.vy -= 2000 * dt  # gravity
        self.center_y += self.vy * dt
        self.center_y = max(32, min(688, self.center_y))

class GameplayScene:
    def __init__(self, window):
        self.window = window
        self.parallax = Parallax(window)
        self.fx = PostFX(window)
        self.player = Player()
        self.eeg = EEGState()
        self.time = 0.0

    def on_update(self, dt: float):
        self.time += dt
        # Fake EEG (replace with real inlet)
        fake_eeg_tick(self.eeg, dt)

        # Trigger jump on focus
        if focus_trigger(self.eeg.focus_idx, self.eeg):
            self.player.vy = 700

        self.player.update(dt)
        self.parallax.update(dt, speed=250)

        # Map calm to shadow softness
        self.fx.shadow_sigma = calm_to_shadow_sigma(self.eeg.calm_idx)

    def on_draw(self):
        # Shadow mask pass
        self.fx.begin_shadow_pass()
        self.player.draw()  # silhouette
        self.fx.end_shadow_pass()

        # World
        arcade.start_render()
        self.parallax.draw()
        self.player.draw()

        # Composite with blurred shadows
        self.fx.composite()

    def on_key_press(self, key, modifiers):
        if key == arcade.key.SPACE:
            self.player.vy = 700
