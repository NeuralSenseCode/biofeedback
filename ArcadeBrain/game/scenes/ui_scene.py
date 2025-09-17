
import arcade
from ..eeg.mapping import EEGState

class UIScene:
    def __init__(self, window):
        self.window = window
        self.eeg = EEGState.shared  # reference to same state

    def on_update(self, dt: float):
        pass

    def on_draw(self):
        # simple HUD
        w, h = self.window.width, self.window.height
        arcade.draw_rectangle_filled(w-150, h-40, 280, 60, (0,0,0,140))
        arcade.draw_text(f"Focus: {self.eeg.focus_idx:.2f}", w-270, h-56, arcade.color.WHITE, 14)
        arcade.draw_text(f"Calm:  {self.eeg.calm_idx:.2f}", w-270, h-36, arcade.color.WHITE, 14)

    def on_key_press(self, key, modifiers):
        pass
