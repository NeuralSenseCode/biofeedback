
import arcade
from .config import cfg
from .scenes.gameplay_scene import GameplayScene
from .scenes.ui_scene import UIScene

class GameApp(arcade.Window):
    def __init__(self):
        super().__init__(
            width=cfg.render.width,
            height=cfg.render.height,
            title=cfg.render.title,
            antialiasing=True,
        )
        arcade.set_background_color(arcade.color.DARK_SLATE_BLUE)
        self.gameplay = GameplayScene(self)
        self.ui = UIScene(self)
        self.show_fps = True

    def on_update(self, dt: float):
        self.gameplay.on_update(dt)
        self.ui.on_update(dt)

    def on_draw(self):
        self.clear()
        self.gameplay.on_draw()
        self.ui.on_draw()
        if self.show_fps:
            arcade.draw_text(f"{arcade.get_fps():.0f} FPS", 10, self.height - 24, arcade.color.WHITE, 14)

    def on_key_press(self, key, modifiers):
        self.gameplay.on_key_press(key, modifiers)
        self.ui.on_key_press(key, modifiers)
