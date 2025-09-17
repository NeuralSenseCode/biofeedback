
import arcade

class Parallax:
    def __init__(self, window):
        self.window = window
        self.layers = [
            (arcade.ShapeElementList(), 0.2, (30, 30, 60)),
            (arcade.ShapeElementList(), 0.5, (40, 40, 80)),
            (arcade.ShapeElementList(), 1.0, (50, 50, 100)),
        ]
        w, h = window.width, window.height
        # Create big soft rectangles as background blobs
        for i, (shape_list, _, col) in enumerate(self.layers):
            for j in range(6):
                x = j * (w/3) + (i*40)
                y = (j%3)* (h/3) + 100
                shape_list.append(arcade.create_rectangle_filled(x, y, 400, 220, (*col, 180)))

        self.offset = 0.0

    def update(self, dt: float, speed: float = 200.0):
        self.offset += speed * dt

    def draw(self):
        w = self.window.width
        for shape_list, speed, _ in self.layers:
            with arcade.render_target(0, 0):  # no-op, keep API simple
                arcade.push_viewport()
                arcade.set_viewport(- (self.offset * speed) % w, w - (self.offset * speed) % w, 0, self.window.height)
                shape_list.draw()
                arcade.pop_viewport()
