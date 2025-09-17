
import arcade

class PostFX:
    def __init__(self, window):
        self.window = window
        self.ctx = window.ctx
        self.shadow_sigma = 6.0
        # Render targets
        w, h = window.width, window.height
        self.rt_shadow = self.ctx.framebuffer(color_attachments=[self.ctx.texture((w//2, h//2))])
        self.rt_blur_h = self.ctx.framebuffer(color_attachments=[self.ctx.texture((w//2, h//2))])
        self.rt_blur_v = self.ctx.framebuffer(color_attachments=[self.ctx.texture((w//2, h//2))])
        # Shaders
        self.blur_h = self.ctx.load_program(
            vertex_shader=self._vs_fullscreen(),
            fragment_shader=self._fs_blur_h(),
        )
        self.blur_v = self.ctx.load_program(
            vertex_shader=self._vs_fullscreen(),
            fragment_shader=self._fs_blur_v(),
        )
        self.quad = arcade.gl.geometry.quad_2d_fs()

    def begin_shadow_pass(self):
        self.rt_shadow.use()
        self.rt_shadow.clear(0, 0, 0, 0)

        # Downscale drawing for performance
        arcade.set_viewport(0, self.window.width//2, 0, self.window.height//2)

    def end_shadow_pass(self):
        # Restore main viewport
        arcade.set_viewport(0, self.window.width, 0, self.window.height)
        # Blur horizontal
        self.rt_blur_h.use()
        self.rt_blur_h.clear(0, 0, 0, 0)
        self.rt_shadow.color_attachments[0].use(0)
        self.blur_h["u_tex"] = 0
        self.blur_h["u_texel"] = (1.0/self.rt_shadow.color_attachments[0].width, 1.0/self.rt_shadow.color_attachments[0].height)
        self.blur_h["u_sigma"] = self.shadow_sigma
        self.quad.render(self.blur_h)

        # Blur vertical
        self.rt_blur_v.use()
        self.rt_blur_v.clear(0, 0, 0, 0)
        self.rt_blur_h.color_attachments[0].use(0)
        self.blur_v["u_tex"] = 0
        self.blur_v["u_texel"] = (1.0/self.rt_blur_h.color_attachments[0].width, 1.0/self.rt_blur_h.color_attachments[0].height)
        self.blur_v["u_sigma"] = self.shadow_sigma
        self.quad.render(self.blur_v)

        # Back to default framebuffer
        self.window.ctx.screen.use()

    def composite(self):
        # Draw blurred shadows under the world draw (already happened)
        tex = self.rt_blur_v.color_attachments[0]
        tex.use(0)
        program = self.ctx.load_program(
            vertex_shader=self._vs_fullscreen(),
            fragment_shader=self._fs_composite(),
        )
        program["u_tex"] = 0
        arcade.gl.geometry.quad_2d_fs().render(program)

    def _vs_fullscreen(self):
        return """
        #version 330
        in vec2 in_vert;
        in vec2 in_uv;
        out vec2 v_uv;
        void main() {
            v_uv = in_uv;
            gl_Position = vec4(in_vert, 0.0, 1.0);
        }
        """

    def _fs_blur_h(self):
        return """
        #version 330
        uniform sampler2D u_tex;
        uniform vec2  u_texel;
        uniform float u_sigma;
        in vec2 v_uv;
        out vec4 fragColor;
        void main() {
            float sigma = max(u_sigma, 0.0001);
            int radius = int(ceil(3.0 * sigma));
            vec4 c = vec4(0.0);
            float sum = 0.0;
            for (int i = -radius; i <= radius; i++) {
                float w = exp(-(float(i*i)) / (2.0 * sigma * sigma));
                c += texture(u_tex, v_uv + vec2(float(i) * u_texel.x, 0.0)) * w;
                sum += w;
            }
            fragColor = c / sum;
        }
        """

    def _fs_blur_v(self):
        return """
        #version 330
        uniform sampler2D u_tex;
        uniform vec2  u_texel;
        uniform float u_sigma;
        in vec2 v_uv;
        out vec4 fragColor;
        void main() {
            float sigma = max(u_sigma, 0.0001);
            int radius = int(ceil(3.0 * sigma));
            vec4 c = vec4(0.0);
            float sum = 0.0;
            for (int i = -radius; i <= radius; i++) {
                float w = exp(-(float(i*i)) / (2.0 * sigma * sigma));
                c += texture(u_tex, v_uv + vec2(0.0, float(i) * u_texel.y)) * w;
                sum += w;
            }
            fragColor = c / sum;
        }
        """

    def _fs_composite(self):
        return """
        #version 330
        uniform sampler2D u_tex;
        in vec2 v_uv;
        out vec4 fragColor;
        void main() {
            vec4 s = texture(u_tex, v_uv);
            // simple dark shadow composite under existing scene
            fragColor = vec4(s.rgb, 0.6 * s.a);
        }
        """
