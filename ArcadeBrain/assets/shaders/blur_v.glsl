
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
