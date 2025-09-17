
#version 330
uniform vec2  u_resolution;
uniform vec2  u_center;       // px
uniform vec2  u_size;         // px (w,h)
uniform float u_radius;       // px
uniform float u_edge_soft;    // px
uniform vec4  u_fill;         // rgba
out vec4 fragColor;

float sdRoundRect(vec2 p, vec2 b, float r) {
    vec2 d = abs(p) - b + vec2(r);
    return length(max(d,0.0)) - r;
}

void main() {
    vec2 uv = gl_FragCoord.xy;
    vec2 p  = uv - u_center;
    vec2 halfs = 0.5 * u_size;
    float d = sdRoundRect(p, halfs, u_radius);
    float alpha = 1.0 - smoothstep(0.0, u_edge_soft, d);
    fragColor = vec4(u_fill.rgb, u_fill.a * alpha);
}
