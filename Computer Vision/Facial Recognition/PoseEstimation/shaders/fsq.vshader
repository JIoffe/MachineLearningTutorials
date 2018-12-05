#version 330 core

layout (location = 0) in vec2 pos;
out vec2 tex;

void main() {
    float tx = min(max(pos.x, 0.0), 1.0);
    float ty = min(max(-pos.y, 0.0), 1.0);
    
    tex = vec2(tx, ty);

    gl_Position = vec4(pos, 0.0, 1.0);
}