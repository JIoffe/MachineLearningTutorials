#version 330 core

layout (location = 0) in vec3 pos;
layout (location = 1) in vec2 v_tex;

uniform mat4 mvp;

out float brightness;

void main() {
    brightness = v_tex.y;
    gl_Position = mvp * vec4(pos, 1.0);
}