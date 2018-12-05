#version 330 core

in vec2 tex;
uniform sampler2D diffuse;
out vec4 frag_color;

void main() {
    frag_color = texture(diffuse, tex);
}