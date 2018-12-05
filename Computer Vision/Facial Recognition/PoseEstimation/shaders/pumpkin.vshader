#version 330 core

layout (location = 0) in vec3 pos;
layout (location = 1) in vec2 v_tex;
layout (location = 2) in vec3 v_norm;

uniform mat4 mvp;
uniform mat4 mv;
uniform mat3 normal_matrix;

out vec2 tex;
out vec3 norm;
out vec3 v_camera;

void main() {
    vec4 v_pos = vec4(pos, 1.0);
    
    tex = v_tex;
    norm = normalize(normal_matrix * v_norm);
    
    v_camera = normalize(-(mv * v_pos).xyz - vec3(0,0,0));
    gl_Position = mvp * v_pos;
}