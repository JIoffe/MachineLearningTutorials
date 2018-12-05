#version 330 core

in float brightness;
out vec4 frag_color;

void main() {
    vec4 inner_glow = vec4(1,1,0,1);
    vec4 outer_glow = vec4(0.6,0.6, 0, 1);
    
    //frag_color = mix(inner_glow, outer_glow, brightness);
    frag_color = inner_glow;
}