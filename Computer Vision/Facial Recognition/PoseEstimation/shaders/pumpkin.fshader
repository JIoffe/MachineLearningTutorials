#version 330 core

in vec2 tex;
in vec3 norm;
in vec3 v_camera;

uniform sampler2D diffuse;
uniform sampler2D bump;

out vec4 frag_color;

void main() {
    //Crude, but essentially encoding emission and perturbation in the same map
    vec4 color = texture(diffuse, tex);
    vec3 bump = texture(bump, tex).xyz;
    
    if(bump.x + bump.y + bump.z == 0.0){
        frag_color = color;
    }else{
        vec3 light_a = -normalize(vec3(0.75, -0.1, -0.5));
        vec3 light_b = -normalize(vec3(-0.75, -0.1, -0.5));
        
        bump.xyz = bump.xyz * 2 - 1;
        vec3 n = normalize(norm + bump * 0.2);
        
        float diffuse = pow(clamp(dot(light_a, n), 0.0, 1.0), 4.0) + pow(clamp(dot(light_b, n), 0.0, 1.0), 2.0);
        float ambience = 0.15;
        float lighting = clamp(diffuse + ambience, 0.0, 1.0);
        
        //Super fake specular
        float spec = 2.0 * pow(max(dot(n, normalize(vec3(-1.0,0.5,1))), 0.0), 32.0)
            + 8.0 * pow(max(dot(n, normalize(vec3(1.0,0.5,1))), 0.0), 64.0);
    
//        float bias = 0.05;
//        float scale = 2;
//        float power = 2;
//        float fresnel = bias + scale * pow(1.0 + dot(normalize(norm), normalize(v_camera)), power);
//        vec4 rim_lighting = vec4(0.2, 0.4, 1, 1) * fresnel;
        
        frag_color = color * lighting + spec;
    }
    //vec3 n = normalize(norm);
    //float a = dot(n, vec3(0.0, 0.0, 1.0));
    //frag_color = vec4(a,a,a,1.0);
    //
    
    //if(bump.x + bump.y + bump.z == 0.0){
    //    frag_color = color;
    //}else{
    //    vec3 light_a = -normalize(vec3(0.75, -0.1, -0.5));
    //    vec3 light_b = -normalize(vec3(-0.75, -0.1, -0.5));
        
    //    vec3 v_view = normalize(v_camera);
    
    //    vec3 n = norm;
        
        
    //    bump.xyz = bump.xyz * 2 - 1;
    //    n += bump * 0.05;
    //    normalize(n);
        
        
        
    //    float diffuse = pow(clamp(dot(light_a, n), 0.0, 1.0), 4.0) + pow(clamp(dot(light_b, n), 0.0, 1.0), 2.0);
    //    float ambience = 0.25;
    //    float lighting = clamp(diffuse + ambience, 0.0, 1.0);
    
    //   float bias = 0.05;
    //    float scale = 2;
    //    float power = 2;
        
    //    float fresnel = bias + scale * pow(1.0 + dot(n, v_view), power);
    //    vec4 rim_lighting = vec4(0.2, 0.4, 1, 1) * fresnel;
        
        
        //Super fake specular
    //    vec3 v_halfway = normalize(light_a + v_view);
    //    //float spec = pow(max(dot(n, v_halfway), 0.0), 64.0);
    //    float spec = 2.0 * pow(max(dot(n, normalize(vec3(-1.0,0.5,1))), 0.0), 32.0)
    //        + 8.0 * pow(max(dot(n, normalize(vec3(1.0,0.5,1))), 0.0), 64.0);
        
    //    frag_color = color * lighting + rim_lighting + spec;
    //}
}