#version 460
#extension GL_EXT_scalar_block_layout : require
#extension GL_GOOGLE_include_directive : enable
#include "particle_draw.glsl"

vec3 circle[30] = vec3[](
    // triangle 0
    vec3(0,-0.5,0),
    vec3(0,-0.5,1),
    vec3(0,0.5,1),
    // triangle 1
    vec3(0,-0.5,0),
    vec3(0,0.5,1),
    vec3(0,0.5,0),
    // triangle 2
    vec3(0,0,0),
    vec3(3.061616997868383e-17,0.5,0),
    vec3(-0.35355339059327373,0.3535533905932738,0),
    // triangle 3
    vec3(0,0,0),
    vec3(-0.35355339059327373,0.3535533905932738,0),
    vec3(-0.5,6.123233995736766e-17,0),
    // triangle 4
    vec3(0,0,0),
    vec3(-0.5,6.123233995736766e-17,0),
    vec3(-0.35355339059327384,-0.35355339059327373,0),
    // triangle 5
    vec3(0,0,0),
    vec3(-0.35355339059327384,-0.35355339059327373,0),
    vec3(-9.184850993605148e-17,-0.5,0),
    // triangle 6
    vec3(0,0,1),
    vec3(-9.184850993605148e-17,-0.5,1),
    vec3(0.3535533905932737,-0.35355339059327384,1),
    // triangle 7
    vec3(0,0,1),
    vec3(0.3535533905932737,-0.35355339059327384,1),
    vec3(0.5,-1.2246467991473532e-16,1),
    // triangle 8
    vec3(0,0,1),
    vec3(0.5,-1.2246467991473532e-16,1),
    vec3(0.35355339059327384,0.3535533905932737,1),
    // triangle 9
    vec3(0,0,1),
    vec3(0.35355339059327384,0.3535533905932737,1),
    vec3(1.5308084989341916e-16,0.5,1)
);

layout(location = 0) out vec3 col;

void main() {
    vec3 vertex = vec3(circle[gl_VertexIndex].xy, 0);
    vec2 position = positions[gl_InstanceIndex];
    
    vertex = vertex * pushConstants.scale + vec3(position, 0);
    col = colors[gl_InstanceIndex];
    gl_Position = gCamera.viewProjMat * vec4(vertex, 1);
}