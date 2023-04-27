#version 460

#extension GL_EXT_scalar_block_layout : require

struct PushConstants { 
    mat4  projection_view;
    ivec2 resolution;
};
layout(push_constant) uniform PushConsts { PushConstants pushConstants; };

struct InterleavedVertex {
    vec3 position;
};
struct LineProperty {
    vec3    color;
    float   width;
};
layout(binding = 0, set = 0, scalar) buffer _VerticesBuffer   { InterleavedVertex vertices[]; };
layout(binding = 1, set = 0, scalar) buffer _LinePropBuffer   { LineProperty lines[]; };

vec3 quad[30] = vec3[](
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

layout(location = 0) out vec3 color;

void main() {
    const int vertexID = gl_VertexIndex;
    const int instanceID = gl_InstanceIndex;

    const vec2 resolution = vec2(pushConstants.resolution);

    const vec3 position = quad[vertexID];
    const LineProperty prop = lines[gl_InstanceIndex];

    // get vertex positions
    const vec4 clip0 = pushConstants.projection_view * vec4(vertices[instanceID*2 + 0].position, 1);
    const vec4 clip1 = pushConstants.projection_view * vec4(vertices[instanceID*2 + 1].position, 1);

    const vec2 screen0 = resolution * (0.5 * clip0.xy/clip0.w + 0.5);
    const vec2 screen1 = resolution * (0.5 * clip1.xy/clip1.w + 0.5);

    vec2 xBasis = normalize(screen1 - screen0);
    vec2 yBasis = vec2(-xBasis.y, xBasis.x);
    vec2 pt0 = screen0 + prop.width * (position.x * xBasis + position.y * yBasis);
    vec2 pt1 = screen1 + prop.width * (position.x * xBasis + position.y * yBasis);
    vec2 pt = mix(pt0, pt1, position.z);
    vec4 clip = mix(clip0, clip1, position.z);

    color = prop.color;
    gl_Position = vec4(clip.w * (2.0 * pt/resolution - 1.0), clip.z, clip.w);
}