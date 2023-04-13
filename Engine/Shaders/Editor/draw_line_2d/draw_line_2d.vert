#version 460

#extension GL_EXT_scalar_block_layout : require

struct PushConstants { 
    mat4 projection;
};
layout(push_constant) uniform PushConsts { PushConstants pushConstants; };

struct InterleavedVertex {
    vec2 position;
};
struct LineProperty {
    vec3    color;
    float   width;
};
layout(binding = 0, set = 0, scalar) buffer _VerticesBuffer   { InterleavedVertex vertices[]; };
layout(binding = 1, set = 0, scalar) buffer _LinePropBuffer   { LineProperty lines[]; };

vec2 quad[6] = vec2[](
  vec2(0, -0.5),
  vec2(1, -0.5),
  vec2(1,  0.5),
  vec2(0, -0.5),
  vec2(1,  0.5),
  vec2(0,  0.5)
);

layout(location = 0) out vec3 color;

void main() {
    const int vertexID = gl_VertexIndex;
    const int instanceID = gl_InstanceIndex;

    const vec2 position = quad[vertexID];
    const LineProperty prop = lines[gl_InstanceIndex];

    // get vertex positions
    const vec2 pointA = vertices[instanceID*2 + 0].position;
    const vec2 pointB = vertices[instanceID*2 + 1].position;
    // get basis for the line
    vec2 xBasis = pointB - pointA;
    vec2 yBasis = normalize(vec2(-xBasis.y, xBasis.x));
    vec2 point = pointA + xBasis * position.x + yBasis * prop.width * position.y;
    
    color = prop.color;
    gl_Position = pushConstants.projection * vec4(point, 0, 1);
}