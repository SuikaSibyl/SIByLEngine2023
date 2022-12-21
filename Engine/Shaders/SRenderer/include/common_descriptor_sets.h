#ifndef _SRENDERER_COMMON_DESCRIPTOR_SET_HEADER_
#define _SRENDERER_COMMON_DESCRIPTOR_SET_HEADER_

#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

/**
* Descriptor Set 0
* --------------------------
* binding 0: global uniforms
* binding 1: bindless textures
* binding 2: vertex buffer
* binding 3: index buffer
* binding 4: geometry buffer
*/
// Global uniforms carrying camera information.
struct GlobalUniforms {
  mat4 view;
  mat4 proj;
  mat4 viewInverse;  // Camera inverse view matrix
  mat4 projInverse;  // Camera inverse projection matrix
};
// interleaved vertex layout
struct InterleavedVertex {
  vec3 position;
  vec3 normal;
  vec3 tangent;
  vec2 texCoords;
};
// geometry info
struct GeometryInfo {
  uint vertexOffset;
  uint indexOffset;
  uint materialID;
  uint padding;
  vec4 transform[3];
};
// binding definition
layout(binding = 0, set = 0, scalar) uniform _GlobalUniforms { GlobalUniforms globalUniform; };
layout(binding = 1, set = 0, scalar) buffer _VerticesBuffer { InterleavedVertex vertices[]; };
layout(binding = 2, set = 0, scalar) buffer _IndicesBuffer  { uint indices[]; };
layout(binding = 3, set = 0, scalar) buffer _GeometryBuffer { GeometryInfo geometryInfos[]; };
layout(binding = 4, set = 0) uniform sampler2D textures[];

#endif