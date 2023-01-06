#ifndef _SRENDERER_COMMON_DESCRIPTOR_SET_HEADER_
#define _SRENDERER_COMMON_DESCRIPTOR_SET_HEADER_

#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_nonuniform_qualifier : enable

/**
* Descriptor Set 0
* --------------------------
* binding 0: global uniforms
* binding 1: vertex buffer
* binding 2: index buffer
* binding 3: geometry buffer
* binding 4: material buffer
* binding 5: bindless textures
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
  uint indexSize;
  uint padding0;
  uint padding1;
  uint padding2;
  float oddNegativeScaling;
  vec4 transform[3];
  vec4 transformInverse[3];
};
// material info
struct MaterialData {
  vec4 albedo_tint;
  vec2 uv_tiling;
  vec2 uv_scaling;
  uint mat_type;
  uint basecolor_opacity_tex;
  uint normal_bump_tex;
  uint roughness_metalic_ao_tex;
};
// binding definition
layout(binding = 0, set = 0, scalar) uniform _GlobalUniforms { GlobalUniforms globalUniform; };
layout(binding = 1, set = 0, scalar) buffer _VerticesBuffer { InterleavedVertex vertices[]; };
layout(binding = 2, set = 0, scalar) buffer _IndicesBuffer  { uint indices[]; };
layout(binding = 3, set = 0, scalar) buffer _GeometryBuffer { GeometryInfo geometryInfos[]; };
layout(binding = 4, set = 0, scalar) buffer _MaterialBuffer { MaterialData materials[]; };
layout(binding = 5, set = 0) uniform sampler2D textures[];


// Utilities

mat4 ObjectToWorld(in GeometryInfo geometry) {
  return transpose(mat4(geometry.transform[0], geometry.transform[1], geometry.transform[2], vec4(0,0,0,1)));
}

mat4 WorldToObject(in GeometryInfo geometry) {
  return transpose(mat4(geometry.transformInverse[0], geometry.transformInverse[1], geometry.transformInverse[2], vec4(0,0,0,1)));
}

mat4 ObjectToWorldNormal(in GeometryInfo geometry) {
  return mat4(geometry.transformInverse[0], geometry.transformInverse[1], geometry.transformInverse[2], vec4(0,0,0,1));
}

#endif