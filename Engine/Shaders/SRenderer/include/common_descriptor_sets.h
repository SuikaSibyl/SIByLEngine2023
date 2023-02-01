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
  float surfaceArea;
  uint lightID;
  uint primitiveType;
  float oddNegativeScaling;
  vec4 transform[3];
  vec4 transformInverse[3];
};
// material info
struct MaterialData {
  vec4 data_pack_0;
  vec4 data_pack_1;
  vec4 data_pack_2;
  vec3 data_pack_3;
  uint bsdf_type;
};
// light info
struct LightData {
  // 0: diffuse area light - sphere
  // 1: diffuse area light - triangle mesh
  // 2: env map
  uint	lightType;
  vec3	intensity;
  uint	index;						// geometry index (type 0/1) or texture index (type 2)
  uint	sample_dist_size_0;			// sample distribution unit size
  uint	sample_dist_offset_pmf_0;	// sample distribution offset for pmf start
  uint	sample_dist_offset_cdf_0;	// sample distribution offset for cdf start
  float pmf;
  uint	sample_dist_size_1;			// (another dim of) sample distribution unit size
  uint	sample_dist_offset_pmf_1;	// (another dim of) sample distribution offset for pmf start
  uint	sample_dist_offset_cdf_1;	// (another dim of) sample distribution offset for cdf start
};
// scene info uniforms
struct SceneInfoUniforms {
  uint  light_num;
  uint  light_offset_pmf;
  uint  light_offset_cdf;
};
// binding definition
layout(binding = 0, set = 0, scalar) uniform _GlobalUniforms  { GlobalUniforms globalUniform; };
layout(binding = 1, set = 0, scalar) buffer _VerticesBuffer   { InterleavedVertex vertices[]; };
layout(binding = 2, set = 0, scalar) buffer _IndicesBuffer    { uint indices[]; };
layout(binding = 3, set = 0, scalar) buffer _GeometryBuffer   { GeometryInfo geometryInfos[]; };
layout(binding = 4, set = 0, scalar) buffer _MaterialBuffer   { MaterialData materials[]; };
layout(binding = 5, set = 0, scalar) buffer _LightBuffer      { LightData lights[]; };
layout(binding = 6, set = 0, scalar) buffer _SampleDistBuffer { float sampleDistDatas[]; };
layout(binding = 7, set = 0, scalar) uniform _SceneInfoBuffer { SceneInfoUniforms sceneInfoUniform; };
layout(binding = 8, set = 0) uniform sampler2D textures[];

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

int upper_bound(int offset, int size, float param) {
  int i=offset;
  for(; i<offset+size; ++i) {
    if(sampleDistDatas[i] > param)
      break;
  }
  return i;
}

int sampleTableDist1D(in int cdf_offset, in int pmf_size, in float rnd_param) {
  const int find = upper_bound(cdf_offset, pmf_size + 1, rnd_param);
  int offset = clamp(find-1, cdf_offset, cdf_offset + pmf_size - 1);
  return offset - cdf_offset;
}

int sampleOneLight(in float x) {
  return sampleTableDist1D(
    int(sceneInfoUniform.light_offset_cdf),
    int(sceneInfoUniform.light_num),
    x);
}

#endif