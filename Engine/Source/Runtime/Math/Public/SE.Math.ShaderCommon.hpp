#pragma once
#include <algorithm>
#include "SE.Math.Geometric.hpp"

#define in 

namespace SIByL::Math {
using float2 = vec2;
using float3 = vec3;
using float4 = vec4;

using int2 = ivec2;
using int3 = ivec3;
using int4 = ivec4;

using uint2 = uvec2;
using uint3 = uvec3;
using uint4 = uvec4;

using bool2 = bvec2;
using bool3 = bvec3;
using bool4 = bvec4;

using uint = uint32_t;

using float3x3 = mat3;
using float4x4 = mat4;

inline auto firstbithigh(uint32_t value) noexcept -> uint32_t {
  return 31 - std::countl_zero(value);
}

using std::min;
using std::max;

//inline float log2(float x) { return std::log2(x); }

inline float clamp(float x, float vmin, float vmax) {
  return std::max(vmin, std::min(vmax, x));
}
inline float2 clamp(float2 v, float vmin, float vmax) {
  return float2(clamp(v.x, vmin, vmax), clamp(v.y, vmin, vmax));
}
inline float3 clamp(float3 v, float vmin, float vmax) {
  return float3(clamp(v.x, vmin, vmax), clamp(v.y, vmin, vmax),
                clamp(v.z, vmin, vmax));
}
inline float4 clamp(float4 v, float vmin, float vmax) {
  return float4(clamp(v.x, vmin, vmax), clamp(v.y, vmin, vmax),
                clamp(v.z, vmin, vmax), clamp(v.w, vmin, vmax));
}

inline bool2 isnan(float2 v) { return bool2(std::isnan(v.x), std::isnan(v.y)); }

using std::pow;
inline float4 pow(float4 a, float4 b) {
  return float4(pow(a.x, b.x), pow(a.y, b.y), pow(a.z, b.z), pow(a.w, b.w));
}

inline float2 select(bool2 pred, float2 v1, float2 v2) {
  return float2(pred.data[0] ? v1.data[0] : v2.data[0],
                pred.data[1] ? v1.data[1] : v2.data[1]);
}
inline float3 select(bool3 pred, float3 v1, float3 v2) {
  return float3(pred.data[0] ? v1.data[0] : v2.data[0],
                pred.data[1] ? v1.data[1] : v2.data[1],
                pred.data[2] ? v1.data[2] : v2.data[2]);
}
inline float4 select(bool4 pred, float4 v1, float4 v2) {
  return float4(pred.data[0] ? v1.data[0] : v2.data[0],
                pred.data[1] ? v1.data[1] : v2.data[1],
                pred.data[2] ? v1.data[2] : v2.data[2],
                pred.data[3] ? v1.data[3] : v2.data[3]);
}

inline float saturate(float v) { return std::max(0.f, std::min(1.f, v)); }
inline vec2 saturate(vec2 const& v) {
  return vec2{saturate(v.x), saturate(v.y)};
}
inline vec3 saturate(vec3 const& v) {
  return vec3{saturate(v.x), saturate(v.y), saturate(v.z)};
}
inline vec4 saturate(vec4 const& v) {
  return vec4{saturate(v.x), saturate(v.y), saturate(v.z), saturate(v.w)};
}

inline bool any(bool2 const& v) { return v.x || v.y; }
inline bool any(bool3 const& v) { return v.x || v.y || v.z; }
inline bool any(bool4 const& v) { return v.x || v.y || v.z || v.w; }

inline uint2 operator&(uint2 v, uint x) { return uint2(v.x & x, v.y & x); }
inline uint2 operator>>(uint2 v, uint x) { return uint2(v.x >> x, v.y >> x); }
inline int2 operator>>(int2 v, uint x) { return int2(v.x >> x, v.y >> x); }

inline float2 ternary(bool2 cond, float2 a, float2 b) {
  return float2{cond.x ? a.x : b.x, cond.y ? a.y : b.y};
}

inline uint32_t f32tof16(float v) { return 0; } // todo
inline float f16tof32(uint32_t v) { return 0; }  // todo

inline float round(float v) { return std::round(v); }
inline float2 round(float2 v) { return float2(round(v.x), round(v.y)); }
inline float3 round(float3 v) {
  return float3(round(v.x), round(v.y), round(v.z));
}
inline float4 round(float4 v) {
  return float4(round(v.x), round(v.y), round(v.z), round(v.w));
}
}

using namespace SIByL::Math;