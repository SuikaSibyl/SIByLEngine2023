#ifndef _SRENDERER_COMMON_LAMBERTIAN_HEADER_
#define _SRENDERER_COMMON_LAMBERTIAN_HEADER_

struct LambertianMaterialData {
  vec4 albedo_tint;                 // 4
  vec2 uv_tiling;                   // 6
  vec2 uv_scaling;                  // 8
  uint mat_type;                    // 9
  uint basecolor_opacity_tex;       // 10
  uint normal_bump_tex;             // 11
  uint padding0;                    // 12
  vec4 padding1;                    // 13
};

layout(binding = 4, set = 0, scalar) buffer _LambertianMaterialBuffer   { LambertianMaterialData lambertian_materials[]; };

#endif