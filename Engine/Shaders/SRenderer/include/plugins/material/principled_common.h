#ifndef _SRENDERER_COMMON_PRINCIPLED_HEADER_
#define _SRENDERER_COMMON_PRINCIPLED_HEADER_

struct PrincipledMaterialData {
  vec4 albedo_tint;                 // 4
  vec2 uv_tiling;                   // 6
  vec2 uv_scaling;                  // 8
  uint mat_type;                    // 9
  uint basecolor_opacity_tex;       // 10
  uint normal_bump_tex;             // 11
  uint roughness_metalic_ao_tex;    // 12
  uint padding0;                    // 13
  uint padding1;                    // 14
  uint padding2;                    // 15
  uint padding3;                    // 16
};

layout(binding = 4, set = 0, scalar) buffer _PrincipledMaterialBuffer   { PrincipledMaterialData principled_materials[]; };

#endif