#ifndef _SRENDERER_ADDON_POSTPROCESSING_HEADER_
#define _SRENDERER_ADDON_POSTPROCESSING_HEADER_

#include "../../include/common/cpp_compatible.hlsli"

float3 ACESToneMapping(in_ref(float3) color) {
    // Cancel out the pre-exposure mentioned in
    // https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
    color *= 0.6;
    const float A = 2.51f;
    const float B = 0.03f;
    const float C = 2.43f;
    const float D = 0.59f;
    const float E = 0.14f;
    return saturate((color * (A * color + B)) / (color * (C * color + D) + E));
}

#endif // !_SRENDERER_ADDON_POSTPROCESSING_HEADER_