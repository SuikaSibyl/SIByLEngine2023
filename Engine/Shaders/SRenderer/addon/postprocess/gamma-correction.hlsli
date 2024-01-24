#ifndef _SRENDERER_ADDON_GAMMA_CORRECTION_HEADER_
#define _SRENDERER_ADDON_GAMMA_CORRECTION_HEADER_

#include "../../include/common/cpp_compatible.hlsli"

// static const float GAMMA = 1.2f;
static const float GAMMA = 2.2f;

float3 gamma(in_ref(float3) color, float g) {
    return pow(color, float3(g));
}

float3 LinearToScreen(in_ref(float3) linearRGB) {
    return gamma(linearRGB, 1.0 / GAMMA);
}

float3 ScreenToLinear(in_ref(float3) screenRGB) {
    return gamma(screenRGB, GAMMA);
}

#endif // !_SRENDERER_ADDON_GAMMA_CORRECTION_HEADER_