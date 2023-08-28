#ifndef _SRENDERER_SHADING_HEADER_
#define _SRENDERER_SHADING_HEADER_

#include "cpp_compatible.hlsli"
#include "math.hlsli"
#include "microfacet.hlsli"

static float k_background_depth = 65504.f;

struct ShadingSurface {
    float3 worldPos;
    float roughness;
    float3 viewDir;
    float viewDepth;
    float3 shadingNormal;
    float transmissionFactor;
    float3 geometryNormal;
    uint bsdfID;
    float3 diffuseAlbedo;
    float3 specularF0;
};

ShadingSurface EmptyShadingSurface() {
    ShadingSurface surface;
    surface.viewDepth = k_background_depth;
    return surface;
}

float GetShadingSurfaceLinearDepth(in_ref(ShadingSurface) surface) {
    return surface.viewDepth;
}
float3 GetShadingSurfaceShadingNormal(in_ref(ShadingSurface) surface) {
    return surface.shadingNormal;
}
float3 GetShadingSurfaceWorldPos(in_ref(ShadingSurface) surface) {
    return surface.worldPos;
}
bool IsShadingSurfaceValid(in_ref(ShadingSurface) surface) {
    return surface.viewDepth != k_background_depth;
}

#endif // !_SRENDERER_SHADING_HEADER_