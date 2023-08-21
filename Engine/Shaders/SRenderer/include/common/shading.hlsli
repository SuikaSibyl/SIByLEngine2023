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
    float3 normal;
    float diffuseProbability;
    float3 geoNormal;
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
float3 GetShadingSurfaceNormal(in_ref(ShadingSurface) surface) {
    return surface.normal;
}
float3 GetShadingSurfaceWorldPos(in_ref(ShadingSurface) surface) {
    return surface.worldPos;
}
bool IsShadingSurfaceValid(in_ref(ShadingSurface) surface) {
    return surface.viewDepth != k_background_depth;
}

// Get the probability of choosing diffuse BRDF
float getSurfaceDiffuseProbability(in_ref(ShadingSurface) surface) {
    float diffuseWeight = luminance(surface.diffuseAlbedo);
    float specularWeight = luminance(SchlickFresnel(surface.specularF0, dot(surface.viewDir, surface.normal)));
    float sumWeights = diffuseWeight + specularWeight;
    return sumWeights < 1e-7f ? 1.f : (diffuseWeight / sumWeights);
}

#endif // !_SRENDERER_SHADING_HEADER_