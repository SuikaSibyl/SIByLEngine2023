#ifndef _SRENDERER_COMMON_GBUFFER_DEF_HEADER_
#define _SRENDERER_COMMON_GBUFFER_DEF_HEADER_

#extension GL_EXT_shader_image_load_formatted : enable

#include "../include/definitions/camera.h"
#include "../include/common/common_octahedral.h"
#include "../include/common/common_microfacet.h"
#include "../include/common/common_spectrum.h"

const float k_background_depth = 65504.f;

// A surface with enough information to evaluate BRDFs
struct GBufferSurface {
    vec3  worldPos;
    vec3  viewDir;
    float viewDepth;
    vec3  normal;
    vec3  geoNormal;
    vec3  diffuseAlbedo;
    vec3  specularF0;
    float roughness;
    float diffuseProbability;
};
// Checks if the given surface is valid.
bool IsSurfaceValid(GBufferSurface surface) {
    // is the surface has background depth, it's invalid
    return surface.viewDepth != k_background_depth;
}
// Create an empty surface with invalid data.
GBufferSurface EmptyGBufferSurface() {
    GBufferSurface surface;
    surface.worldPos = vec3(0.f); surface.viewDir = vec3(0.f);
    surface.normal = vec3(0.f); surface.geoNormal = vec3(0.f);
    surface.diffuseAlbedo = vec3(0.f); surface.specularF0 = vec3(0.f);
    surface.roughness = 0.f; surface.diffuseProbability = 0.f; surface.viewDepth = k_background_depth;
    return surface;
}

vec3 ViewDepthToWorldPos(
    in const PlanarViewData view,
    in const ivec2 pixelPosition,
    in const float viewDepth
) {
    const vec2 uv = (vec2(pixelPosition) + 0.5) * view.viewportSizeInv;
    const vec4 clipPos = vec4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.5, 1);
    vec4 viewPos = view.matClipToView * clipPos;
    viewPos.xy /= viewPos.z; viewPos.zw = vec2(1.0); viewPos.xyz *= viewDepth;
    return (view.matViewToWorld * viewPos).xyz;
}

float getSurfaceDiffuseProbability(GBufferSurface surface) {
    float diffuseWeight = luminance(surface.diffuseAlbedo);
    SchlickFresnel(0., 1.);
    float specularWeight = luminance(SchlickFresnel(surface.specularF0, dot(surface.viewDir, surface.normal)));
    float sumWeights = diffuseWeight + specularWeight;
    return sumWeights < 1e-7f ? 1.f : (diffuseWeight / sumWeights);
}

GBufferSurface GetGBufferSurface(
    in const ivec2 pixelPosition, 
    in const PlanarViewData view, 
    sampler2D  depthTexture, 
    usampler2D normalsTexture, 
    usampler2D geoNormalsTexture, 
    sampler2D  diffuseAlbedoTexture, 
    sampler2D  specularRoughTexture
) {
    GBufferSurface surface = EmptyGBufferSurface();
    if (any(greaterThanEqual(pixelPosition, view.viewportSize))) return surface;
    surface.viewDepth = texelFetch(depthTexture, pixelPosition, 0).r;
    if(surface.viewDepth == k_background_depth) return surface;
    surface.normal = Unorm32OctahedronToUnitVector(texelFetch(normalsTexture, pixelPosition, 0).r);
    surface.geoNormal = Unorm32OctahedronToUnitVector(texelFetch(geoNormalsTexture, pixelPosition, 0).r);
    surface.diffuseAlbedo = texelFetch(diffuseAlbedoTexture, pixelPosition, 0).rgb;
    const vec4 specularRough = texelFetch(specularRoughTexture, pixelPosition, 0).rgba;
    surface.specularF0 = specularRough.rgb;
    surface.roughness = specularRough.a;
    surface.worldPos = ViewDepthToWorldPos(view, pixelPosition, surface.viewDepth);
    surface.viewDir = normalize(view.cameraDirectionOrPosition.xyz - surface.worldPos);
    surface.diffuseProbability = getSurfaceDiffuseProbability(surface);
    return surface;
}

#endif // _SRENDERER_COMMON_GBUFFER_DEF_HEADER_