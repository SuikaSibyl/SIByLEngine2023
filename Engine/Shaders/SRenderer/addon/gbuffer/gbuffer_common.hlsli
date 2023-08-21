#ifndef _SRENDERER_GBUFFER_ADDON_COMMON_HEADER_
#define _SRENDERER_GBUFFER_ADDON_COMMON_HEADER_

#include "../../include/common/math.hlsli"
#include "../../include/common/microfacet.hlsli"
#include "../../include/scene_descriptor_set.hlsli"

float3 getMotionVector(
    in_ref(CameraData) camera,
    in_ref(CameraData) prevCamera,
    in_ref(GeometryInfo) geometryInfo,
    in_ref(GeometryInfo) prevGeometryInfo,
    float3 objectSpacePosition,
    float3 prevObjectSpacePosition,
    out float o_viewDepth)
{
    const float3 worldSpacePosition = mul(float4(objectSpacePosition, 1.0), ObjectToWorld(geometryInfo)).xyz;
    const float3 prevWorldSpacePosition = mul(float4(prevObjectSpacePosition, 1.0), ObjectToWorld(prevGeometryInfo)).xyz;

    float4 clipPos = mul(float4(worldSpacePosition, 1.0), camera.viewProjMat);
    clipPos.xyz /= clipPos.w;
    float4 prevClipPos = mul(float4(prevWorldSpacePosition, 1.0), prevCamera.viewProjMat);
    prevClipPos.xyz /= prevClipPos.w;

    o_viewDepth = clipPos.w;

    if (clipPos.w <= 0 || prevClipPos.w <= 0)
        return float3(0);

    const float2 windowPos = clipPos.xy * camera.clipToWindowScale * float2(1, -1) + camera.clipToWindowBias;
    const float2 prevWindowPos = prevClipPos.xy * prevCamera.clipToWindowScale * float2(1, -1) + prevCamera.clipToWindowBias;

    float3 motion;
    motion.xy = prevWindowPos.xy - windowPos.xy;
    motion.xy += (float2(camera.jitterX, camera.jitterY) - float2(prevCamera.jitterX, prevCamera.jitterY));
    motion.z = prevClipPos.w - clipPos.w;
    return motion;
}

float computeFWidthDepth(
    in_ref(float) cone_radius,
    in_ref(float3) ray_direction,
    in_ref(float3) normal
) {
    // Compute ellipse axes.
    float3 a1 = ray_direction - dot(normal, ray_direction) * normal;
    float3 p1 = a1 - dot(ray_direction, a1) * ray_direction;
    a1 *= cone_radius / max(0.0001, length(p1));

    float3 a2 = cross(normal, a1);
    float3 p2 = a2 - dot(ray_direction, a2) * ray_direction;
    a2 *= cone_radius / max(0.0001, length(p2));
    
    return 1.0 / max(0.1, abs(dot(a1, ray_direction)) + abs(dot(a2, ray_direction)));
}

#endif // !_SRENDERER_GBUFFER_ADDON_HEADER_