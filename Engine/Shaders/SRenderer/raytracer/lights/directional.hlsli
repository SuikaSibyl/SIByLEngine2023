#ifndef _SRENDERER_RAYTTRACER_LIGHT_DIRECTIONAL_HEADER_
#define _SRENDERER_RAYTTRACER_LIGHT_DIRECTIONAL_HEADER_

#include "../../include/common/math.hlsli"
#include "../../include/common/geometry.hlsli"
#include "../../include/common/sampling.hlsli"
#include "../../include/common/cpp_compatible.hlsli"

/**
 * Create a sample from a directional light.
 * By using a bounding sphere of the scene.
 * @param direction The direction of the light.
 * @param boundingSphere The bounding sphere of the scene.
 * @param uv A pair of random samples.
 * @param origin The origin of the sample ray.
 * @param pdf The probability density function of the sample.
 */
void DirectionalLightSample(
    in_ref(float3) direction,
    in_ref(float4) boundingSphere,
    in_ref(float2) uv,
    out_ref(float3) origin,
    out_ref(float) pdf
) {
    const float3x3 obn = createONB(direction);
    const float2 cd = uniformSampleDisk(uv);
    origin = boundingSphere.xyz - boundingSphere.w * direction 
                       + boundingSphere.w * (cd.x * obn[0] + cd.y * obn[1]);
    pdf = 1. / (k_pi * boundingSphere.w * boundingSphere.w);
}

#endif // !_SRENDERER_RAYTTRACER_LIGHT_DIRECTIONAL_HEADER_