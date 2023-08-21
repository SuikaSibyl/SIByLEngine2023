#ifndef _SRENDERER_ADDON_VPL_INTERFACE_HEADER_
#define _SRENDERER_ADDON_VPL_INTERFACE_HEADER_

#include "../../include/common/cpp_compatible.hlsli"

/**
* @brief Evaluate the attenuation function for VPLs.
* @param p The point to evaluate the attenuation function at.
* @param N The normal at the point.
* @param lightPos The position of the VPL light.
* @param lightNormal The normal of the VPL light.
* @param invNumPaths The inverse number of paths used to generate the VPL.
* @return The attenuation factor.
*/
float3 AttenFuncVPL(
    in_ref(float3) p,
    in_ref(float3) N,
    in_ref(float3) lightPos,
    in_ref(float3) lightNormal,
    in_ref(float) invNumPaths,
    out_ref(float4) ray_desc
) {
    float3 lightDir = lightPos - p;
    const float lightDist = length(lightDir);
    lightDir /= lightDist;
    float3 output = saturate(dot(lightNormal, -lightDir)) *
                    saturate(dot(N, lightDir)) / (lightDist * lightDist);
    output *= invNumPaths; // stores invNumPaths here
    ray_desc = float4(lightDir, lightDist);
    return output;
}

#endif // _SRENDERER_ADDON_VPL_INTERFACE_HEADER_