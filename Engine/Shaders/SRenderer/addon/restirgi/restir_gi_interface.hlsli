#ifndef _SRENDERER_RESTIR_GI_INTERFACE_HEADER_
#define _SRENDERER_RESTIR_GI_INTERFACE_HEADER_

#include "../../include/common/cpp_compatible.hlsli"
#include "../../include/common/math.hlsli"
#include "../../include/common/microfacet.hlsli"
#include "../../include/common/shading.hlsli"

// Creates a GI reservoir from a raw light sample.
// Note: the original sample PDF can be embedded into sampleRadiance, in which case the samplePdf parameter should be set to 1.0.
RTXDI_GIReservoir RTXDI_MakeGIReservoir(
    const float3 samplePos,
    const float3 sampleNormal,
    const float3 sampleRadiance,
    const float samplePdf)
{
    RTXDI_GIReservoir reservoir;
    reservoir.position = samplePos;
    reservoir.normal = sampleNormal;
    reservoir.radiance = sampleRadiance;
    reservoir.weightSum = samplePdf > 0.0 ? 1.0 / samplePdf : 0.0;
    reservoir.M = 1;
    reservoir.age = 0;
    return reservoir;
}

// static const float kMinRoughness = 0.05f;

// struct SplitBrdf {
//     float demodulatedDiffuse;
//     float3 specular;
// };

// SplitBrdf EvaluateBrdf(in_ref(ShadingSurface) surface, in_ref(float3) samplePosition) {
//     float3 N = surface.normal;
//     float3 V = surface.viewDir;
//     float3 L = normalize(samplePosition - surface.worldPos);

//     SplitBrdf brdf;
//     brdf.demodulatedDiffuse = lambert(surface.normal, -L);
//     brdf.specular = 0;
//     // TODO :: include specular lobe
//     // if (surface.roughness == 0)
//     //     brdf.specular = 0;
//     // else
//     //     brdf.specular = GGX_times_NdotL(V, L, surface.normal, max(surface.roughness, kMinRoughness), surface.specularF0);
//     return brdf;
// }

// // Computes the weight of the given GI sample when the given surface is shaded using that GI sample.
// float GetGISampleTargetPdfForSurface(
//     in_ref(float3) samplePosition,
//     in_ref(float3) sampleRadiance,
//     in_ref(ShadingSurface) surface)
// {
//     SplitBrdf brdf = EvaluateBrdf(surface, samplePosition);
//     const float3 reflectedRadiance = sampleRadiance * (brdf.demodulatedDiffuse * surface.diffuseAlbedo + brdf.specular);
//     return luminance(reflectedRadiance);
// }

// // Check if the sample is fine to be used as a valid spatial sample.
// // This function also be able to clamp the value of the Jacobian.
// bool ValidateGISampleWithJacobian(inout_ref(float) jacobian) {
//     // Sold angle ratio is too different. Discard the sample.
//     if (jacobian > 10.0 || jacobian < 1 / 10.0) {
//         return false;
//     }
//     // clamp Jacobian.
//     jacobian = clamp(jacobian, 1 / 3.0, 3.0);
//     return true;
// }

// bool GetConservativeVisibility(RaytracingAccelerationStructure accelStruct, RAB_Surface surface, float3 samplePosition) {
//     RayDesc ray = setupVisibilityRay(surface, samplePosition);

//     RayPayload payload = (RayPayload)0;
//     payload.instanceID = ~0u;
//     payload.throughput = 1.0;

//     TraceRay(accelStruct, RAY_FLAG_CULL_NON_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, INSTANCE_MASK_OPAQUE, 0, 0, 0, ray, payload);

//     bool visible = (payload.instanceID == ~0u);

//     REPORT_RAY(!visible);

//     return visible;
// }

// // Traces a cheap visibility ray that returns approximate, conservative visibility
// // between the surface and the light sample. Conservative means if unsure, assume the light is visible.
// // Significant differences between this conservative visibility and the final one will result in more noise.
// // This function is used in the spatial resampling functions for ray traced bias correction.
// bool GetConservativeVisibility(in_ref(ShadingSurface) surface, float3 samplePosition)
// {
//     return GetConservativeVisibility(SceneBVH, surface, samplePosition);
// }

// // Same as RAB_GetConservativeVisibility but for temporal resampling.
// // When the previous frame TLAS and BLAS are available, the implementation should use the previous position and the previous AS.
// // When they are not available, use the current AS. That will result in transient bias.
// bool RAB_GetTemporalConservativeVisibility(RAB_Surface currentSurface, RAB_Surface previousSurface, float3 samplePosition)
// {
//     if (g_Const.enablePreviousTLAS)
//         return GetConservativeVisibility(PrevSceneBVH, previousSurface, samplePosition);
//     else
//         return GetConservativeVisibility(SceneBVH, currentSurface, samplePosition);
// }

#endif // !_SRENDERER_RESTIR_GI_INTERFACE_HEADER_