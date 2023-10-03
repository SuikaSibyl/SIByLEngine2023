#include "spt_interface.hlsli"
#include "../include/common/random.hlsli"
#include "../include/scene_descriptor_set.hlsli"

#include "primitives/trimesh.hlsli"
#include "primitives/sphere.hlsli"

#include "materials/lambertian.hlsli"
#include "materials/roughplastic.hlsli"
#include "materials/roughdielectric.hlsli"

// Shadow ray ...
[shader("closesthit")]
void ShadowRayClosestHit(inout ShadowPayload payload: SV_RayPayload) {
    payload.occluded = true;
}
[shader("miss")]
void ShadowRayMiss(inout ShadowPayload payload: SV_RayPayload) {
    payload.occluded = false;
}

[shader("miss")]
void CommonRayMiss(inout PrimaryPayload payload: SV_RayPayload) {
    SetHit(payload.hit, false);
}