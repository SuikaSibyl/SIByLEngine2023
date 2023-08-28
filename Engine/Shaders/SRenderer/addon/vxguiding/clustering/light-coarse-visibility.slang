#include "../../../include/common/octahedral.hlsli"
#include "../../../raytracer/spt_interface.hlsli"

const StructuredBuffer<float4> u_RepresentPixel;
const StructuredBuffer<float4> u_RepresentVPL;

RWStructuredBuffer<uint> u_SparseVisibility;

[shader("compute")]
[numthreads(32, 8, 1)]
void ComputeMain(
    int3 DTid: SV_DispatchThreadID,
    int gidx: SV_GroupIndex,
) {
    // Get the representative vpl of the voxel
    const uint compactID = DTid.y;
    const float4 vpl = u_RepresentVPL[compactID];
    const float3 vpl_position = vpl.xyz;
    const float3 vpl_normal = Unorm32OctahedronToUnitVector(asuint(vpl.w));

    // Get the representative pixel
    const uint pixelID = DTid.x;
    const float4 shadePoint = u_RepresentPixel[compactID];
    float3 pixel_position = shadePoint.xyz;
    float3 pixel_normal = Unorm32OctahedronToUnitVector(asuint(shadePoint.w));

    // Get the visibility and insert it into the buffer
    bool visible = true;

    // Check if the vpl is visible from the pixel
    float3 direction = vpl_position - pixel_position;
    const float distannce = length(direction);
    direction /= distannce;
    if (dot(pixel_normal, direction) < 0.0 
     || dot(vpl_normal, -direction) < 0.0) {
        visible = false; }
    if (visible) {
        Ray visiblityRay;
        visiblityRay.origin = pixel_position;
        visiblityRay.direction = direction;
        visiblityRay.tMax = max(distannce - 0.02, 0);
        visiblityRay.tMin = 0.01;
        RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> q;
        q.TraceRayInline(
            SceneBVH,
            0, 0xff,
            ToRayDesc(visiblityRay));
        if (q.Proceed()) {
            visible = false;
        }
    }
    
    // Insert the visibility into the buffer
    const uint4 visibilityMask = WaveActiveBallot(visible);
    if (WaveIsFirstLane()) {
        const uint maskID = pixelID / 32;
        u_SparseVisibility[compactID * 4 + 0] = visibilityMask.x;
    }
}