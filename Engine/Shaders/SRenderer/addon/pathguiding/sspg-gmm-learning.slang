#include "../gbuffer/gbuffer_common.hlsli"
#include "../gbuffer/gbuffer_interface.hlsli"
#include "../gbuffer/gbuffer_prev_interface.hlsli"
#include "../../include/common/concentric_map.hlsli"
#include "../../include/common/random.hlsli"
#include "include/pathguiding.hlsli"

RWTexture2D<float4> u_gmmStatisticsPack0;
RWTexture2D<float4> u_gmmStatisticsPack1;

const Texture2D<float4> u_vpls;

groupshared float4 gs_vpls[20][20];
groupshared float4 gs_geometry[20][20];

[[vk::push_constant]]
cbuffer PushConstants {
    int2 resolution;
    int reuse_number;
    float exponential_factor;
    bool adaptation;
    int random_seed;
};

bool Valid(int2 index) {
    return all(index >= 0 && index < resolution);
}

void UpdateGMM(
    in_ref(float) sumWeight, in_ref(float2) square_coord,
    inout_ref(float4) pack0, inout_ref(float4) pack1,
) {
    uint epoch_count = uint(pack1.z);
    if (sumWeight > 0) {
        epoch_count = clamp(epoch_count, 0, 10);
        // exponential smoothing vMF
        const float alpha = pow(exponential_factor, epoch_count);
        const float4 new_pack0 = sumWeight * float4(square_coord.x, square_coord.y, square_coord.x * square_coord.x, square_coord.y * square_coord.y);
        const float2 new_pack1 = sumWeight * float2(square_coord.x * square_coord.y, 1);
        const float4 update_pack0 = ExponentialSmooth(pack0, new_pack0, alpha);
        const float2 update_pack1 = ExponentialSmooth(pack1.xy, new_pack1, alpha);
        epoch_count += 1;
        pack0 = update_pack0;
        pack1 = float4(update_pack1, epoch_count, 0);
    }
}

int2 GetNeighborIndex(int2 center_xy, inout_ref(RandomSamplerState) RNG) {
    const float2 rand = GetNextRandomFloat2(RNG);
    const int2 offset = clamp(int2(rand * 5), 0, 4);
    return center_xy + offset - int2(2);
}

static const float depthThreshold = 0.1;
static const float normalThreshold = 0.5;

[shader("compute")]
[numthreads(16, 16, 1)] // 8x8x8 threads in a group
void ComputeMain(
    int3 dtid: SV_DispatchThreadID,
    int3 gt: SV_GroupThreadID,
    int3 gid: SV_GroupID,
    int gi: SV_GroupIndex,
) {
    if (!adaptation) return;

    const int2 pixel = dtid.xy;
    const int2 group = gt.xy;
    RandomSamplerState RNG = InitRandomSampler(pixel, random_seed);

    // First load 20 x 20 the vpls into shared memory
    {   // The offset indicate the left-top element position
        const int2 offset = gid.xy * 16 - 2;
        if (gi < 200) {
            const int2 flatten = int2(gi % 20, gi / 20);
            { // First half of the group
                const int2 index_0 = offset + flatten;
                const float4 vpl_0 = Valid(index_0) ? u_vpls[index_0] : float4(0);
                float4 geometry_0 = float4(0);
                if (Valid(index_0)) {
                    const ShadingSurface surface = GetGBufferSurface(index_0, globalUniform.cameraData);
                    geometry_0 = float4(surface.geometryNormal, surface.viewDepth);
                }
                gs_vpls[flatten.x][flatten.y] = vpl_0;
                gs_geometry[flatten.x][flatten.y] = geometry_0;
            }
            { // Second half of the group
                const int2 index_1 = offset + flatten + int2(0, 10);
                const float4 vpl_1 = Valid(index_1) ? u_vpls[index_1] : float4(0);
                float4 geometry_1 = float4(0);
                if (Valid(index_1)) {
                    const ShadingSurface surface = GetGBufferSurface(index_1, globalUniform.cameraData);
                    geometry_1 = float4(surface.geometryNormal, surface.viewDepth);
                }
                gs_vpls[flatten.x][flatten.y + 10] = vpl_1;
                gs_geometry[flatten.x][flatten.y] = geometry_1;
            }
        }
    }

    GroupMemoryBarrierWithGroupSync();

    const int2 center_xy = group.xy + 2;
    const ShadingSurface center_surface = GetGBufferSurface(pixel, globalUniform.cameraData);
    if (!IsShadingSurfaceValid(center_surface)) {
        return;
    }

    float4 pack0 = u_gmmStatisticsPack0[pixel];
    float4 pack1 = u_gmmStatisticsPack1[pixel];
    const float3x3 frame = createFrame(center_surface.shadingNormal);

    // First, load the center vpls.
    const float4 center_vpl = gs_vpls[center_xy.x][center_xy.y];
    const float3 direction = normalize(center_vpl.xyz - center_surface.worldPos);
    const float2 square_coord = FromConcentricMap(UniformHemisphereToConcentricDisk(to_local(frame, direction)));
    UpdateGMM(center_vpl.w, square_coord, pack0, pack1);

    for (int i = 0; i < reuse_number; ++i) {
        const int2 neighbor_id = GetNeighborIndex(center_xy, RNG);
        // Then, load the neighbor vpls.
        const float4 geometry = gs_geometry[neighbor_id.x][neighbor_id.y];
        const float4 neighbor_vpl = gs_vpls[neighbor_id.x][neighbor_id.y];
        if (IsValidNeighbor(
                center_surface.geometryNormal, geometry.xyz,
                center_surface.viewDepth, geometry.w,
                normalThreshold, depthThreshold)) {
            const float3 direction = normalize(neighbor_vpl.xyz - center_surface.worldPos);
            const float2 square_coord = FromConcentricMap(UniformHemisphereToConcentricDisk(to_local(frame, direction)));
            UpdateGMM(neighbor_vpl.w, square_coord, pack0, pack1);
        }
    }

    u_gmmStatisticsPack0[pixel] = pack0;
    u_gmmStatisticsPack1[pixel] = pack1;
}