#include "../../include/common/concentric_map.hlsli"
#include "../../include/common/random.hlsli"
#include "../../include/common/vmf.hlsli"
#include "../gbuffer/gbuffer_common.hlsli"
#include "../gbuffer/gbuffer_interface.hlsli"
#include "../gbuffer/gbuffer_prev_interface.hlsli"
#include "../vbuffer/vbuffer.hlsli"
#include "include/pathguiding.hlsli"


const Texture2D<float4> u_vpl0;
const Texture2D<float4> u_vpl1;
// const Texture2D<float4> u_vpl2;
const Texture2D<uint4> u_vBuffer;

RWTexture2D<float4> u_vMFStatistics;
RWTexture2D<uint> u_epochCounter;

[[vk::push_constant]]
cbuffer PushConstants {
    int2 resolution;
    int reuse_number;
    float exponential_factor;
    bool adaptation;
};

int UpdateVMFLobe(
    in_ref(float4) vpl,
    in_ref(float3) position,
    inout_ref(vMFMLEStatistics) vMFStat,
    inout_ref(int) epochCounter,
) {
    const float3 direction = normalize(vpl.xyz - position);
    const float sumWeights = vpl.w;

    if (sumWeights > 0 && adaptation) {
        uint epochCount = clamp(epochCounter, 0, 20);
        // exponential smoothing vMF
        const float alpha = pow(0.7, epochCount);
        const float3 sumeightedDirections = direction * sumWeights;
        vMFStat.sumeightedDirections = ExponentialSmooth(vMFStat.sumeightedDirections, sumeightedDirections, alpha);
        vMFStat.sumWeights = ExponentialSmooth(vMFStat.sumWeights, sumWeights, alpha);
        epochCounter += 1;
        return 1;
    }
    else return 0;
}

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

    // Load Primary Hit from VBuffer
    // --------------------------------------------------------------------
    const Ray primaryRay = generateRay(pixel, resolution, false, globalUniform.cameraData);
    const VBufferData vhit = UnpackVBufferData(u_vBuffer[pixel]);
    if (IsInvalid(vhit)) return;
    const GeometryHit hit = GetGeometryHit(vhit, primaryRay);

    // Load vMF Statistics
    // --------------------------------------------------------------------
    vMFMLEStatistics vMFStat = vMFMLEStatistics(u_vMFStatistics[pixel]);
    vMFDistribution vMF = vMFDistribution(vMFStat);
    int epochCounter = u_epochCounter[pixel];

    // Update vMF Lobe
    // --------------------------------------------------------------------
    int update_count = 0;
    update_count += UpdateVMFLobe(u_vpl0[pixel], hit.position, vMFStat, epochCounter);
    update_count += UpdateVMFLobe(u_vpl1[pixel], hit.position, vMFStat, epochCounter);
    // update_count += UpdateVMFLobe(u_vpl2[pixel], hit.position, vMFStat, epochCounter);

    // Save vMF Statistics
    // --------------------------------------------------------------------
    u_vMFStatistics[pixel] = vMFStat.Pack();
    u_epochCounter[pixel] += update_count;
}