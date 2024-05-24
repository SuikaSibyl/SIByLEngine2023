RWTexture2D<float> pmf_texture;
RWTexture2D<float> sum_texture;
RWStructuredBuffer<int> summation;

groupshared float group_sum[32];
groupshared uint group_index;

[shader("compute")]
[numthreads(32, 32, 1)]
void ComputeMain(
    int3 dtid: SV_DispatchThreadID,
    int3 gtid: SV_GroupThreadID,
    int3 gid: SV_GroupID,
    int gi: SV_GroupIndex
) {
    // Each thread reads 4 texels
    const int2 texel = int2(dtid.xy) * 2;
    const float v00 = pmf_texture[texel + int2(0, 0)].x;
    const float v01 = pmf_texture[texel + int2(0, 1)].x;
    const float v10 = pmf_texture[texel + int2(1, 0)].x;
    const float v11 = pmf_texture[texel + int2(1, 1)].x;

    // Sum the 4 texels
    const float val = v00 + v01 + v10 + v11;
    // warp level reduction
    const float warp_val = WaveActiveSum(val);
    const uint wave = gi / 32;
    if (WaveIsFirstLane()) {
        group_sum[wave] = warp_val;
    }

    GroupMemoryBarrierWithGroupSync();
    if (wave == 0) {
        const float val = group_sum[WaveGetLaneIndex()];
        const float group_val = WaveActiveSum(val);
        if (gi == 0) {
            sum_texture[gid.xy] = group_val;
            uint original;
            InterlockedAdd(summation[1], 1, original);
            group_index = original;
        }
    }
    
    GroupMemoryBarrierWithGroupSync();
    DeviceMemoryBarrierWithGroupSync();

    const uint2 snd_texel = uint2(gi % 8, gi / 8);
    const float snd_val = sum_texture[snd_texel].x;
    const float snd_group_val = WaveActiveSum(snd_val);
    if (WaveGetLaneIndex() == 0) {
        group_sum[wave] = snd_group_val;
    }

    GroupMemoryBarrierWithGroupSync();

    if ((group_index == 63) && (gi < 8 * 8)) {
        if (gi == 0) {
            float all_sum = group_sum[0] + group_sum[1];
            summation[0] = asint(all_sum);
            summation[1] = 0;
        }
    }
}