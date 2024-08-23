#include "octmap.hlsli"
#include "common/waveintrinsic.hlsli"

RWByteAddressBuffer octmap_buffer;

groupshared float shared_buffer[512];

[shader("compute")]
[numthreads(512, 1, 1)]
void ComputeMain(
    int3 dtid: SV_DispatchThreadID,
    int3 gtid: SV_GroupThreadID,
    int3 ctid: SV_GroupID,
    int gid: SV_GroupIndex
) {
    const int threadID = dtid.x;
    const int octmapID = threadID / 64;
    const int elementID = threadID % 64;

    const int offset = octmapID * 84;
    const float element = octmap_buffer.Load<float>((offset + elementID) * 4);
    const int submapID = gid / 64;
    shared_buffer[submapID * 64 + elementID] = element;

    const int index_map[16] = {
        0, 1, 8, 9,
        2, 3, 10, 11,
        16, 17, 24, 25,
        18, 19, 26, 27
    };
    const int index_offset[4] = { 0, 4, 32, 36 };
    const int element_index = index_map[elementID % 16] 
        + index_offset[elementID / 16];
    
    GroupMemoryBarrierWithGroupSync();
    const float element_remap = shared_buffer[submapID * 64 + element_index];
    const float cluster_sum4 = glsl_subgroupClusteredAdd_ClusterSize4(element_remap);
    const int constant_b[4] = { 0, 1, 4, 5 };
    const int constant_c[4] = { 0, 2, 8, 10 };
    if (elementID % 4 == 0) {
        const int tmp = elementID / 4;
        const int subindex = constant_b[tmp % 4] + constant_c[tmp / 4];
        octmap_buffer.Store<float>((offset + 64 + subindex) * 4, cluster_sum4);
    }

    const float cluster_sum16 = glsl_subgroupClusteredAdd_ClusterSize16(element_remap);
    if (elementID % 16 == 0) {
        octmap_buffer.Store<float>((offset + 80 + elementID / 16) * 4, cluster_sum16);
    }
}