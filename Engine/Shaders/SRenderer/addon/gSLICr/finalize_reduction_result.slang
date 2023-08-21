#include "../../include/common/cpp_compatible.hlsli"
#include "gSLICr_interface.hlsli"

[[vk::push_constant]]
cbuffer PushConstants {
    int2 map_size;
    int2 img_size;
    int spixel_size;
    int no_blocks_per_line;
    int no_blocks_per_spixel;
};

StructuredBuffer<spixel_info> u_inAccumMap;
RWStructuredBuffer<spixel_info> u_spixelList;

void finalize_reduction_result(
    StructuredBuffer<spixel_info> accum_map,
    RWStructuredBuffer<spixel_info> spixel_list,
    int2 map_size,
    int no_blocks_per_spixel,
    int x, int y
) {
    int spixel_idx = y * map_size.x + x;

    spixel_list[spixel_idx].center = float2(0, 0);
    spixel_list[spixel_idx].color_info = float4(0, 0, 0, 0);
    spixel_list[spixel_idx].no_pixels = 0;

    for (int i = 0; i < no_blocks_per_spixel; i++) {
        int accum_list_idx = spixel_idx * no_blocks_per_spixel + i;
        
        spixel_list[spixel_idx].center += accum_map[accum_list_idx].center;
        spixel_list[spixel_idx].color_info += accum_map[accum_list_idx].color_info;
        spixel_list[spixel_idx].no_pixels += accum_map[accum_list_idx].no_pixels;
    }

    if (spixel_list[spixel_idx].no_pixels != 0) {
        spixel_list[spixel_idx].center /= (float)spixel_list[spixel_idx].no_pixels;
        spixel_list[spixel_idx].color_info /= (float)spixel_list[spixel_idx].no_pixels;
    }
}

[shader("compute")]
[numthreads(gSLICr_BLOCK_DIM, gSLICr_BLOCK_DIM, 1)]
void ComputeMain(int3 DTid: SV_DispatchThreadID) {
    if (any(DTid.xy >= img_size)) return;
    finalize_reduction_result(
        u_inAccumMap, u_spixelList, map_size,
        no_blocks_per_spixel,
        DTid.x, DTid.y);
}