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

Texture2D<float4> u_inImg;
Texture2D<int> u_inIdxImg;
RWStructuredBuffer<spixel_info> u_outAccumMap;

groupshared float4 color_shared[gSLICr_BLOCK_DIM * gSLICr_BLOCK_DIM];
groupshared float2 xy_shared[gSLICr_BLOCK_DIM * gSLICr_BLOCK_DIM];
groupshared int count_shared[gSLICr_BLOCK_DIM * gSLICr_BLOCK_DIM];
groupshared bool should_add;

/**
 * For each superpixel, ...
 */
void update_cluster_center(
    Texture2D<float4> inimg,
    Texture2D<int> in_idx_img,
    RWStructuredBuffer<spixel_info> accum_map,
    int2 map_size,
    int2 img_size,
    int spixel_size,
    int no_blocks_per_line,
    int2 threadIdx,
    int3 blockIdx
) {
    int local_id = threadIdx.y * gSLICr_BLOCK_DIM + threadIdx.x;

    color_shared[local_id] = float4(0, 0, 0, 0);
    xy_shared[local_id] = float2(0, 0);
    count_shared[local_id] = 0;
    should_add = false;
    GroupMemoryBarrierWithGroupSync();
    
    int spixel_id = blockIdx.y * map_size.x + blockIdx.x;

    // compute the relative position in the search window
    int block_x = blockIdx.z % no_blocks_per_line;
    int block_y = blockIdx.z / no_blocks_per_line;

    int x_offset = block_x * gSLICr_BLOCK_DIM + threadIdx.x;
    int y_offset = block_y * gSLICr_BLOCK_DIM + threadIdx.y;

    if (x_offset < spixel_size * 3 && y_offset < spixel_size * 3) {
        // compute the start of the search window
        int x_start = blockIdx.x * spixel_size - spixel_size;
        int y_start = blockIdx.y * spixel_size - spixel_size;

        int x_img = x_start + x_offset;
        int y_img = y_start + y_offset;

        if (x_img >= 0 && x_img < img_size.x && y_img >= 0 && y_img < img_size.y)
        {
            int img_idx = y_img * img_size.x + x_img;
            if (in_idx_img[img_idx] == spixel_id) {
                color_shared[local_id] = inimg[img_idx];
                xy_shared[local_id] = float2(x_img, y_img);
                count_shared[local_id] = 1;
                should_add = true;
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();

    if (should_add) {
        if (local_id < 128) {
            color_shared[local_id] += color_shared[local_id + 128];
            xy_shared[local_id] += xy_shared[local_id + 128];
            count_shared[local_id] += count_shared[local_id + 128];
        }
        GroupMemoryBarrierWithGroupSync();

        if (local_id < 64) {
            color_shared[local_id] += color_shared[local_id + 64];
            xy_shared[local_id] += xy_shared[local_id + 64];
            count_shared[local_id] += count_shared[local_id + 64];
        }
        GroupMemoryBarrierWithGroupSync();

        if (local_id < 32) {
            color_shared[local_id] += color_shared[local_id + 32];
            color_shared[local_id] += color_shared[local_id + 16];
            color_shared[local_id] += color_shared[local_id + 8];
            color_shared[local_id] += color_shared[local_id + 4];
            color_shared[local_id] += color_shared[local_id + 2];
            color_shared[local_id] += color_shared[local_id + 1];

            xy_shared[local_id] += xy_shared[local_id + 32];
            xy_shared[local_id] += xy_shared[local_id + 16];
            xy_shared[local_id] += xy_shared[local_id + 8];
            xy_shared[local_id] += xy_shared[local_id + 4];
            xy_shared[local_id] += xy_shared[local_id + 2];
            xy_shared[local_id] += xy_shared[local_id + 1];

            count_shared[local_id] += count_shared[local_id + 32];
            count_shared[local_id] += count_shared[local_id + 16];
            count_shared[local_id] += count_shared[local_id + 8];
            count_shared[local_id] += count_shared[local_id + 4];
            count_shared[local_id] += count_shared[local_id + 2];
            count_shared[local_id] += count_shared[local_id + 1];
        }
    }
    GroupMemoryBarrierWithGroupSync();

    if (local_id == 0) {
        int accum_map_idx = spixel_id * no_blocks_per_spixel + blockIdx.z;
        accum_map[accum_map_idx].center = xy_shared[0];
        accum_map[accum_map_idx].color_info = color_shared[0];
        accum_map[accum_map_idx].no_pixels = count_shared[0];
    }
}

/**
 * Use one thread per cluster center to initialize superpixel map.
 * This is an ns_r x ns_c image which contains, or each entry, center coordinates,
 * number of associated pixels and colour information.
 * ns_r and ns_c represent the number of superpixels per image row and column, respectively.
 */
[shader("compute")]
[numthreads(gSLICr_BLOCK_DIM, gSLICr_BLOCK_DIM, 1)]
void ComputeMain(
    int3 DTid: SV_DispatchThreadID,
    int3 GTid: SV_GroupThreadID,
    int3 Gid: SV_GroupID
) {
    if (any(DTid.xy >= map_size)) return;

    update_cluster_center(
        u_inImg,
        u_inIdxImg,
        u_outAccumMap,
        map_size,
        img_size,
        spixel_size,
        no_blocks_per_line,
        GTid.xy,
        Gid);
}