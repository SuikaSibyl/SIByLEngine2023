#include "../../include/common/cpp_compatible.hlsli"
#include "../../include/common/octahedral.hlsli"
#include "gSLICr_interface.hlsli"

[[vk::push_constant]]
cbuffer PushConstants {
    int2 map_size;
    int2 img_size;
    int spixel_size;
    float weight;
    float max_xy_dist;
    float max_color_dist;
};

Texture2D<float4> u_inImg;
StructuredBuffer<spixel_info> u_inSPixel;
RWTexture2D<int> u_outIdxImg;

RWTexture2D<float4> u_fuzzyWeight;
RWTexture2D<int4> u_fuzzyIDX;

struct FuzzyVec {
    int center[4];
    float dist2[4];
    int size = 0;

    int get_maximum() {
        int max_idx = 0;
        float max_dist = dist2[0];
        for (int i = 1; i < 4; ++i) {
            if (dist2[i] > max_dist) {
                max_dist = dist2[i];
                max_idx = i;
            }
        }
        return max_idx;
    }

    [mutating]
    void insert(int c, float d2) {
        if (size < 4) {
            center[size] = c;
            dist2[size] = d2;
            size++;
        }
        else {
            int max_idx = get_maximum();
            if (d2 < dist2[max_idx]) {
                center[max_idx] = c;
                dist2[max_idx] = d2;
            }
        }
    }
};

void Pack(in_ref(FuzzyVec) vec, int2 idx_img) {
    float4 weight = float4(0, 0, 0, 0);
    int4 center = int4(-1);
    if (vec.size == 0) {
        u_fuzzyIDX[idx_img] = center;
        u_fuzzyWeight[idx_img] = weight;
        return;
    }
    int zero_idx = -1;
    for (int i = 0; i < vec.size; ++i) {
        if (vec.dist2[i] == 0) zero_idx = i;
        weight[i] = 1.f / vec.dist2[i];
        center[i] = vec.center[i];
    }
    if (zero_idx != -1) {
        // for (int i = 0; i < 4; ++i)
        //     weight[i] = isnan(weight[i]) ? 1 : 0;
        weight = float4(0);
        weight[zero_idx] = 1;
    }
    weight /= dot(weight, float4(1));
    u_fuzzyIDX[idx_img] = center;
    u_fuzzyWeight[idx_img] = weight;
}

float2 compute_slic_distance(
    in_ref(float3) pix_pos, in_ref(float3) pix_normal, in_ref(float2) pix_xy,
    in_ref(float3) sp_pos, in_ref(float3) sp_normal, in_ref(float2) sp_xy,
    float weight,
    float normalizer_xy,
    float normalizer_color
) {
    float dot_normal = dot(pix_normal, sp_normal);
    const float3 diff_color = pix_pos - sp_pos;
    float dcolor = dot(diff_color, diff_color);
    const float2 diff_xy = pix_xy - sp_xy;
    const float dxy = dot(diff_xy, diff_xy);
    const float retval = dcolor * normalizer_color + weight * dxy * normalizer_xy;
    const float sqrt_retval = sqrt(retval);
    return float2((dot_normal > 0.01) ? sqrt_retval : 1000000, sqrt_retval);
}

groupshared float3 spixel_pos[3 * 3];
groupshared float3 spixel_normals[3 * 3];
groupshared float2 spixel_xy[3 * 3];
groupshared int spixel_ids[3 * 3];

int offset2idx(int2 offset) {
    return (offset.y + 1) * 3 + offset.x + 1;
}

void find_center_association(
    Texture2D<float4> inimg,
    StructuredBuffer<spixel_info> in_spixel_map,
    RWTexture2D<int> out_idx_img,
    int2 map_size,
    int2 img_size,
    int spixel_size,
    float weight,
    int x, int y,
    float max_xy_dist,
    float max_color_dist,
    int gid
) {
    // idx_imag is the pixel coordinate
    const int2 idx_img = int2(x, y);
    // ctr x/y should be similar for the whole group
    const int ctr_x = x / spixel_size;
    const int ctr_y = y / spixel_size;
    // load the pixel color
    const float4 pix_val = inimg[idx_img];
    const float3 pix_normal = Unorm32OctahedronToUnitVector(asuint(pix_val.w));

    if (gid < 9) {
        const int2 offset = int2(gid % 3 - 1, gid / 3 - 1);
        const int idx = offset2idx(offset);
        int ctr_x_check = ctr_x + offset.x;
        int ctr_y_check = ctr_y + offset.y;
        if (ctr_x_check >= 0 && ctr_y_check >= 0 && ctr_x_check < map_size.x && ctr_y_check < map_size.y) {
            int ctr_idx = ctr_y_check * map_size.x + ctr_x_check;
            spixel_info center_info = in_spixel_map[ctr_idx];
            spixel_pos[idx] = center_info.color_info.xyz;
            spixel_normals[idx] = Unorm32OctahedronToUnitVector(asuint(center_info.color_info.w));
            spixel_xy[idx] = center_info.center;
            spixel_ids[idx] = center_info.id;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    int minidx = -1;
    float dist = 999999.9999f;

    int minidx_fallback = -1;
    float dist_fallback = 999999.9999f;

    FuzzyVec fuzzy_vec;
    FuzzyVec fuzzy_vec_fallback;
    fuzzy_vec.size = 0;
    fuzzy_vec_fallback.size = 0;

    // search 3x3 neighborhood
    // TODO :: use shared memory to speed up
    for (int i = -1; i <= 1; i++)
        for (int j = -1; j <= 1; j++) {
            int ctr_x_check = ctr_x + j;
            int ctr_y_check = ctr_y + i;
            if (ctr_x_check >= 0 && ctr_y_check >= 0 && ctr_x_check < map_size.x && ctr_y_check < map_size.y) {
                int ctr_idx = ctr_y_check * map_size.x + ctr_x_check;
                const int offset = offset2idx(int2(j, i));
                float2 cdist = compute_slic_distance(
                    pix_val.xyz, pix_normal, float2(x, y),
                    spixel_pos[offset], spixel_normals[offset], spixel_xy[offset],
                    weight, max_xy_dist, max_color_dist);
                if (cdist.x < dist) {
                    dist = cdist.x;
                    minidx = spixel_ids[offset];
                }
                // if(cdist.y < dist_fallback) {
                //     dist_fallback = cdist.y;
                //     minidx_fallback = in_spixel_map[ctr_idx].id;
                // }
                // fuzzy_vec.insert(ctr_idx, cdist.x);
                // fuzzy_vec_fallback.insert(ctr_idx, cdist.y);
            }
        }
    out_idx_img[idx_img] = minidx;
    // if (minidx >= 0) {
    //     out_idx_img[idx_img] = minidx;
    //     Pack(fuzzy_vec, idx_img);
    // }
    // else {
    //     out_idx_img[idx_img] = minidx_fallback;
    //     if (minidx_fallback == -1)
    //         fuzzy_vec_fallback.size = 0;
    //     Pack(fuzzy_vec_fallback, idx_img);
    // }
}

[shader("compute")]
[numthreads(gSLICr_BLOCK_DIM, gSLICr_BLOCK_DIM, 1)]
void ComputeMain(
    int3 DTid: SV_DispatchThreadID,
    int gid: SV_GroupIndex
) {
    if (any(DTid.xy >= img_size)) return;
    find_center_association(
        u_inImg, u_inSPixel, u_outIdxImg, map_size,
        img_size, spixel_size, weight, DTid.x, DTid.y,
        max_xy_dist, max_color_dist, gid);
}