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
    in_ref(float4) pix,
    int x, int y,
    in_ref(spixel_info) center_info,
    float weight,
    float normalizer_xy,
    float normalizer_color
) {
    const float3 pix_normal = Unorm32OctahedronToUnitVector(asuint(pix.w));
    const float3 sp_normal = Unorm32OctahedronToUnitVector(asuint(center_info.color_info.w));
    float dot_normal = dot(pix_normal, sp_normal);

    const float3 diff_color = pix.xyz - center_info.color_info.xyz;
    float dcolor = dot(diff_color, diff_color);
    const float2 diff_xy = float2(x, y) - center_info.center;
    const float dxy = dot(diff_xy, diff_xy);
    const float retval = dcolor * normalizer_color + weight * dxy * normalizer_xy;
    const float sqrt_retval = sqrt(retval);

    return float2((dot_normal > 0.01) ? sqrt_retval : 1000000, sqrt_retval);
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
    float max_color_dist
) {
    int2 idx_img = int2(x, y);

    int ctr_x = x / spixel_size;
    int ctr_y = y / spixel_size;

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
                float2 cdist = compute_slic_distance(inimg[idx_img], x, y, in_spixel_map[ctr_idx], weight, max_xy_dist, max_color_dist);
                if (cdist.x < dist) {
                    dist = cdist.x;
                    minidx = in_spixel_map[ctr_idx].id;
                }
                if(cdist.y < dist_fallback) {
                    dist_fallback = cdist.y;
                    minidx_fallback = in_spixel_map[ctr_idx].id;
                }
                fuzzy_vec.insert(ctr_idx, cdist.x);
                fuzzy_vec_fallback.insert(ctr_idx, cdist.y);
            }
        }
    if (minidx >= 0) {
        out_idx_img[idx_img] = minidx;
        Pack(fuzzy_vec, idx_img);
    }
    else {
        out_idx_img[idx_img] = minidx_fallback;
        if (minidx_fallback == -1)
            fuzzy_vec_fallback.size = 0;
        Pack(fuzzy_vec_fallback, idx_img);
    }
}

[shader("compute")]
[numthreads(gSLICr_BLOCK_DIM, gSLICr_BLOCK_DIM, 1)]
void ComputeMain(int3 DTid: SV_DispatchThreadID) {
    if (any(DTid.xy >= img_size)) return;
    find_center_association(
        u_inImg, u_inSPixel, u_outIdxImg, map_size,
        img_size, spixel_size, weight, DTid.x, DTid.y,
        max_xy_dist, max_color_dist);
}