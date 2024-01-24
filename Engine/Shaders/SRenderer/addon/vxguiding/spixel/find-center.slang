#include "../../../include/common/cpp_compatible.hlsli"
#include "../../../include/common/math.hlsli"
#include "../../../include/common/octahedral.hlsli"

[[vk::push_constant]]
cbuffer PushConstants {
    int2 map_size;
    int2 img_size;
    int spixel_size;
    float weight;
    float max_xy_dist;
    float max_color_dist;
};

Texture2D<float4> u_input;
Texture2D<float4> u_center;

RWTexture2D<int> u_index;
RWTexture2D<int4> u_fuzzyIdx;
RWTexture2D<float4> u_fuzzyWeight;

RWTexture2D<uint> u_spixel_counter;  // 40 * 23
RWTexture2D<int2> u_spixel_gathered; // 1280 * 736 = (40*32) * (23*32)

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
            // if size smaller than 4,
            // just push back the id/dist
            center[size] = c;
            dist2[size] = d2;
            size++;
        } else {
            // find the slot with maximum distance,
            // displace it if the distance is smaller.
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
        u_fuzzyIdx[idx_img] = center;
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
    u_fuzzyIdx[idx_img] = center;
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
    const float sqrt_retval = retval;
    return float2((dot_normal > 0.01) ? sqrt_retval : 1000000, sqrt_retval);
}

groupshared float3 spixel_pos[3 * 3];
groupshared float3 spixel_normals[3 * 3];
groupshared float2 spixel_xy[3 * 3];
groupshared int spixel_ids[3 * 3];

groupshared uint gs_spixel_counter[9];
groupshared uint gs_spixel_offset[9];

int offset2idx(int2 offset) {
    return (offset.y + 1) * 3 + offset.x + 1;
}

void preload_mem(
    int n,
    out float3 spixel_pos_preload,
    out float3 spixel_normal_preload,
    out float2 spixel_xy_preload,
) {
    const int i = n / 3 - 1;
    const int j = n % 3 - 1;
    const int offset = offset2idx(int2(j, i));
    spixel_pos_preload = spixel_pos[offset];
    spixel_normal_preload = spixel_normals[offset];
    spixel_xy_preload = spixel_xy[offset];
}

void find_center_association(
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
    if (gid < 9) {
        const int2 offset = int2(gid % 3 - 1, gid / 3 - 1);
        const int idx = offset2idx(offset);
        int ctr_x_check = ctr_x + offset.x;
        int ctr_y_check = ctr_y + offset.y;
        if (ctr_x_check >= 0 && ctr_y_check >= 0 && ctr_x_check < map_size.x && ctr_y_check < map_size.y) {
            int ctr_idx = ctr_y_check * map_size.x + ctr_x_check;
            float4 center_info = u_center[int2(ctr_x_check, ctr_y_check)];
            spixel_pos[idx] = center_info.xyz;
            spixel_normals[idx] = Unorm32OctahedronToUnitVector(asuint(center_info.w));
            int img_x = ctr_x_check * spixel_size + spixel_size / 2;
            int img_y = ctr_y_check * spixel_size + spixel_size / 2;
            // modify the center if it is out of image boundary
            img_y = img_y >= img_size.y ? (ctr_y_check * spixel_size + img_size.y) / 2 : img_y;
            const int cluster_idx = ctr_y_check * map_size.x + ctr_x_check;
            spixel_xy[idx] = float2((float)img_x, (float)img_y);
            spixel_ids[idx] = cluster_idx;
        }

        gs_spixel_counter[gid] = 0;
        gs_spixel_offset[gid] = 0;
    }
    // load the pixel color
    const float4 pix_val = u_input[idx_img];
    const float3 pix_normal = Unorm32OctahedronToUnitVector(asuint(pix_val.w));
    
    int minidx = -1;
    float dist = 999999.9999f;

    int minidx_fallback = -1;
    float dist_fallback = 999999.9999f;

    FuzzyVec fuzzy_vec;
    FuzzyVec fuzzy_vec_fallback;
    fuzzy_vec.size = 0;
    fuzzy_vec_fallback.size = 0;

    GroupMemoryBarrierWithGroupSync();

    // search 3x3 neighborhood
    // TODO :: use shared memory to speed up
    float3 spixel_pos_neighbor;
    float3 spixel_normal_neighbor;
    float2 spixel_xy_neighbor;
    preload_mem(0, spixel_pos_neighbor, spixel_normal_neighbor, spixel_xy_neighbor);

    // preload
    float3 spixel_pos_preload;
    float3 spixel_normal_preload;
    float2 spixel_xy_preload;

    [unroll]
    for (int n = 0; n < 9; ++n) {
        const int i = n / 3 - 1;
        const int j = n % 3 - 1;
        int ctr_x_check = ctr_x + j;
        int ctr_y_check = ctr_y + i;
        
        if (n < 8)
            preload_mem(n + 1, spixel_pos_preload, spixel_normal_preload, spixel_xy_preload);

        const int offset = offset2idx(int2(j, i));
        int spixel_id = spixel_ids[offset];

        if (ctr_x_check >= 0 && ctr_y_check >= 0 && ctr_x_check < map_size.x && ctr_y_check < map_size.y) {
            int ctr_idx = ctr_y_check * map_size.x + ctr_x_check;
            float2 cdist = compute_slic_distance(
                pix_val.xyz, pix_normal, float2(x, y),
                spixel_pos_neighbor, spixel_normal_neighbor, spixel_xy_neighbor,
                weight, max_xy_dist, max_color_dist);
            if (cdist.x < dist) {
                dist = cdist.x;
                minidx = spixel_id;
            }
            if (cdist.y < dist_fallback) {
                dist_fallback = cdist.y;
                minidx_fallback = spixel_id;
            }
            fuzzy_vec.insert(ctr_idx, cdist.x);
            fuzzy_vec_fallback.insert(ctr_idx, cdist.y);
        }

        spixel_pos_neighbor = spixel_pos_preload;
        spixel_normal_neighbor = spixel_normal_preload;
        spixel_xy_neighbor = spixel_xy_preload;
    }

    bool is_valid = (minidx >= 0);
    int spixelID = is_valid ? minidx : minidx_fallback;
    u_index[idx_img] = spixelID;
    if (minidx_fallback == -1)
        fuzzy_vec_fallback.size = 0;
    Pack(is_valid ? fuzzy_vec : fuzzy_vec_fallback, idx_img);


    const int2 spixelID2D = int2(spixelID % map_size.x, spixelID / map_size.x);
    const int2 offset_sp = (spixelID2D - int2(ctr_x, ctr_y) + int2(1, 1));
    const int gs_sp_offset = offset_sp.x + offset_sp.y * 3;

    uint subtaskIDFlatten_gs;
    if (spixelID != -1) {
        InterlockedAdd(gs_spixel_counter[gs_sp_offset], 1, subtaskIDFlatten_gs);
    }

    GroupMemoryBarrierWithGroupSync();

    if (gid < 9) {
        const int gs_sp_offset = gid;
        const int2 offset_sp = int2(gs_sp_offset % 3, gs_sp_offset / 3) - int2(1, 1);
        const int2 spixelID2D = int2(ctr_x, ctr_y) + offset_sp;
        if (all(spixelID2D >= 0 && spixelID2D < map_size)) {
            InterlockedAdd(u_spixel_counter[spixelID2D], gs_spixel_counter[gs_sp_offset], gs_spixel_offset[gs_sp_offset]);
        }
    }

    GroupMemoryBarrierWithGroupSync();

    if (spixelID != -1) {
        const int2 taskOffset = spixelID2D * 32;
        uint subtaskIDFlatten;
        // InterlockedAdd(u_spixel_counter[spixelID2D], 1, subtaskIDFlatten);
        subtaskIDFlatten = gs_spixel_offset[gs_sp_offset] + subtaskIDFlatten_gs;
        if (subtaskIDFlatten < 1024) {
            const int2 subtaskID = int2(subtaskIDFlatten % 32, subtaskIDFlatten / 32);
            u_spixel_gathered[taskOffset + subtaskID] = int2(x, y);
        }
    }
}

[shader("compute")]
[numthreads(16, 16, 1)]
void ComputeMain(
    int3 DTid: SV_DispatchThreadID,
    int gid: SV_GroupIndex
) {
    if (any(DTid.xy >= img_size)) return;
    find_center_association(map_size,
        img_size, spixel_size, weight, DTid.x, DTid.y,
        max_xy_dist, max_color_dist, gid);
}

// [[vk::push_constant]]
// cbuffer PushConstants {
//     int2 map_size;
//     int2 img_size;
//     int spixel_size;
//     float weight;
//     float max_xy_dist;
//     float max_color_dist;
// };

// Texture2D<float4> u_input;
// Texture2D<float4> u_center;

// RWTexture2D<int> u_index;
// RWTexture2D<int4> u_fuzzyIdx;
// RWTexture2D<float4> u_fuzzyWeight;

// struct FuzzyVec {
//     int center[4];
//     float4 dist2;
//     int size = 0;
//     bool normal_valid;

//     int get_maximum() {
//         int max_idx = 0;
//         float max_dist = dist2[0];
//         for (int i = 1; i < 4; ++i) {
//             if (dist2[i] > max_dist) {
//                 max_dist = dist2[i];
//                 max_idx = i;
//             }
//         }
//         return max_idx;
//     }

//     [mutating]
//     void insert(int c, float2 cdist) {
//         float d2; bool c_normal_valid;
//         if (cdist.x < 999999.9999f) {
//             d2 = cdist.x;
//             c_normal_valid = true;
//         } else {
//             d2 = cdist.y;
//             c_normal_valid = false;
//         }
//         if (normal_valid == false && c_normal_valid == true) {
//             size = 0; normal_valid = true;
//         } else if (normal_valid == true && c_normal_valid == false) {
//             return;
//         }
//         if (size < 4) {
//             // if size smaller than 4, 
//             // just push back the id/dist
//             center[size] = c;
//             dist2[size] = d2;
//             size++;
//         } else {
//             // find the slot with maximum distance,
//             // displace it if the distance is smaller.
//             int max_idx = get_maximum();
//             if (d2 < dist2[max_idx]) {
//                 center[max_idx] = c;
//                 dist2[max_idx] = d2;
//             }
//         }
//     }
// };

// void Pack(in_ref(FuzzyVec) vec, int2 idx_img) {
//     float4 weight = float4(0, 0, 0, 0);
//     int4 center = int4(-1);
//     if (vec.size == 0) {
//         u_fuzzyIdx[idx_img] = center;
//         u_fuzzyWeight[idx_img] = weight;
//         return;
//     }
//     int zero_idx = -1;
//     for (int i = 0; i < vec.size; ++i) {
//         if (vec.dist2[i] == 0) zero_idx = i;
//         weight[i] = 1.f / vec.dist2[i];
//         center[i] = vec.center[i];
//     }
//     if (zero_idx != -1) {
//         // for (int i = 0; i < 4; ++i)
//         //     weight[i] = isnan(weight[i]) ? 1 : 0;
//         weight = float4(0);
//         weight[zero_idx] = 1;
//     }
//     weight /= dot(weight, float4(1));
//     u_fuzzyIdx[idx_img] = center;
//     u_fuzzyWeight[idx_img] = weight;
// }

// float2 compute_slic_distance(
//     in_ref(float3) pix_pos, in_ref(float3) pix_normal, in_ref(float2) pix_xy,
//     in_ref(float3) sp_pos, in_ref(float3) sp_normal, in_ref(float2) sp_xy,
//     float weight,
//     float normalizer_xy,
//     float normalizer_color
// ) {
//     float dot_normal = dot(pix_normal, sp_normal);
//     const float3 diff_color = pix_pos - sp_pos;
//     float dcolor = dot(diff_color, diff_color);
//     const float2 diff_xy = pix_xy - sp_xy;
//     const float dxy = dot(diff_xy, diff_xy);
//     const float retval = dcolor * normalizer_color + weight * dxy * normalizer_xy;
//     const float sqrt_retval = retval;
//     return float2((dot_normal > 0.01) ? sqrt_retval : 1000000, sqrt_retval);
// }

// groupshared float3 spixel_pos[3 * 3];
// groupshared float3 spixel_normals[3 * 3];
// groupshared float2 spixel_xy[3 * 3];
// groupshared int spixel_ids[3 * 3];

// int offset2idx(int2 offset) {
//     return (offset.y + 1) * 3 + offset.x + 1;
// }

// groupshared uint gs_test[8];

// [shader("compute")]
// [numthreads(16, 16, 1)]
// void ComputeMain(
//     int3 DTid: SV_DispatchThreadID,
//     int gid: SV_GroupIndex
// ) {
//     const int2 pixel = DTid.xy;
//     if (any(pixel >= img_size)) return;

//     // ctr x/y should be similar for the whole group
//     const int2 ctr_spixel = pixel / spixel_size;
//     // First, load all the 3x3 neighbor spixels info to shared memory
//     if (gid < 9) {
//         const int2 offset = int2(gid % 3 - 1, gid / 3 - 1);
//         const int2 ctr_check = ctr_spixel + offset;
//         // load spixels center information
//         float3 position = float3(0. / 0.);
//         float3 normal = float3(0. / 0.);
//         if (all(ctr_check) >= 0 && all(ctr_check < map_size)) {
//             const float4 center_info = u_center[ctr_check];
//             position = center_info.xyz;
//             normal = Unorm32OctahedronToUnitVector(asuint(center_info.w));
//         }
//         // compute superpixel index and center
//         const int cluster_idx = ctr_check.y * map_size.x + ctr_check.x;
//         int img_x = ctr_check.x * spixel_size + spixel_size / 2;
//         int img_y = ctr_check.y * spixel_size + spixel_size / 2;
//         img_x = img_x >= img_size.x ? (ctr_check.x * spixel_size + img_size.x) / 2 : img_x;
//         img_y = img_y >= img_size.y ? (ctr_check.y * spixel_size + img_size.y) / 2 : img_y;
//         // save the superpixel information
//         spixel_pos[gid] = position;
//         spixel_normals[gid] = normal;
//         spixel_xy[gid] = float2((float)img_x, (float)img_y);
//         spixel_ids[gid] = ctr_check.y * map_size.x + ctr_check.x;
//     }

//     // load the pixel information
//     const float4 pix_val = u_input[pixel];
//     const float3 pix_position = pix_val.xyz;
//     const float3 pix_normal = Unorm32OctahedronToUnitVector(asuint(pix_val.w));

//     // find the closest spixel with appropriate weights
//     FuzzyVec fuzzy_vec;
//     fuzzy_vec.normal_valid = false;
//     fuzzy_vec.size = 0;

//     GroupMemoryBarrierWithGroupSync();

//     // search 3x3 neighborhood
//     [ForceUnroll]
//     for (int n = 0; n < 9; ++n) {
//         // preload from shared memory
//         const int2 offset = int2(n % 3 - 1, n / 3 - 1);
//         const int2 ctr_check = ctr_spixel + offset;
//         const float3 spixel_pos_neighbor = spixel_pos[n];
//         const float3 spixel_normal_neighbor = spixel_normals[n];
//         const float2 spixel_xy_neighbor = spixel_xy[n];
//         const int spixel_id_neighbor = spixel_ids[n];
//         // compare the distance
//         if (all(ctr_check) >= 0 && all(ctr_check < map_size)) {
//             float2 cdist = compute_slic_distance(
//                 pix_position, pix_normal, float2(pixel),
//                 spixel_pos_neighbor, spixel_normal_neighbor, spixel_xy_neighbor,
//                 weight, max_xy_dist, max_color_dist);
//             fuzzy_vec.insert(spixel_id_neighbor, cdist);
//         }
//     }
//     const int minidx = (fuzzy_vec.size == 0) 
//         ? -1 : fuzzy_vec.center[maxDimension(fuzzy_vec.dist2)];
//     u_index[pixel] = minidx;
//     Pack(fuzzy_vec, pixel);
// }