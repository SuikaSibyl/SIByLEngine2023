#include "../../../include/common/cpp_compatible.hlsli"
#include "../../../include/common/packing.hlsli"
#include "../../../include/common/random.hlsli"

[[vk::push_constant]]
cbuffer PushConstants {
    int2 img_size;
    int mode;
};

Texture2D<int> u_idxImg;
RWTexture2D<float4> u_color;

Texture2D<float4> u_fuzzyWeight;
Texture2D<int4> u_fuzzyIDX;

/**
 * Draw superpixel boundry.
 * Set a pixel to red color if it is a boundry pixel.
 * Here we assume that if a 4-connected pixel is not in the same superpixel,
 * it is a boundry pixel.
 */
void draw_superpixel_boundry(
    Texture2D<int> idx_img,
    RWTexture2D<float4> img,
    int2 img_size, int x, int y
) {
    const int2 idx = int2(x, y);

    if (mode == 0) {
        if (idx_img[idx] == -1) {
            img[idx] = float4(1, 0, 1, 1);
            return;
        }

        if (idx_img[idx] != idx_img[idx + int2(+1, +0)]
            || idx_img[idx] != idx_img[idx + int2(-1, +0)]
            || idx_img[idx] != idx_img[idx + int2(+0, +1)]
            || idx_img[idx] != idx_img[idx + int2(+0, -1)]) {
            img[idx] = float4(1, 0, 0, 1);
        }
    }
    else if (mode == 1) {
        const int id = idx_img[idx];
        const float3 color = img[idx].xyz;
        uint colorPacked = Hash_murmur(id);
        const float3 cluster_color = UNPACK_FLOAT3_UINT(colorPacked);
        img[idx] = float4(cluster_color, 1);
    }
    else {
        float4 weights = u_fuzzyWeight[idx];
        int4 ids = u_fuzzyIDX[idx];

        float3 color = 0;
        for (int i = 0; i < 4; i++) {
            if (weights[i] > 0) {
                uint colorPacked = Hash_murmur(ids[i]);
                const float3 cluster_color = UNPACK_FLOAT3_UINT(colorPacked);
                color += weights[i] * cluster_color;
            }
        }
        img[idx] = float4(color, 1);
    }
}

[shader("compute")]
[numthreads(16, 16, 1)]
void ComputeMain(int3 DTid: SV_DispatchThreadID) {
    if (any((DTid.xy == int2(0)) || (DTid.xy > img_size - int2(2))))
        return;

    draw_superpixel_boundry(
        u_idxImg,
        u_color,
        img_size,
        DTid.x,
        DTid.y);
}