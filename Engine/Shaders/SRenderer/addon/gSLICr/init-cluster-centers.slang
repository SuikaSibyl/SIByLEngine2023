#include "../../include/common/cpp_compatible.hlsli"
#include "gSLICr_interface.hlsli"

[[vk::push_constant]]
cbuffer PushConstants {
    int2 map_size;
    int2 img_size;
    int spixel_size;
};

Texture2D<float4> u_inImg;
RWStructuredBuffer<spixel_info> u_outSPixel;

/**
 * Use one thread per cluster center to initialize superpixel map.
 * This is an ns_r x ns_c image which contains, or each entry, center coordinates,
 * number of associated pixels and colour information.
 * ns_r and ns_c represent the number of superpixels per image row and column, respectively.
 * @param inimg Input image
 * @param out_spixel Output superpixel map
 * @param map_size Size of superpixel map
 * @param img_size Size of input image
 * @param spixel_size Size of superpixel
 * @param x X coordinate of thread / superpixel
 * @param y Y coordinate of thread / superpixel
 */
void init_cluster_centers(
    Texture2D<float4> inimg,
    RWStructuredBuffer<spixel_info> out_spixel,
    in_ref(int2) map_size,
    in_ref(int2) img_size, 
    int spixel_size, 
    int x, int y
) {
    const int cluster_idx = y * map_size.x + x;
    int img_x = x * spixel_size + spixel_size / 2;
    int img_y = y * spixel_size + spixel_size / 2;
    // modify the center if it is out of image boundary
    img_x = img_x >= img_size.x ? (x * spixel_size + img_size.x) / 2 : img_x;
    img_y = img_y >= img_size.y ? (y * spixel_size + img_size.y) / 2 : img_y;
    // initialize superpixel map
    out_spixel[cluster_idx].id = cluster_idx;
    out_spixel[cluster_idx].center = float2((float)img_x, (float)img_y);
    out_spixel[cluster_idx].color_info = inimg[int2(img_x, img_y)];
    out_spixel[cluster_idx].no_pixels = 0;
}

[shader("compute")]
[numthreads(gSLICr_BLOCK_DIM, gSLICr_BLOCK_DIM, 1)]
void ComputeMain(int3 DTid: SV_DispatchThreadID) {
    if (any(DTid.xy >= map_size)) return;
    init_cluster_centers(u_inImg, u_outSPixel, map_size, img_size, spixel_size, DTid.x, DTid.y);
}