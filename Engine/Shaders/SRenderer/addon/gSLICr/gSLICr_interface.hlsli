#ifndef _SRENDERER_ADDON_GSLICR_INTERFACE_HEADER_
#define _SRENDERER_ADDON_GSLICR_INTERFACE_HEADER_

#ifndef gSLICr_BLOCK_DIM
#define gSLICr_BLOCK_DIM 16
#endif

struct spixel_info {
    float4 color_info;  // avg color of the superpixel
    float2 center;      // center (avg pixel pos) of the superpixel
    int no_pixels;      // number of pixels
    int id;             // superpixel id
};

#endif // !_SRENDERER_ADDON_GSLICR_INTERFACE_HEADER_