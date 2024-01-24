
struct spixel_info {
    float4 color_info;  // avg color of the superpixel
    float2 center;      // center (avg pixel pos) of the superpixel
    int no_pixels;      // number of pixels
    int id;             // superpixel id
};
