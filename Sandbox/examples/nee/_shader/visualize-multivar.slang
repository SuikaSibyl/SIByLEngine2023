#include "common/colormaps.hlsli"

[[vk::push_constant]] cbuffer PushConstants {
    int estimator_type;
    int num_samples;
};

RWTexture2D<float4> u_image;
RWTexture2D<float4> u_image_cv;
RWTexture2D<float4> u_output;

[shader("compute")]
[numthreads(16, 16, 1)]
void ComputeMain(int3 dtid: SV_DispatchThreadID) {
    const int2 texel = int2(dtid.xy);
    const float3 color = u_image[texel].rgb;
    float3 color_cv = u_image_cv[texel].rgb;

    int spp = max(num_samples, 1);
    // residual control variates
    if (estimator_type == 6) {
        u_output[texel] = float4((color + color_cv) / spp, 1.0);
    }
    // ratio control variates
    else if (estimator_type == 7) {
        if (color_cv.r == 0) color_cv.r = spp;
        if (color_cv.g == 0) color_cv.g = spp;
        if (color_cv.b == 0) color_cv.b = spp;
        u_output[texel] = float4((color / color_cv), 1.0);
    }
    else {
        u_output[texel] = float4(color / spp, 1.0);
    }
}