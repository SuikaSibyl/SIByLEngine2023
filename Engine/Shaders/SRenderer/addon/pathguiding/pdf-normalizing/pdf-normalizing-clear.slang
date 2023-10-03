#include "../../../include/common/colormaps.hlsli"

RWTexture2D<float> u_PdfNormalizing;
RWTexture2D<uint> u_PdfNormalizingInfo;

[[vk::push_constant]]
cbuffer PushConstants {
    int2 resolution;
    bool clear;
};

[shader("compute")]
[numthreads(16, 16, 1)]
void ComputeMain(int3 dtid: SV_DispatchThreadID) {
    const int2 pixel = dtid.xy;
    if (any(pixel >= resolution)) return;

    if (clear) {
        u_PdfNormalizing[pixel] = 0;
        if (all(pixel == 0)) {
            u_PdfNormalizingInfo[pixel] = 0;
        }
    }
}