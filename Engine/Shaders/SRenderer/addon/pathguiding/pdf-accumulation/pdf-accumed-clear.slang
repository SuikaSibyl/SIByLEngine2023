#include "../../../include/common/colormaps.hlsli"

RWTexture2D<uint> u_PdfAccumulator;
RWTexture2D<uint> u_PdfAccumulatorInfo;

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
        u_PdfAccumulator[pixel] = 0;
        if (all(pixel == 0)) {
            u_PdfAccumulatorInfo[pixel] = 0;
        }
    }
}