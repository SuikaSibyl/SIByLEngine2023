#include "../pathguiding/pdf-accumulation/pdf-accumulator.hlsli"
#include "include/GIReservoir.hlsli"

[[vk::push_constant]]
cbuffer PushConstants {
    GIResamplingRuntimeParameters runtimeParams;
    int2 pixel;
    uint shadeInputBufferIndex;
};

RWStructuredBuffer<PackedGIReservoir> u_GIReservoirs;

[shader("compute")]
[numthreads(1, 1, 1)] // 8x8x8 threads in a group
void ComputeMain(int3 dtid: SV_DispatchThreadID) {
    GIReservoir reservoir = LoadGIReservoir(
        runtimeParams, pixel, shadeInputBufferIndex, u_GIReservoirs);
    
    AddOneSample2Info();
    // AddOneSample2Map
}