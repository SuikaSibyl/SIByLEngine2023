#include "../../../include/common/octahedral.hlsli"
#include "../../../raytracer/spt_interface.hlsli"

RWStructuredBuffer<float4> u_RepresentPixel;
RWStructuredBuffer<float4> u_RepresentVPL;

[shader("compute")]
[numthreads(32, 8, 1)]
void ComputeMain(
    int3 DTid: SV_DispatchThreadID,
    int gidx: SV_GroupIndex,
) {
    
}