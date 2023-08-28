#include "../../include/common/cpp_compatible.hlsli"
#include "../../include/common/packing.hlsli"
#include "../../include/common/random.hlsli"
#include "asvgf.hlsli"

RWTexture2D<float4> u_Color;

const Texture2D<uint> u_HF;

[shader("compute")]
[numthreads(16, 16, 1)]
void ComputeMain(uint3 DTid: SV_DispatchThreadID) {
    const int2 ipos = int2(DTid.xy);
    u_Color[ipos] = float4(UnpackRGBE(u_HF[ipos]), 1.0f);
}