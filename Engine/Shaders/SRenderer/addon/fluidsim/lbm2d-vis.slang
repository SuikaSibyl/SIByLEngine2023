#include "../../include/common/colormaps.hlsli"

const Texture2D<float2> u_uxy;
const Texture2D<uint32_t> u_cylinder;

RWTexture2D<float4> u_color;

static const int Nx = 400;
static const int Ny = 100;
float4 load_uxy_global(int2 index) {
    if (index.y < 0) index.y += Ny;
    if (index.x < 0) index.x += Nx;
    if (index.y >= Ny) index.y -= Ny;
    if (index.x >= Nx) index.x -= Nx;
    return u_color[index];
}

float3 bwr(float t) {
    return float3(t, 1, (1 - t));
}

float2 load_uxy(int2 i) {
    if (i.x < 0) i.x = 0;
    if (i.x >= Nx) i.x = Nx - 1;
    if (i.y < 0) i.y = 0;
    if (i.y >= Ny) i.y = Ny - 1;
    return u_uxy[i];
}

[shader("compute")]
[numthreads(16, 16, 1)]
void ComputeMain(uint3 DTid: SV_DispatchThreadID) {
    const int2 pos = int2(DTid.xy);
    if (any((pos >= int2(400, 100)))) return;

    uint cylinder = u_cylinder[pos];
    float2 uxy = u_uxy[pos];

    float vorticity = (load_uxy(pos + int2(0, +1)).x - load_uxy(pos + int2(0, -1)).x) 
        - (load_uxy(pos + int2(+1, 0)).y - load_uxy(pos + int2(-1, 0)).y);

    vorticity = clamp(vorticity, -1, 1) * 5 + 0.5;
    u_color[pos] = (cylinder == 1) ? float4(1, 0, 1, 1) : float4(viridis(vorticity), 1.f);
}