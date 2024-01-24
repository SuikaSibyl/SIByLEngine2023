#include "../../include/common/random.hlsli"
#include "../../include/common/gaussian.hlsli"

RWTexture2D<float2> u_uxy;
RWTexture2D<uint32_t> u_cylinder;
RWStructuredBuffer<float> u_lattice;

[[vk::push_constant]]
cbuffer PushConstants {
    uint initialize;
    uint iteration;
};

static const int Nx = 400;
static const int Ny = 100;
static const float tau = 0.6f;
static const int rho0 = 100;

// velocity of the grids
// ---------
// | 8 1 2 |
// | 7 0 3 |
// | 6 5 4 |
// ---------
static const int cxs[9] = { 0, 0, 1, 1, 1, 0, -1, -1, -1 };
static const int cys[9] = { 0, 1, 1, 0, -1, -1, -1, 0, 1 };
static const float weights[9] = { 4.f / 9, 1.f / 9, 1.f / 36,
                                  1.f / 9, 1.f / 36, 1.f / 9,
                                  1.f / 36, 1.f / 9, 1.f / 36 };
static const int opposite[9] = { 0, 5, 6, 7, 8, 1, 2, 3, 4 };

float loadF_global(int2 index, int idx) {
    if (index.y < 0) index.y += Ny;
    if (index.x < 0) index.x += Nx;
    if (index.y >= Ny) index.y -= Ny;
    if (index.x >= Nx) index.x -= Nx;
    
    uint offset = (index.y * Nx + index.x) * 9;
    return u_lattice[offset + idx];
}

void setF_global(int2 index, float F[9]) {
    uint offset = (index.y * Nx + index.x) * 9;
    for (int i = 0; i < 9; ++i)
        u_lattice[offset + i] = F[i];
}

float gaussian(inout RandomSamplerState RNG) {
    return BoxMuller(GetNextRandomFloat2(RNG)).x;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void ComputeMain(uint3 DTid: SV_DispatchThreadID) {
    const int2 pos = int2(DTid.xy);
    if (any((pos >= int2(400, 100)))) return;

    float F[9];

    // do initialization
    if (initialize == 1) {
        // initialize the collisiong information
        if (square(pos.x - Nx / 4) + square(pos.y - Ny / 2) < square(Ny / 4)) {
            u_cylinder[pos] = 1;
        } else {
            u_cylinder[pos] = 0;
        }

        RandomSamplerState RNG = InitRandomSampler(pos, 13);
        // initialize the fluid status
        for (int i = 0; i < 9; ++i) F[i] = 1.f + 0.01 * gaussian(RNG);
        // F[3] += 2 * (1 + 0.2 * cos(2 * k_pi * pos.x / Nx * 4));
        F[3] = 2.3;
        // float rho = 0.f;
        // for (int i = 0; i < 9; ++i) rho += F[i];
        // for (int i = 0; i < 9; ++i) F[i] *= rho0 / rho;
        setF_global(pos, F);
        return;
    }

    // Drift
    for (int i = 0; i < 9; ++i) {
        F[i] = loadF_global(pos - int2(cxs[i], cys[i]), i);
    }

    const bool is_collided = (u_cylinder[pos] == 1);
    float bndryF[9];
    // invert the velocity if is collided
    if (is_collided) {
        for (int i = 0; i < 9; ++i)
            bndryF[i] = F[opposite[i]];
    }

    // Calculate fluid variables
    float rho = 0.f; float ux = 0.f; float uy = 0.f;
    for (int i = 0; i < 9; ++i) {
        rho += F[i];
        ux += F[i] * cxs[i];
        uy += F[i] * cys[i];
    }
    ux /= rho;
    uy /= rho;

    if (is_collided) {
        ux = 0.f;
        uy = 0.f;
        for (int i = 0; i < 9; ++i) 
            F[i] = bndryF[i];
    }

    // Apply Collision
    for (int i = 0; i < 9; ++i) {
        // idxs, cxs, cys, weights
        float Feq = rho * weights[i] * (1 + 3 * (cxs[i] * ux + cys[i] * uy)
            + 9 * square(cxs[i] * ux + cys[i] * uy) / 2 - 3 * (square(ux) + square(uy)) / 2);
        F[i] += -(1.0 / tau) * (F[i] - Feq);
    }
    
    setF_global(pos, F);
    u_uxy[pos] = float2(ux, uy);
}