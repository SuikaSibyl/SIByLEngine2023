#ifndef _SRENDERER_MEDIUM_HLSLI_
#define _SRENDERER_MEDIUM_HLSLI_

#include "common/math.hlsli"

namespace imedium {
struct tr_in {
    __init(float3 o, float3 d, float tMax) {
        this.o = o;
        this.d = d;
        this.tMax = tMax;
    }
    float3 o; // origin
    float3 d; // direction
    float tMax;
};
}

interface IMedium {

};

struct HomogeneousMediumData {
    float3 sigma_a;
    float3 sigma_s;
    float3 sigma_t;
    float g;
};

struct HomogeneousMedium : IMedium {

    static float3 Tr(imedium::tr_in i, HomogeneousMediumData data) {
        return exp(-data.sigma_t * min(i.tMax * length(i.d), k_inf));
    }

    // static float3 sample(HomogeneousMediumData data) {
    //     // Sample a channel and distance along the ray
    //     // int channel = min((int)(sampler.Get1D() * Spectrum::nSamples), 2);
    //     // float dist = -log(1 - sampler.Get1D()) / sigma_t[channel];
    //     // float t = min(dist * ray.d.Length(), ray.tMax);
    //     // bool sampledMedium = t < ray.tMax;
    //     // if (sampledMedium)
    //     //     *mi = MediumInteraction(ray(t), -ray.d, ray.time, this,
    //     //                             ARENA_ALLOC(arena, HenyeyGreenstein)(g));
    //     // Compute the transmittance and sampling density
    //     float3 Tr = exp(-data.sigma_t * min(t, k_inf) * ray.d.Length());
    // }
};

struct GridDensityMediumData {
    int nx; int ny; int nz;
};

// struct GridDensityMedium : IMedium {

//     static float density(float3 p, GridDensityMediumData data) {
//         // Compute voxel coordinates and offsets for p
//         const float3 pSamples = float3(p.x * data.nx - .5f,
//                                        p.y * data.ny - .5f, p.z * data.nz - .5f);
//         const int3 pi = (int3)floor(pSamples);
//         const float3 d = pSamples - (float3)pi;
//         // Trilinearly interpolate density values to compute local density
//         const float d00 = lerp(D(pi), D(pi + int3(1, 0, 0)), d.x);
//         const float d10 = lerp(D(pi + int3(0, 1, 0)), D(pi + int3(1, 1, 0)), d.x);
//         const float d01 = lerp(D(pi + int3(0, 0, 1)), D(pi + int3(1, 0, 1)), d.x);
//         const float d11 = lerp(D(pi + int3(0, 1, 1)), D(pi + int3(1, 1, 1)), d.x);
//         const float d0 = lerp(d00, d10, d.y);
//         const float d1 = lerp(d01, d11, d.y);
//         return lerp(d0, d1, d.z);
//     }

//     // static float D(int3 p, GridDensityMediumData data) {
//     //     Bounds3i sampleBounds(int3(0, 0, 0), int3(data.nx, data.ny, data.nz));
//     //     if (!InsideExclusive(p, sampleBounds))
//     //         return 0;
//     //     return density[(p.z * ny + p.y) * nx + p.x];
//     // }

//     static float3 Tr(imedium::tr_in i, HomogeneousMediumData data) {
//         return exp(-data.sigma_t * min(i.tMax * length(i.d), k_inf));
//     }
// };

#endif // _SRENDERER_MEDIUM_HLSLI_