#ifndef _SRENDERER_SAMPLING_OCTMAP_HLSLI_
#define _SRENDERER_SAMPLING_OCTMAP_HLSLI_

#include "common/mapping.hlsli"

struct Octmap8x8 {
    int offset;
    // fetch value from index
    float fetch_value(RWByteAddressBuffer octmap, int index) {
        return octmap.Load<float>((offset + index) * 4);
    }
    // normalize a size=4 pmf
    static float4 normalize_pmf4(float4 pmf) {
        float sum = pmf.x + pmf.y + pmf.z + pmf.w;
        return pmf / sum;
    }
    // fetch top level pmf
    float4 fetch_top_level_pmf(RWByteAddressBuffer octmap) {
        return normalize_pmf4(float4(
            fetch_value(octmap, 80 + 0),
            fetch_value(octmap, 80 + 1),
            fetch_value(octmap, 80 + 2),
            fetch_value(octmap, 80 + 3)
        ));
    }
    float fetch_top_level_sum(RWByteAddressBuffer octmap) {
        return dot(float4(
            fetch_value(octmap, 80 + 0),
            fetch_value(octmap, 80 + 1),
            fetch_value(octmap, 80 + 2),
            fetch_value(octmap, 80 + 3)
        ), float4(1));
    }
    // fetch second level pmf
    float4 fetch_snd_level_pmf(RWByteAddressBuffer octmap, int2 idx) {
        int x = idx.x * 2;
        int y = (1 - idx.y) * 2;
        int id = 4 * y + x;
        return normalize_pmf4(float4(
            fetch_value(octmap, 64 + id),
            fetch_value(octmap, 64 + id + 1),
            fetch_value(octmap, 64 + id + 4),
            fetch_value(octmap, 64 + id + 5)
        ));
    }
    // fetch third level pmf
    float4 fetch_thrd_level_pmf(RWByteAddressBuffer octmap, int2 idx) {
        int x = idx.x * 2;
        int y = (3 - idx.y) * 2;
        int id = 8 * y + x;
        return normalize_pmf4(float4(
            fetch_value(octmap, 0 + id),
            fetch_value(octmap, 0 + id + 1),
            fetch_value(octmap, 0 + id + 8),
            fetch_value(octmap, 0 + id + 9)
        ));
    }
    // sample from a 8x8 quad tree
    float3 sample_quadmap(RWByteAddressBuffer octmap, float2 uv) {
        float4 pmf = fetch_top_level_pmf(octmap);
        float pdf = Octmap8x8::warp_square_samples_to_pmf(pmf, uv);
        
        float4 grid = Octmap8x8::identify_grid(uv, 2);
        pmf = fetch_snd_level_pmf(octmap, int2(grid.xy));
        pdf *= Octmap8x8::warp_square_samples_to_pmf(pmf, uv);
        uv = Octmap8x8::recover_uv(grid, uv);

        grid = Octmap8x8::identify_grid(uv, 4);
        pmf = fetch_thrd_level_pmf(octmap, int2(grid.xy));
        pdf *= Octmap8x8::warp_square_samples_to_pmf(pmf, uv);
        uv = Octmap8x8::recover_uv(grid, uv);
        pdf *= 64;
        return float3(uv, pdf);
    }
    // warp square samples to pmf
    static float warp_square_samples_to_pmf(float4 pmf, inout float2 u) {
        float pdf = 1.f;
        float left = pmf[0] + pmf[2];
        float right = pmf[1] + pmf[3];
        if (u.x < left) {
            u.x = u.x * 0.5 / left;
            pdf *= left;
            float top = pmf[0] / (pmf[0] + pmf[2]);
            float bottom = pmf[2] / (pmf[0] + pmf[2]);
            if (u.y < bottom) {
                u.y = u.y * 0.5 / bottom;
                pdf *= bottom;
            } else {
                u.y = (u.y - bottom) * 0.5 / top + 0.5;
                pdf *= top;
            }
        } else {
            u.x = (u.x - left) * 0.5 / right + 0.5;
            pdf *= right;
            float top = pmf[1] / (pmf[1] + pmf[3]);
            float bottom = pmf[3] / (pmf[1] + pmf[3]);
            if (u.y < bottom) {
                u.y = u.y * 0.5 / bottom;
                pdf *= bottom;
            } else {
                u.y = (u.y - bottom) * 0.5 / top + 0.5;
                pdf *= top;
            }
        }
        return pdf;
    }
    // identify grid
    static float4 identify_grid(inout float2 uv, int resolution) {
        float grid_size = 1.f / resolution;
        float x = floor(uv.x / grid_size);
        float y = floor(uv.y / grid_size);
        uv.x = (uv.x - x * grid_size) / grid_size;
        uv.y = (uv.y - y * grid_size) / grid_size;
        return float4(x, y, grid_size, grid_size);
    }
    // recover uv
    static float2 recover_uv(float4 grid, float2 uv) {
        return float2((grid.x + uv.x) * grid.z, (grid.y + uv.y) * grid.w);
    }
};

#endif // _SRENDERER_SAMPLING_OCTMAP_HLSLI_