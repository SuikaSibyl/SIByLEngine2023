#ifndef _SRENDERER_SAMPLING_PMF_HLSLI_
#define _SRENDERER_SAMPLING_PMF_HLSLI_

StructuredBuffer<float> se_cdf_buffer;

namespace piecewise_sample {
struct sample_out_1d {
    float x;
    float pdf;
    uint offset;
};

struct sample_out_2d {
    float2 x;
    float pdf;
    uint2 offset;
};
}

piecewise_sample::sample_out_1d sample_pmf_1d(
    float u, uint cdf_offset, uint cdf_size) {
    // Find surrounding CDF segments and offset
    int size = (int)cdf_size - 2;
    int first = 1;
    while (size > 0) {
        // Evaluate predicate at midpoint and update first and size
        int half = (uint)size >> 1;
        int middle = first + half;
        bool predResult = se_cdf_buffer[cdf_offset + middle] <= u;
        first = predResult ? middle + 1 : first;
        size = predResult ? size - (half + 1) : half;
    }
    int o = (uint)clamp((int)first - 1, 0, cdf_size - 2);
    // Compute offset along CDF segment
    float cdf_o = se_cdf_buffer[cdf_offset + o];
    float cdf_o1 = se_cdf_buffer[cdf_offset + o + 1];
    float du = u - cdf_o;
    if (cdf_o1 - cdf_o > 0)
        du /= cdf_o1 - cdf_o;
    piecewise_sample::sample_out_1d output;
    output.offset = o;
    output.pdf = max(cdf_o1 - cdf_o, 0) * (cdf_size - 1);
    output.x = lerp(0, 1, (o + du) / (cdf_size - 1));
    return output;
}

piecewise_sample::sample_out_2d sample_pmf_2d(
    float2 u, uint marginal_offset, uint marginal_size,
    uint condition_offset, uint condition_size) {
    // Sample marginal distribution
    piecewise_sample::sample_out_1d d1 = sample_pmf_1d(
        u.y, marginal_offset, marginal_size);
    piecewise_sample::sample_out_1d d0 = sample_pmf_1d(
        u.x, condition_offset + condition_size * d1.offset, condition_size);
    piecewise_sample::sample_out_2d output;
    output.offset = uint2(d0.offset, d1.offset);
    output.pdf = d0.pdf * d1.pdf;
    output.x = float2(d0.x, d1.x);
    return output;
}

#endif // _SRENDERER_SAMPLING_PMF_HLSLI_