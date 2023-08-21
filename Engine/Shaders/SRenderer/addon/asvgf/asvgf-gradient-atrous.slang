#include "../../include/common/cpp_compatible.hlsli"
#include "asvgf.hlsli" // see description and copyright info here

[[vk::push_constant]]
cbuffer PushConstantBuffer {
    int2 resolution;
    int iteration;
}

// output grad image
RWTexture2D<float2> u_grad_HF_SPEC;
const Texture2D<float2> u_grad_HF_SPEC_prev;

#ifdef USE_LF_FILTERING
// output grad image
RWTexture2D<float2> u_grad_LF;
const Texture2D<float2> u_grad_LF_prev;
/** Computes the gradient for a given pixel, based on the current and previous frame. */
float get_gradient(float l_curr, float l_prev) {
    float l_max = max(l_curr, l_prev);
    if (l_max == 0) return 0;
    float ret = abs(l_curr - l_prev) / l_max;
    ret *= ret; // make small changes less significant
    return ret;
}
#endif

// Blur the gradients
float2 filter_image(
    in_ref(int2) ipos,
    in_ref(Texture2D<float2>) img
) {
    // the size of the gradient image
    const int2 grad_size = int2(resolution) / GRAD_DWN;
    // the value of the center pixel
    const float2 color_center = img[ipos].xy;
    // the step size of the atrous wavelet kernel
    const int step_size = int(1u << iteration);
    // weighted sum the pixels in the kernel
    float2 sum_color = float2(0.f); // accumulator of values
    float sum_w = 0.f;              // accumulator of weights
    // loop over the kernel
    for (int yy = -1; yy <= 1; yy++) {
        for (int xx = -1; xx <= 1; xx++) {
            const int2 p = ipos + int2(xx, yy) * step_size;
            const float2 c = (all(p < grad_size) && all(p >= 0))
                                 ? img[p].xy
                                 : float2(0.f); // if out of bounds, use 0
            const float w = wavelet_kernel[abs(xx)][abs(yy)];
            sum_color += c * w;
            sum_w += w;
        }
    }
    // normalize by the sum of weights
    sum_color /= sum_w;
    return sum_color;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void ComputeMain(int3 DTid: SV_DispatchThreadID) {
    const int2 ipos = DTid.xy;
    const int2 grad_size = resolution / GRAD_DWN;
    if (any(ipos >= grad_size)) return;

    if (iteration <= 2) {
        float2 filtered_hf_spec = filter_image(ipos, u_grad_HF_SPEC_prev);
        u_grad_HF_SPEC[ipos] = filtered_hf_spec;
    }

#ifdef USE_LF_FILTERING
    float2 filtered_lf = filter_image(ipos, u_grad_LF_prev);
    if (iteration <= 6) {
        filtered_lf.x = get_gradient(filtered_lf.x, filtered_lf.y);
        filtered_lf.y = 0;
    }
    u_grad_LF[ipos] = filtered_lf;
#endif
}