#include "../../gbuffer/gbuffer_interface.hlsli"
#include "../../include/common/cpp_compatible.hlsli"
#include "../../include/common/packing.hlsli"
#include "asvgf.hlsli" // see description and copyright info here

[[vk::push_constant]]
cbuffer PushConstantBuffer {
    int2 resolution;
    int iteration;
}

// output grad image
RWTexture2D<float2> u_grad_HF_SPEC;
// input
const Texture2D<uint> u_GradSamplePos;
const Texture2D<float2> u_HFSpecLum_prev;
const Texture2D<uint> u_HF;
const Texture2D<uint> u_Spec;

#ifdef USE_LF_FILTERING
// output grad image
RWTexture2D<float2> u_grad_LF;
// input
const Texture2D<float4> u_LF_SH;
const Texture2D<float4> u_LF_SH_prev;
#endif

/**
 * Computes the gradient for a given pixel, based on the current and previous frame.
 * Gradient for HF and SPEC channels is computed as the relative difference between
 * path tracer outputs on the current and previous frame, for a given gradient pixel.
 * @param l_curr The current frame's path tracer output for a given pixel.
 * @param l_prev The previous frame's path tracer output for a given pixel.
 * @return The gradient for a given pixel.
 */
float get_gradient(float l_curr, float l_prev) {
    float l_max = max(l_curr, l_prev);
    if (l_max == 0) return 0;
    float ret = abs(l_curr - l_prev) / l_max;
    ret *= ret; // make small changes less significant
    return ret;
}

/**
 * Gradient for the LF channel is computed as the relative difference in average brightness
 * of all pixels in a very large screen region. Also, instead of the raw path tracer output
 * from the previous frame, the temporally accumulated LF color is used to continuously
 * detect invalid history and not miss some sudden flashes.
 * The approach that's used for HF doesn't work here because the LF signal is very sparse,
 * so the chance of a gradient pixel getting a different value is very low - because in many cases,
 * both the current and the previous pixels will be black.
 * @param ipos The position of the pixel.
 * @return The previous and current color fetched for filtering.
 */
#ifdef USE_LF_FILTERING
float2 get_lf_gradient(in_ref(int2) ipos) {
    // Find the same surface on the pvreious frame
    const float2 motion = GetMotionVector(ipos).xy;
    const int2 pos_prev = ipos + int2(motion.xy);

    // Ignore if the surface was outside of the screen
    if (any(pos_prev < 0) || any(pos_prev >= resolution))
        return float2(0, 0);

    // Get the current path tracer output and the temporally accumulated history.
    // Ignore disocclusion, doesn't seem to be necessary here as there is a huge blur pass
    // done on the LF gradient samples after.
    float lum_curr = u_LF_SH[ipos].w;
    float lum_prev = u_LF_SH_prev[pos_prev].w;

    // Return raw colors, do not divide until after the blur pass. We want to detect
    // brightness changes over large parts of the screen to avoid noise.
    return float2(lum_curr, lum_prev);
}
#endif

[shader("compute")]
[numthreads(256, 1, 1)]
void ComputeMain(int3 DTid: SV_DispatchThreadID) {
    // Create the gradient image
    const int2 ipos = int2(DTid.xy);
    const int2 offset = ipos * GRAD_DWN;
    if (any(ipos >= (resolution / GRAD_DWN)))
        return;

    const uint u = u_GradSamplePos[ipos];

    float2 grad_lf = float2(0);
    float grad_hf = 0;
    float grad_spec = 0;

    // Process reprojected HF and SPEC samples
    if (u != 0u) {
        // position of sample inside of stratum in the current frame
        const int2 grad_strata_pos = UnpackStratumOffset(u);
        // full position in current frame for gradient sample
        const int2 grad_sample_pos_curr = offset + grad_strata_pos;
        // hf and spec luminance from previous frame
        const float2 prev_hf_spec_lum = u_HFSpecLum_prev[ipos];
        // hf and spec luminance from current frame
        const float3 curr_hf = UnpackRGBE(u_HF[grad_sample_pos_curr]);
        const float3 curr_spec = UnpackRGBE(u_Spec[grad_sample_pos_curr]);
        // compute gradients for hf and spec
        grad_hf = get_gradient(luminance(curr_hf), prev_hf_spec_lum.x);
        grad_spec = get_gradient(luminance(curr_spec), prev_hf_spec_lum.y);
    }

    // save the gradient to image
    u_grad_HF_SPEC[ipos] = float2(grad_hf, grad_spec);

#ifdef USE_LF_FILTERING
    // Process all LF samples in the 3x3 square, accumulate the luminances
    for (int yy = 0; yy < GRAD_DWN; yy++) {
        for (int xx = 0; xx < GRAD_DWN; xx++) {
            grad_lf += get_lf_gradient(offset + int2(xx, yy));
        }
    }
    // save the gradient to image
    u_grad_LF[ipos] = grad_lf;
#endif

}