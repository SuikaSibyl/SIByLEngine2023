#include "../gbuffer/gbuffer_interface.hlsli"
#include "../../include/common/cpp_compatible.hlsli"
#include "../../include/common/packing.hlsli"
#include "../../include/common/random.hlsli"
#include "asvgf.hlsli"

[[vk::push_constant]]
cbuffer PushConstantBuffer {
    int2 resolution;
    int spec_iteration;
    int rnd_seed;
}

// wavelet fitler sensitivity to depth, [0..inf)
static const float flt_atrous_depth = 0.5f;
// number of a-trous wavelet filter iterations on the HF channel, [0..4]
static const int flt_atrous_hf = 4;
// number of a-trous wavelet filter iterations on the SPEC channel, [0..4]
static const int flt_atrous_spec = 3;
// wavelet filter sensitivity to luminance, [0..inf)
static const float flt_atrous_lum_hf = 16;
// wavelet filter sensitivity to normals on the HF channel, [0..inf)
static const float flt_atrous_normal_hf = 64;
// wavelet filter sensitivity to normals on the SPEC channel, [0..inf)
static const float flt_atrous_normal_spec = 64;

// ========================================================================== //
// This is the a-trous wavelet filter for the HF (direct diffuse) channel.
// Multiple invocations of this shader with different values for `spec_iteration`
// will effectively compute a large-radius bilateral blur on the lighting data.
// The last invocation of this shader also performs image compositing,
// i.e. combining all lighting channels and applying the albedo texture.
// ========================================================================== //

const Texture2D<float4> u_MomentsHistlenHF;
const Texture2D<uint> s_Atrous_HF;
const Texture2D<uint> s_Atrous_Spec;
const Texture2D<float2> s_Atrous_Moments;

RWTexture2D<uint> u_Atrous_HF;
RWTexture2D<uint> u_Atrous_Spec;
RWTexture2D<float2> u_Atrous_Moments;
RWTexture2D<float4> u_Color;
RWTexture2D<uint> u_IsCorrelated;

// TEX_ASVGF_ATROUS_PING_HF, TEX_ASVGF_ATROUS_PING_SPEC, TEX_ASVGF_ATROUS_PING_MOMENTS

// Converts a square of roughness to a Phong specular power
float RoughnessSquareToSpecPower(in float alpha) {
    return max(0.01, 2.0f / (square(alpha) + 1e-4) - 2.0f);
}

// Filter kernel for the HF channel
void filter_image(
    in_ref(int2) ipos,
    in_ref(float2) rand,
    out_ref(float3) filtered_hf,
    out_ref(float3) filtered_spec,
    out_ref(float2) filtered_moments
) {
    const float3 color_center_hf = UnpackRGBE(s_Atrous_HF[ipos]);
    const float3 color_center_spec = UnpackRGBE(s_Atrous_Spec[ipos]);
    const float2 moments_center = s_Atrous_Moments[ipos].xy;

    // If exceed the maximum number of iterations, just do no filtering and return the input
    if (flt_atrous_hf <= spec_iteration && flt_atrous_spec <= spec_iteration) {
        filtered_hf = color_center_hf;
        filtered_spec = color_center_spec;
        filtered_moments = moments_center;
        return;
    }

    const float3 normal_center = GetGeometryNormal(ipos);
    const float depth_center = GetViewDepth(ipos);
    const float fwidth_depth = GetFWidthDepth(ipos);
    const float roughness_center = GetSpecularRoughness(ipos).w;

    float lum_mean_hf = 0;
    float sigma_l_hf = 0;

    const float hist_len_hf = u_MomentsHistlenHF[ipos].b;

    if (flt_atrous_lum_hf != 0 && hist_len_hf > 1) {
        // Compute luminance variance from the statistical moments: Var(X) = E[X^2] - E[X]^2
        // The `asvgf_temporal` shader computes a combination of temporal and spatial (3x3) moments,
        // and stores these into a texture. This shader will combine moments of the surrounding
        // pixels using the same weights as for colors, and the combined moments are used on the next iteration.
        lum_mean_hf = moments_center.x;
        float lum_variance_hf = max(1e-8, moments_center.y - moments_center.x * moments_center.x);
        sigma_l_hf = min(hist_len_hf, flt_atrous_lum_hf) / (2.0 * lum_variance_hf);
    }
    else {
        // If there is no history, treat all moments as invalid, because 3x3 spatial
        // is just not enough to get reasonable filtering. Ignore luminance in this case,
        // and perform a depth-normal-guided bilateral blur.
        sigma_l_hf = 0;
    }

    // Compute the normal weight for the HF & spec channel
    // reduce the HF filter sensitivity to normals when the lighting is invalidated
    const float normal_weight_scale = clamp(hist_len_hf / 8, 0, 1);
    const float normal_weight_hf = flt_atrous_normal_hf * normal_weight_scale;
    float normal_weight_spec = RoughnessSquareToSpecPower(square(roughness_center)) * flt_atrous_normal_spec;
    normal_weight_spec = clamp(normal_weight_spec, 8, 1024) * normal_weight_scale;

    const int step_size = int(1u << spec_iteration);

    float3 sum_color_hf = color_center_hf.rgb;
    float3 sum_color_spec = color_center_spec.rgb;
    float2 sum_moments = moments_center;

    float sum_w_hf = 1.0;
    float sum_w_spec = 1.0;

    // Add some jitter to sample positions to hide the a-trous filter aliasing patterns
    const int2 jitter = int2((rand - 0.5) * float(step_size));
    float spec_filter_width_scale = clamp(roughness_center * 30 - spec_iteration, 0, 1);

    // Compute the weighted average of color and moments from a sparse 3x3 pattern around the target pixel
    for (int yy = -1; yy <= 1; ++yy) {
        for (int xx = -1; xx <= 1; ++xx) {
            // p: position of the sampled pixel
            const int2 p = ipos + int2(xx, yy) * step_size + jitter;
            // skip the center pixel
            if (xx == 0 && yy == 0) continue;
            // w: weight of the sampled pixel, set to 0 if the pixel is outside the screen
            float w = (all(p >= int2(0, 0)) && all(p < resolution)) ? 1.0f : 0.0f;
            // load the information of the sampled pixel
            const float3 normal = GetGeometryNormal(p);
            const float depth = GetViewDepth(p);
            const float roughness = GetSpecularRoughness(p).a;
            // compute the edge-stopping functions
            float dist_z = abs(depth_center - depth) * fwidth_depth * flt_atrous_depth;
            w *= exp(-dist_z / float(step_size));   // times depth similarity
            w *= wavelet_kernel[abs(xx)][abs(yy)];  // times wavelet kernel

            float w_hf = w;
            if (spec_iteration == 0) {
                // remove the correlated pixels
                if (u_IsCorrelated[p] == 1)
                    w_hf = 0;
            }

            const float3 c_hf = UnpackRGBE(s_Atrous_HF[p]);
            const float3 c_spec = UnpackRGBE(s_Atrous_Spec[p]);
            const float2 c_mom = u_Atrous_Moments[p].xy;
            const float l_hf = luminance(c_hf.rgb);
            const float dist_l_hf = abs(lum_mean_hf - l_hf);

            w_hf *= exp(-dist_l_hf * dist_l_hf * sigma_l_hf);

            float w_spec = w_hf;
            w_spec *= max(0, 1 - 10 * abs(roughness - roughness_center));
            w_spec *= spec_filter_width_scale;

            // Apply normal weight to stopping function
            const float NdotN = saturate(dot(normal_center, normal));
            if (normal_weight_hf > 0)   w_hf *= pow(NdotN, normal_weight_hf);
            if (normal_weight_spec > 0) w_spec *= pow(NdotN, normal_weight_spec);
            // If the iteration pass is larger than the maximum iteration, the weight is set to 0
            if (flt_atrous_hf <= spec_iteration) w_hf = 0;
            if (flt_atrous_spec <= spec_iteration) w_spec = 0;
            // Accumulate the weighted color and moments
            sum_color_hf += c_hf.rgb * w_hf;
            sum_color_spec += c_spec.rgb * w_spec;
            sum_w_hf += w_hf;
            sum_w_spec += w_spec;
            sum_moments += c_mom * w_hf;
        }
    }

    filtered_hf = sum_color_hf / sum_w_hf;
    filtered_spec = sum_color_spec / sum_w_spec;
    filtered_moments = sum_moments / sum_w_hf;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void ComputeMain(uint3 DTid: SV_DispatchThreadID) {
    const int2 ipos = int2(DTid.xy);
    if (any((ipos >= resolution))) return;
    RandomSamplerState RNG = InitRandomSampler(ipos, rnd_seed);
    
    float3 filtered_hf;
    float3 filtered_spec;
    float2 filtered_moments;

    // filter the image if the iteration pass is smaller equal than the maximum iteration
    if (spec_iteration <= 3) {
        const float2 rnd = float2(GetNextRandom(RNG), GetNextRandom(RNG));
        filter_image(ipos, rnd, filtered_hf, filtered_spec, filtered_moments);
    }
    // save the filter image if the iteration pass is smaller than the maximum iteration
    if (spec_iteration <= 2) {
        u_Atrous_HF[ipos] = PackRGBE(filtered_hf);
        u_Atrous_Spec[ipos] = PackRGBE(filtered_spec);
        u_Atrous_Moments[ipos] = filtered_moments;
    }
    
    // Perform compositing on the last iteration
    if (spec_iteration == 3) {
        // // SH filtered_lf = interpolate_lf(TEX_ASVGF_ATROUS_PING_LF_SH, TEX_ASVGF_ATROUS_PING_LF_COCG, ipos);

        // // filtered_lf.shY /= STORAGE_SCALE_LF;
        // // filtered_lf.CoCg /= STORAGE_SCALE_LF;
        // filtered_hf /= STORAGE_SCALE_HF;
        // filtered_spec /= STORAGE_SCALE_SPEC;

        // const float3 normal = GetGeometryNormal(ipos);
        // const float3 base_color = GetDiffuseAlbedo(ipos);
        // const float4 metallic_roughness = GetSpecularRoughness(ipos);
        // float specular_factor = base_color.a;
        // float metallic = metallic_roughness.x;
        // float roughness = metallic_roughness.y;

        // // Load the other image channels
        // vec4 transparent = texelFetch(TEX_PT_TRANSPARENT, ipos, 0);
        // vec3 throughput = texelFetch(TEX_PT_THROUGHPUT, ipos, 0).rgb;
        const float view_depth = GetViewDepth(ipos);
        const float3 normal_center = GetDiffuseAlbedo(ipos);
        float3 final_color = float3(0);
        if (view_depth != k_background_depth) {
            final_color = filtered_hf * normal_center + filtered_spec;
        }
        
        // // Composite
        // vec3 final_color = composite_color(base_color.rgb, metallic, throughput, projected_lf, filtered_hf, filtered_spec, transparent);

        // if (global_ubo.flt_show_gradients != 0)
        // {
        //     // Debug visualization of gradients
        //     float gradient_lf = texelFetch(TEX_ASVGF_GRAD_LF_PONG, ipos / GRAD_DWN, 0).r;
        //     vec2 gradient_hf_spec = texelFetch(TEX_ASVGF_GRAD_HF_SPEC_PONG, ipos / GRAD_DWN, 0).rg;
        //     final_color.r += gradient_lf * global_ubo.flt_scale_lf;
        //     final_color.g += gradient_hf_spec.x * global_ubo.flt_scale_hf;
        //     final_color.b += gradient_hf_spec.y * global_ubo.flt_scale_spec;
        // }

        // final_color *= STORAGE_SCALE_HDR;
        u_IsCorrelated[ipos] = 0;
        u_Color[ipos] = float4(final_color, 1);
    }
}