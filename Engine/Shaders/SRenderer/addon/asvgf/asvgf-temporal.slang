#include "../gbuffer/gbuffer_interface.hlsli"
#include "../gbuffer/gbuffer_prev_interface.hlsli"
#include "../../include/common/cpp_compatible.hlsli"
#include "../../include/scene_descriptor_set.hlsli"
#include "asvgf.hlsli" // see description and copyright info here

/**
 * This is the temporal filter part of all denoisers.
 * Computes a weighted average between the current output of the path tracer
 * and the history for the same surfaces. Also downsamples the LF channel
 * into a 1/3 resolution version for further denoising.
 */

#define GROUP_SIZE 15
// spatially compute variance in a 3x3 (radius = 1) or a 5x5 (radius = 2) window
#define FILTER_RADIUS 1
// size of the shared memory copies of color, depth, and normals
#define SHARED_SIZE (GROUP_SIZE + FILTER_RADIUS * 2)

static const int flt_antilag_hf = 1;  // A-SVGF anti-lag filter strength, [0..inf)
static const int flt_temporal_hf = 1; // temporal filter strength, [0..1]
static const int flt_antilag_spec = 2;
static const int flt_temporal_spec = 1;
static const float flt_antilag_spec_motion = 0.004; // scaler for motion vector scaled specular anti-blur adjustment
static const float flt_min_alpha_color_hf = 0.02;   // minimum weight for the new frame data, color channel, (0..1]
static const float flt_min_alpha_moments_hf = 0.01; // minimum weight for the new frame data, moments channel, (0..1]
static const float flt_min_alpha_color_spec = 0.01;

RWTexture2D<float4> u_MomentsHistlenHF;
RWTexture2D<float4> u_ColorHistlenSpec;
RWTexture2D<uint> u_AtrousHF;
RWTexture2D<uint> u_AtrousSpec;
RWTexture2D<float2> u_AtrousMoments;
RWTexture2D<float4> u_Debug;

const Texture2D<float4> u_MomentsHistlenHF_Prev;
const Texture2D<float4> u_ColorHistlenSpec_Prev;
const Texture2D<float2> u_GradHFSpec;
const Texture2D<uint> u_HF;
const Texture2D<uint> u_Spec;
const Texture2D<uint> u_HFFiltered_Prev;

// Store some color data in shared memory for efficient access and for downsampling
groupshared float4 s_normal_lum[SHARED_SIZE][SHARED_SIZE];
groupshared float s_depth[SHARED_SIZE][SHARED_SIZE];

/** Preload the color data into shared memory */
void preload(in_ref(int2) groupID, int localID) {
    const int2 groupBase = groupID * GROUP_SIZE - FILTER_RADIUS;
    // The size of these shared memory buffers is larger than the group size because we
    // use them for some spatial filtering. So a single load per thread is not enough.
    // Rename the threads so that some of them will load 2 pixels, and most will load 1 pixel,
    // in the most dense way possible.
    for (int linear_idx = localID; linear_idx < SHARED_SIZE * SHARED_SIZE; linear_idx += GROUP_SIZE * GROUP_SIZE) {
        // Convert the linear index to 2D index in a SHARED_SIZE x SHARED_SIZE virtual group
        const float t = (float(linear_idx) + 0.5) / float(SHARED_SIZE);
        const int xx = int(floor(frac(t) * float(SHARED_SIZE)));
        const int yy = int(floor(t));
        // Load
        const int2 ipos = groupBase + int2(xx, yy);
        const float depth = GetViewDepth(ipos);
        const float3 normal = GetGeometryNormal(ipos);
        const float3 color_hf = UnpackRGBE(u_HF[ipos]);
        // Store
        s_normal_lum[yy][xx] = (float4(normal.xyz, luminance(color_hf.rgb)));
        s_depth[yy][xx] = depth;
    }
}

/** Load the color and normal data from shared memory */
void get_shared_data(
    in_ref(int2) local_id,
    in_ref(int2) offset,
    out_ref(float) depth,
    out_ref(float3) normal,
    out_ref(float) lum_hf
) {
    const int2 addr = local_id + int2(FILTER_RADIUS) + offset;
    depth = s_depth[addr.y][addr.x];
    const float4 unpack = s_normal_lum[addr.y][addr.x];
    normal = unpack.xyz;
    lum_hf = unpack.w;
}

[shader("compute")]
[numthreads(GROUP_SIZE, GROUP_SIZE, 1)]
void ComputeMain(
    int3 DTid: SV_DispatchThreadID,
    int3 GTid: SV_GroupThreadID,
    int3 Gid: SV_GroupID,
    int Gidx: SV_GroupIndex
) {
    preload(Gid.xy, Gidx);
    GroupMemoryBarrierWithGroupSync();

    const int2 ipos = DTid.xy;
    const int2 local_id = GTid.xy;
    const float4 motion = t_MotionVectors[ipos];
    const float2 pos_prev = ipos + motion.xy + float2(0.5);
    const float motion_length = length(motion.xy);
    const int2 resolution = getViewportSize(globalUniform.cameraData);

    // Load the parameters of the target pixel
    float depth_curr;
    float3 normal_curr;
    float lum_curr_hf;
    get_shared_data(local_id, int2(0), depth_curr, normal_curr, lum_curr_hf);

    const float4 metal_rough = GetSpecularRoughness(ipos);
    const float shininess = clamp(2.0 / square(square(metal_rough.w)) - 2.0, 0.0, 32.0);
    const float3 geo_normal_curr = GetGeometryNormal(ipos);

    // Try to get the history sample for all channels, including HF moments
    bool temporal_sample_valid_diff = false;
    bool temporal_sample_valid_spec = false;
    float3 temporal_color_hf = float3(0);
    float4 temporal_color_histlen_spec = float4(0);
    float4 temporal_moments_histlen_hf = float4(0);

    {
        float temporal_sum_w_diff = 0.0;
        float temporal_sum_w_spec = 0.0;

        float2 pos_ld = floor(pos_prev - float2(0.5));
        float2 subpix = frac(pos_prev - float2(0.5) - pos_ld);

        // Bilinear/bilateral filter
        const int2 off[4] = { { 0, 0 }, { 1, 0 }, { 0, 1 }, { 1, 1 } };
        float w[4] = {
            (1.0 - subpix.x) * (1.0 - subpix.y),
            (subpix.x) * (1.0 - subpix.y),
            (1.0 - subpix.x) * (subpix.y),
            (subpix.x) * (subpix.y)
        };

        for (int i = 0; i < 4; i++) {
            const int2 p = int2(pos_ld) + off[i];
            if (any(p < 0 || p >= resolution))  continue;
            
            const float depth_prev = GetViewDepthPrev(p);
            const float3 normal_prev = GetNormalPrev(p);
            const float3 geo_normal_prev = GetGeometryNormalPrev(p);

            float dist_depth = abs(depth_curr - depth_prev + motion.z) / abs(depth_curr);
            float dot_normals = dot(normal_curr, normal_prev);
            float dot_geo_normals = dot(geo_normal_curr, geo_normal_prev);

            if (depth_curr < 0) {
                // Reduce the filter sensitivity to depth for secondary surfaces,
                // because reflection/refraction motion vectors are often inaccurate.
                dist_depth *= 0.25;
            }

            if (dist_depth < 0.1 && dot_geo_normals > 0.5) {
                float w_diff = w[i] * max(dot_normals, 0);
                float w_spec = w[i] * pow(max(dot_normals, 0), shininess);

                temporal_color_hf += UnpackRGBE(u_HFFiltered_Prev[p]) * w_diff;
                temporal_color_histlen_spec += u_ColorHistlenSpec_Prev[p] * w_spec;
                temporal_moments_histlen_hf += u_MomentsHistlenHF_Prev[p].rgba * w_diff;
                temporal_sum_w_diff += w_diff;
                temporal_sum_w_spec += w_spec;
            }
        }

        // We found some relevant surfaces - good
        if (temporal_sum_w_diff > 1e-6) {
            float inv_w_diff = 1.0 / temporal_sum_w_diff;
            temporal_color_hf *= inv_w_diff;
            temporal_moments_histlen_hf *= inv_w_diff;
            temporal_sample_valid_diff = true;
        }

        if (temporal_sum_w_spec > 1e-6) {
            float inv_w_spec = 1.0 / temporal_sum_w_spec;
            temporal_color_histlen_spec *= inv_w_spec;
            temporal_sample_valid_spec = true;
        }
    }

    // Compute spatial moments of the HF channel in a 3x3 window
    float2 spatial_moments_hf = float2(lum_curr_hf, lum_curr_hf * lum_curr_hf);

    {
        float spatial_sum_w_hf = 1.0;
        for (int yy = -FILTER_RADIUS; yy <= FILTER_RADIUS; yy++) {
            for (int xx = -FILTER_RADIUS; xx <= FILTER_RADIUS; xx++) {
                if (xx == 0 && yy == 0) continue;
                int2 p = ipos + int2(xx, yy);
                
                float depth;
                float3 normal;
                float lum_p_hf;
                get_shared_data(local_id, int2(xx, yy), depth, normal, lum_p_hf);

                float dist_z = abs(depth_curr - depth) * motion.a;
                if (dist_z < 2.0) {
                    const float w_hf = pow(max(0.0, dot(normal, normal_curr)), 128.0);
                    spatial_moments_hf += float2(lum_p_hf * w_hf, lum_p_hf * lum_p_hf * w_hf);
                    spatial_sum_w_hf += w_hf;
                }
            }
        }

        float inv_w2_hf = 1.0 / spatial_sum_w_hf;
        spatial_moments_hf *= inv_w2_hf;
    }

    // Load the target pixel colors for all channels
    const float3 color_curr_hf = UnpackRGBE(u_HF[ipos]);
    const float3 color_curr_spec = UnpackRGBE(u_Spec[ipos]);

    float3 out_color_hf;
    float4 out_color_histlen_spec;
    float4 out_moments_histlen_hf;

    // Load the gradients
    float2 grad_hf_spec = u_GradHFSpec[ipos / GRAD_DWN].rg;
    grad_hf_spec = clamp(grad_hf_spec, float2(0), float2(1));

    if (temporal_sample_valid_diff) {
        // Compute the antilag factors based on the gradients
        float antilag_alpha_hf = clamp(lerp(1.0, flt_antilag_hf * grad_hf_spec.x, flt_temporal_hf), 0, 1);
        // Adjust the history length, taking the antilag factors into account
        float hist_len_hf = min(temporal_moments_histlen_hf.b * pow(1.0 - antilag_alpha_hf, 10) + 1.0, 256.0);
        // Compute the blending weights based on history length, so that the filter
        // converges faster. I.e. the first frame has weight of 1.0, the second frame 1/2, third 1/3 and so on.
        float alpha_color_hf = max(flt_min_alpha_color_hf, 1.0 / hist_len_hf);
        float alpha_moments_hf = max(flt_min_alpha_moments_hf, 1.0 / hist_len_hf);
        // Adjust the blending factors, taking the antilag factors into account again
        alpha_color_hf = lerp(alpha_color_hf, 1.0, antilag_alpha_hf);
        alpha_moments_hf = lerp(alpha_moments_hf, 1.0, antilag_alpha_hf);
        // Blend!
        out_color_hf.rgb = lerp(temporal_color_hf.rgb, color_curr_hf.rgb, alpha_color_hf);
        out_moments_histlen_hf.rg = lerp(temporal_moments_histlen_hf.rg, spatial_moments_hf.rg, alpha_moments_hf);
        out_moments_histlen_hf.b = hist_len_hf;
        u_Debug[ipos] = float4(float3(alpha_color_hf), 1);
    }
    else {
        u_Debug[ipos] = float4(1, 0, 1, 1);
        // No valid history - just use the current color and spatial moments
        out_color_hf.rgb = color_curr_hf;
        out_moments_histlen_hf = float4(spatial_moments_hf, 1, 1);
    }

    if (temporal_sample_valid_spec) {
        // Same sequence as above, only for the specular channel
        float antilag = grad_hf_spec.y * flt_antilag_spec + motion_length * flt_antilag_spec_motion;
        float antilag_alpha_spec = clamp(lerp(1.0, antilag, flt_temporal_spec), 0, 1);
        float hist_len_spec = min(temporal_color_histlen_spec.a * pow(1.0 - antilag_alpha_spec, 10) + 1.0, 256.0);
        float alpha_color_spec = max(flt_min_alpha_color_spec, 1.0 / hist_len_spec);
        alpha_color_spec = lerp(alpha_color_spec, 1.0, antilag_alpha_spec);
        out_color_histlen_spec.rgb = lerp(temporal_color_histlen_spec.rgb, color_curr_spec.rgb, alpha_color_spec);
        out_color_histlen_spec.a = hist_len_spec;
    }
    else {
        out_color_histlen_spec = float4(color_curr_spec, 1);
    }

    // Store the outputs for furhter processing by the a-trous HF filter
    u_MomentsHistlenHF[ipos] = out_moments_histlen_hf;
    u_ColorHistlenSpec[ipos] = out_color_histlen_spec;
    u_AtrousHF[ipos] = PackRGBE(out_color_hf);
    u_AtrousSpec[ipos] = PackRGBE(out_color_histlen_spec.rgb);
    u_AtrousMoments[ipos] = out_moments_histlen_hf.xy;
}