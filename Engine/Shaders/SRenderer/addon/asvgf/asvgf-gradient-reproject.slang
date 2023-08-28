#include "../vbuffer/vbuffer.hlsli"
#define _GBUFFER_RW_ENABLE_
#include "../gbuffer/gbuffer_interface.hlsli"
#undef _GBUFFER_RW_ENABLE_
#include "../../include/common/cpp_compatible.hlsli"
#include "../gbuffer/gbuffer_prev_interface.hlsli"
#include "asvgf.hlsli" // see description and copyright info here

[[vk::push_constant]]
cbuffer PushConstants {
    uint sample_batch;
    uint init_rand;
};

#define GROUP_SIZE_GRAD 8
#define GROUP_SIZE_PIXELS (GROUP_SIZE_GRAD * GRAD_DWN)

RWTexture2D<uint> u_GradSamplePos;
RWTexture2D<float2> u_HfSpecLum_prev;
RWTexture2D<uint4> u_VBuffer;
RWTexture2D<uint> u_RNGSeed;
RWTexture2D<uint> u_IsCorrelated;

const Texture2D<uint> u_RNGSeed_prev;
const Texture2D<uint> u_GradSamplePos_prev;
const Texture2D<uint> u_HF_prev;
const Texture2D<uint> u_Spec_prev;
const Texture2D<uint4> u_VBuffer_prev;

/**
 * This shader tries to match the surfaces from the current frame with the
 * previous frame. In every 3x3 square, a single matching pixel is selected.
 * This pixel becomes a "gradient" sample for the lighting passes. Gradient
 * samples are shaded using the previous frame's random number sequence and
 * important surface parameters like normal and roughness. The goal is to
 * compare the lighting environment for such pixels between the current and
 * previous frames.
 */

// for megatile, use backward projection to find the previous frame position and luminance of the surface
// the float4 saves [x, y, luminance_hf, luminance_spec]
groupshared float4 s_reprojected_pixels[GROUP_SIZE_PIXELS][GROUP_SIZE_PIXELS];

/**
 * For a given pixel p, find its surface in the previous frame based on the motion vector.
 * If such surface exists - based on depth and normal similarity - get the old diffuse
 * and specular luminances that will be used to compute gradients later.
 * @param p The current pixel position
 * @param local_pos The local position of the pixel in the group
 */
void reproject_pixel(in_ref(int2) p, in_ref(int2) local_pos, in_ref(int2) resolution) {
    // Initialize the shared memory unconditionally
    s_reprojected_pixels[local_pos.y][local_pos.x] = float4(0);
    // Compute the previous frame position of this surface
    const float3 motion = GetMotionVector(p);
    const int2 pp = p + int2(motion.xy);
    // If the previous frame position is outside the screen, don't use it
    if (any(pp < 0 || pp >= resolution)) return;

    // Fetch the previous frame gradient position...
    const int2 pos_grad_prev = pp / GRAD_DWN;
    const uint prev_grad_sample_pos = u_GradSamplePos_prev[pos_grad_prev];
    const int2 stratum_prev = UnpackStratumOffset(prev_grad_sample_pos);

    // If this pixel was a gradient on the previous frame, don't use it. Two reasons:
    // 1) Carrying forward the same random number sequence over multiple frames introduces bias.
    // 2) Gradient pixels use light lists from the previous frame. If the same pixel was used
    // as a gradient for more than one frame, we would need to keep the light lists from 2+ frames behind.
    if (all(pos_grad_prev * GRAD_DWN + stratum_prev == pp)) return;

    // Load the data for surface matching
    const float depth_curr = GetViewDepth(p);
    const float depth_prev = GetViewDepthPrev(pp);
    const float3 geo_normal_curr = GetGeometryNormal(p);
    const float3 geo_normal_prev = GetGeometryNormalPrev(pp);
    
    const float dist_depth = abs(depth_curr - depth_prev + motion.z) / abs(depth_curr);
    const float dot_geo_normals = dot(geo_normal_curr, geo_normal_prev);

    // Compare the surfaces
    if (dist_depth < 0.1 && dot_geo_normals > 0.9) {
        const float3 prev_hf = UnpackRGBE(u_HF_prev[pp]);
        const float3 prev_spec = UnpackRGBE(u_Spec_prev[pp]);
        const float2 prev_lum = float2(luminance(prev_hf), luminance(prev_spec));
        // Store the results into shared memory: previous frame position and luminances
        s_reprojected_pixels[local_pos.y][local_pos.x] = float4(pp, prev_lum);
    }
}

// Using the visibility buffer, compute the new position of the surface that we found
// in the previous frame. Exact matching of gradient surface positions is important
// to avoid false positive gradients in the light penumbras, where a subpixel shift
// that normally happens between frames can make a difference between a light sample
// being visible or shadowed.
float3 patch_position(in_ref(int2) found_pos_prev) {
    // Read the visibility buffer
    const VBufferData vhit = UnpackVBufferData(u_VBuffer_prev[found_pos_prev]);
    return GetVBufferPosition(vhit);
}

[shader("compute")]
[numthreads(GROUP_SIZE_PIXELS, GROUP_SIZE_PIXELS, 1)]
void ComputeMain(
    int3 DTid: SV_DispatchThreadID,
    int3 GTid: SV_GroupThreadID,
    int3 Gid: SV_GroupID,
    int Gidx: SV_GroupIndex
) {
    // First pass: the entire thread group is busy matching pixels with the previous frame
    // ------------------------------------------------------------------------------------------------
    int2 ipos = DTid.xy;
    int2 pos_grad = ipos / GRAD_DWN;
    const int2 resolution = getViewportSize(globalUniform.cameraData);

    if(init_rand == 1u) {
        // Initialize the random number sequence
        RandomSamplerState RNG = InitRandomSampler(ipos, sample_batch * 0);
        u_RNGSeed[ipos] = RNG.state;
    }

    // Read the previous frame's position and luminance
    // found by backward reprojection
    reproject_pixel(ipos, GTid.xy, resolution);
    // barrier
    GroupMemoryBarrierWithGroupSync();
    
    // Second pass: the first (GROUP_SIZE_GRAD)^2 pixels are looking for the brightest
    // matching pixels in each 3x3 square.
    // ------------------------------------------------------------------------------------------------
    // Picking the brightest pixel helps prevent bright trails when the light has moved.
    // If we just pick a random pixel in the the penumbra of the sun light for example,
    // there is a high chance that this pixel will not receive any sun light due to random sampling of the sun.
    // Overall, we'll miss the changing luminance of the moving penumbra, which is very well visible.

    // Using the front several wraps to do second pass, each thread per tile
    const int2 local_pos = int2(Gidx % GROUP_SIZE_GRAD, Gidx / GROUP_SIZE_GRAD);
    if (local_pos.y >= GROUP_SIZE_GRAD) return;
    // set pos_grad to the top left corner of the tile
    pos_grad = Gid.xy * GROUP_SIZE_GRAD + local_pos;
    // set ipos to the top left corner of the pixel
    ipos = pos_grad * GRAD_DWN;

    bool found = false;
    int2 found_offset = int2(0);
    int2 found_pos_prev = int2(0);
    float2 found_prev_lum = float2(0);

    for (int offy = 0; offy < GRAD_DWN; offy++) {
        for (int offx = 0; offx < GRAD_DWN; offx++) {
            const int2 p = local_pos * GRAD_DWN + int2(offx, offy);
            const float4 reprojected_pixel = s_reprojected_pixels[p.y][p.x];
            const float2 prev_lum = reprojected_pixel.zw;
            // Use total luminance of diffuse and specular as the heuristic
            // always pick the brightest pixel
            if (prev_lum.x + prev_lum.y > found_prev_lum.x + found_prev_lum.y) {
                found_prev_lum = prev_lum;
                found_offset = int2(offx, offy);
                found_pos_prev = int2(reprojected_pixel.xy);
                found = true;
            }
        }
    }
    
    if (!found) {
        u_GradSamplePos[pos_grad] = 0u;
        return;
    }

    // Final pass: store the gradient information and patch the surface parameters
    // ------------------------------------------------------------------------------------------------
    ipos += found_offset;
    const uint gradient_idx = PackStratumOffset(found_offset); /* current frame */

    // Store the gradient index offset
    u_GradSamplePos[pos_grad] = gradient_idx;
    u_HfSpecLum_prev[pos_grad] = found_prev_lum;
    
    // Patch the surface parameters
    t_GBufferPosition[ipos].xyz = patch_position(found_pos_prev);
    u_IsCorrelated[ipos] = 1u;
    u_RNGSeed[ipos] = u_RNGSeed_prev[found_pos_prev];
    t_GBufferNormals[ipos] = t_PrevGBufferNormals[found_pos_prev];
    t_GBufferGeoNormals[ipos] = t_PrevGBufferGeoNormals[found_pos_prev];
    t_GBufferDiffuseAlbedo[ipos] = t_PrevGBufferDiffuseAlbedo[found_pos_prev];
    t_GBufferSpecularRough[ipos] = t_PrevGBufferSpecularRough[found_pos_prev];
}