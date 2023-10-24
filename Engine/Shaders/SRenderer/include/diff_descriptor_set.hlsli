#ifndef _SRENDERER_COMMON_DIFF_DESCRIPOTR_SET_HEADER_
#define _SRENDERER_COMMON_DIFF_DESCRIPOTR_SET_HEADER_

#include "common/cpp_compatible.hlsli"

enum DiffResourceType {
    DIFF_RESOURCE_TYPE_NONE = 0,
    DIFF_RESOURCE_TYPE_TEXTURE = 1,
    DIFF_RESOURCE_TYPE_BUFFER = 2,
};

struct DiffResourceDesc {
    uint data_size;      // the size of the data in bytes
    uint data_offset;    // the offset of the data in bytes
    uint data_extend;    // texture height(16) | width(16) in pixels
    uint data_flag;      // mask (4 bits) | type (2 bits)
};

// Get the mask of the resource data type, be used for partially differentiable resource
// For example, in a packed resource Texture<float4> with 4 channels, about various information
// including roughness, metalness, etc., And we only want to differentiate the roughness channel,
// then we can set the mask to 0b0001, and the data_flag will be 0b000101.
uint get_data_mask(in_ref(DiffResourceDesc) desc) { return (desc.data_flag >> 2) & 0xF; }

/** Differentiable resource descriptor set. */
[[vk::binding(0, 2)]] RWByteAddressBuffer ParamGradients;
[[vk::binding(1, 2)]] const StructuredBuffer<DiffResourceDesc> DiffResourcesDescs;
[[vk::binding(2, 2)]] const StructuredBuffer<int> DiffableTextureIndices;

void SetGrad(uint index, float gradient) { ParamGradients.Store(index * sizeof(float), gradient); }

float GetGrad(uint index) { return asfloat(ParamGradients.Load(index * sizeof(float))); }

float InterlockedAddGrad(uint index, float gradient) {
    float old_gradient; ParamGradients.InterlockedAddF32(index * sizeof(float), 
    gradient, old_gradient); return old_gradient;
}

float3 InterlockedAddGradFloat3(uint index, float3 gradient) {
    float3 old_value = float3(0);
    old_value.x = InterlockedAddGrad(index * 3 + 0, gradient.x);
    old_value.x = InterlockedAddGrad(index * 3 + 1, gradient.y);
    old_value.x = InterlockedAddGrad(index * 3 + 2, gradient.z);
    return old_value;
}

static const uint TEXMEM_BLOCK_SIZE = 16; // The block size of a tiled texture, which is used in FlattensPixelToIndex

/**
 * Flattens a 2D pixel position to a 1D index into a texture,
 * so that neighboring pixels are locally coherent in memory.
 * @param pixelPosition The input 2D pixel position.
 * @param textureSize The size of the texture.
 * @return The flattened index.
 */
uint FlattensPixelToIndex(
    in_ref(int2) pixelPosition,
    in_ref(int) data_pitch) {
    // compute the block index and position within the block
    const uint2 blockIdx = pixelPosition / TEXMEM_BLOCK_SIZE;
    const uint2 positionInBlock = pixelPosition % TEXMEM_BLOCK_SIZE;
    // compute the pitch of a row of blocks
    const uint32_t renderWidthBlocks = (data_pitch + TEXMEM_BLOCK_SIZE - 1) / TEXMEM_BLOCK_SIZE;
    const uint32_t reservoirBlockRowPitch = renderWidthBlocks * (TEXMEM_BLOCK_SIZE * TEXMEM_BLOCK_SIZE);
    // return the flattened index
    return blockIdx.y * reservoirBlockRowPitch
        + blockIdx.x * (TEXMEM_BLOCK_SIZE * TEXMEM_BLOCK_SIZE)
        + positionInBlock.y * TEXMEM_BLOCK_SIZE
        + positionInBlock.x;
}

/**
 * Software implementation of bilinear interpolation,
 * @param texture The texture to sample.
 * @param texcoord The texture coordinate to sample.
 * @param tex_dimension The dimension of the texture.
 */
float4 software_bilinear_interpolation_rgba(
    Sampler2D texture, 
    float2 texcoord, 
    int2 tex_dimension
) {
    // do some math to get the texel coordinate
    const float2 pixel = texcoord * tex_dimension + 0.5;
    const float2 fract = frac(pixel);
    // gather texels for all the channels
    const float4 reds = texture.GatherRed(texcoord);
    const float4 greens = texture.GatherGreen(texcoord);
    const float4 blues = texture.GatherBlue(texcoord);
    const float4 alphas = texture.GatherAlpha(texcoord);
    // integrate the channels to reconstruct each channel
    const float4 val1 = float4(reds.x, greens.x, blues.x, alphas.x);
    const float4 val2 = float4(reds.y, greens.y, blues.y, alphas.y);
    const float4 val3 = float4(reds.z, greens.z, blues.z, alphas.z);
    const float4 val4 = float4(reds.w, greens.w, blues.w, alphas.w);
    // apply bilinear interpolation
    const float4 top_row_val = lerp(val4, val3, fract.x);
    const float4 bottom_row_red = lerp(val1, val2, fract.x);
    const float4 final_val = lerp(top_row_val, bottom_row_red, fract.y);
    return final_val;
}

#endif // !_SRENDERER_COMMON_DIFF_DESCRIPOTR_SET_HEADER_