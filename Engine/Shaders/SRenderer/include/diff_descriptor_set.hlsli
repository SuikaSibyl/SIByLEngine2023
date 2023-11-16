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

/** Various differentiable information types. */
enum DiffInfoType {
    DIFF_INFO_TYPE_TEXTURE = 0,   // gradient of textures
    DIFF_INFO_TYPE_MATERIAL = 1,  // gradient of material parameters
    DIFF_INFO_TYPE_TRANSFORM = 2, // gradient of transform parameters
    DIFF_INFO_TYPE_VERTICES = 3,  // gradient of vertices positions
};

/** The gradient layout for various DiffInfoType. */
struct DiffInfoGradLayoutDesc {
    int offset; // offset of the texture gradient in element
    int dim;    // dimension of the texture gradient in number of floats
    int hash;   // hash size of the texture gradient
    int size;   // size of the texture gradient in bytes (dim * hash)
};

// Get the mask of the resource data type, be used for partially differentiable resource
// For example, in a packed resource Texture<float4> with 4 channels, about various information
// including roughness, metalness, etc., And we only want to differentiate the roughness channel,
// then we can set the mask to 0b0001, and the data_flag will be 0b000101.
uint get_data_mask(in_ref(DiffResourceDesc) desc) { return (desc.data_flag >> 2) & 0xF; }

/** Differentiable resource descriptor set. */
[[vk::binding(0, 2)]] RWByteAddressBuffer ParamGradients;
[[vk::binding(1, 2)]] const StructuredBuffer<DiffInfoGradLayoutDesc> GradLayoutDescs;
[[vk::binding(2, 2)]] const StructuredBuffer<DiffResourceDesc> DiffResourcesDescs;
[[vk::binding(3, 2)]] const StructuredBuffer<int> DiffableTextureIndices;


// =================================================================================================
// Gradient buffer (raw) manipulation.
// =================================================================================================

/** Set the gradient at a specific global index */
void SetGrad(uint index, float gradient) { ParamGradients.Store(index * sizeof(float), gradient); }
/** Get the gradient at a specific global index */
float GetGrad(uint index) { return ParamGradients.Load<float>(index * sizeof(float)); }
/** Atomic Add the gradient at a specific global index */
void AtomicAddGrad(uint index, float gradient) { 
    return ParamGradients.InterlockedAddF32(index * sizeof(float), gradient); }
/**
 * AtomicAdd the gradient to the specified offset of the parameter.
 * @param type The type of the parameter (as DiffInfoType).
 * @param offset The offset of the parameter (in element/float count).
 * @param gradient The gradient to add.
 * @param hash The hash of the gradient addition behavior.  */
void InterlockedAddGrad(DiffInfoType type, uint offset, float gradient, uint hash = 0) {
    const DiffInfoGradLayoutDesc layout = GradLayoutDescs[int(type)];
    hash = hash % layout.hash; // hash should be less than the hash size of the gradient
    if (offset < layout.dim) { // offset should be less than the dimension of the gradient
        const uint index = layout.offset + hash * layout.dim + offset;
        AtomicAddGrad(index, gradient); // add the gradient to the global index
    }
}
/** The vector variant of the InterlockedAddGrad.
 * AtomicAdd the gradient to the specified offset of the parameter.
 * @param type The type of the parameter (as DiffInfoType).
 * @param offset The offset of the parameter (in element/float count).
 * @param gradient The gradient to add.
 * @param hash The hash of the gradient addition behavior. */
void InterlockedAddGrad<let N : int>(DiffInfoType type, uint offset, vector<float, N> gradient, uint hash = 0) {
    const DiffInfoGradLayoutDesc layout = GradLayoutDescs[int(type)];
    hash = hash % layout.hash; // hash should be less than the hash size of the gradient
    if (offset < layout.dim) { // offset should be less than the dimension of the gradient
        const uint index = layout.offset + hash * layout.dim + offset;
        [ForceUnroll]
        for (uint i = 0; i < N; i++)
            AtomicAddGrad(index + i, gradient[i]);
    }
}

// =================================================================================================
// Gradient buffer associated with texture resources.
// =================================================================================================
// The block size of a tiled texture, which is used in FlattensPixelToIndex
static const uint TEXMEM_BLOCK_SIZE = 16;
/** Flattens a 2D pixel position to a 1D index into a texture,
 * so that neighboring pixels are locally coherent in memory.
 * @param pixelPosition The input 2D pixel position.
 * @param textureSize The size of the texture.
 * @return The flattened index. */
uint FlattensPixelToIndex(
    in_ref(int2) pixelPosition,
    in_ref(int) data_pitch
) {
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
/** xth power of 4, e.i. 4^{x} */
int xth_power_of_four(int x) { return 1 << (x * 2); }
/** offset of a mipmap texture with power2 */
int mipmap_offset_power2(int size, int level) {
    const int mipmap_level = 1 + int(floor(log2(size))); // 10
    const int sum_lhs = xth_power_of_four(mipmap_level); // 1,048,576
    const int index = mipmap_level - level;              // 8
    const int sum_rhs = xth_power_of_four(index);        // 262,144
    return (sum_lhs - sum_rhs) / 3;                      // 1,048,576
}
/** Software implementation of bilinear interpolation,
 * Instead of using load, we use gather to sample the texture.
 * @param texture The texture to sample.
 * @param texcoord The texture coordinate to sample.
 * @param tex_dimension The dimension of the texture. */
float4 software_bilinear_interpolation_rgba_gather(
    Sampler2D texture, 
    float2 texcoord, 
    int2 tex_dimension
) {
    // do some math to get the texel coordinate
    const float2 pixel = texcoord * tex_dimension + 0.5;
    // we require a offset to avoid some artifacts when frac==1,
    // please refer to the webpage to see why it is 1/512:
    // @url: https://www.reedbeta.com/blog/texture-gathers-and-coordinate-precision/
    const float offset = 1.0 / 512.0;
    const float2 fract = frac(pixel + offset);
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
/** Software implementation of bilinear interpolation,
 * Instead of using gather, we use load to sample the texture.
 * @param texture The texture to sample.
 * @param texcoord The texture coordinate to sample.
 * @param tex_dimension The dimension of the texture. */
float4 software_bilinear_interpolation_rgba_load(
    Sampler2D texture,
    float2 texcoord,
    int2 tex_dimension
) {
    // do some math to get the texel coordinate
    const float2 pixel = texcoord * tex_dimension + 0.5;
    const float2 fract = frac(pixel);
    // gather texels for all the channels
    const int2 lb_pixel = int2(floor(pixel - float2(1.0)));
    // apply bilinear interpolation
    float4 neighbor_weights;
    neighbor_weights.x = (1.0 - fract.x) * (1.0 - fract.y);
    neighbor_weights.y = (1.0 - fract.x) * fract.y;
    neighbor_weights.z = fract.x * (1.0 - fract.y);
    neighbor_weights.w = fract.x * fract.y;
    const float4 final_val =
        texture.Load(int3((lb_pixel + int2(0, 0)), 0)) * neighbor_weights[0] +
        texture.Load(int3((lb_pixel + int2(0, 1)), 0)) * neighbor_weights[1] +
        texture.Load(int3((lb_pixel + int2(1, 0)), 0)) * neighbor_weights[2] +
        texture.Load(int3((lb_pixel + int2(1, 1)), 0)) * neighbor_weights[3];
    return final_val;
}
/** Splat the gradient towards a texture resource,
 * @param gradient The gradient to splat.
 * @param texcoord The texture coordinate to splat.
 * @param tex_dimension The dimension of the texture.
 * @param offset The offset of the gradient. 
 * @param hash The hash of the gradient addition behavior.
 */
void bilinear_gradient_spatting(
    float gradient,
    float2 texcoord,
    int2 tex_dimension,
    int offset,
    int hash = 0
) {
    // forget about the gradient if it is zero
    if (all(gradient == 0)) return;
    if (any(tex_dimension == 0)) return;
    // do some math to get the texel coordinate
    const float2 pixel = texcoord * tex_dimension + 0.5;
    const float2 fract = frac(pixel);
    // gather texels for all the channels
    const int2 lb_pixel = int2(floor(pixel - float2(1.0)));
    // apply bilinear interpolation
    float4 neighbor_weights;
    neighbor_weights.x = (1.0 - fract.x) * (1.0 - fract.y);
    neighbor_weights.y = (1.0 - fract.x) * fract.y;
    neighbor_weights.z = fract.x * (1.0 - fract.y);
    neighbor_weights.w = fract.x * fract.y;
    // splat the gradient to the 4 neighbors
    const int2 offset_arr[4] = {
        int2(0, 0),
        int2(0, 1),
        int2(1, 0),
        int2(1, 1), };
    [ForceUnroll]
    for (uint i = 0; i < 4; i++)
        InterlockedAddGrad(
            DiffInfoType::DIFF_INFO_TYPE_TEXTURE,
            offset + FlattensPixelToIndex(int2(lb_pixel + offset_arr[i]), tex_dimension.x),
            gradient * neighbor_weights[i], hash);
}
/** Splat the gradient towards a texture resource,
 * but using a multi-channel gradient.
 * @param gradient The gradient to splat.
 * @param texcoord The texture coordinate to splat.
 * @param tex_dimension The dimension of the texture.
 * @param offset The offset of the gradient.
 * @param hash The hash of the gradient addition behavior.
 */
void bilinear_gradient_spatting<let N : int>(
    vector<float, N> gradient,
    float2 texcoord,
    int2 tex_dimension,
    int offset,
    int hash = 0
) {
    // forget about the gradient if it is zero
    if (all(gradient == 0)) return;
    if (any(tex_dimension == 0)) return;
    // do some math to get the texel coordinate
    const float2 pixel = texcoord * tex_dimension + 0.5;
    const float2 fract = frac(pixel);
    // gather texels for all the channels
    const int2 lb_pixel = int2(floor(pixel - float2(1.0)));
    // apply bilinear interpolation
    float4 neighbor_weights;
    neighbor_weights.x = (1.0 - fract.x) * (1.0 - fract.y);
    neighbor_weights.y = (1.0 - fract.x) * fract.y;
    neighbor_weights.z = fract.x * (1.0 - fract.y);
    neighbor_weights.w = fract.x * fract.y;
    // splat the gradient to the 4 neighbors
    const int2 offset_arr[4] = {
        int2(0, 0),
        int2(0, 1),
        int2(1, 0),
        int2(1, 1), };
    [ForceUnroll]
    for (uint i = 0; i < 4; i++)
        InterlockedAddGrad(
            DiffInfoType::DIFF_INFO_TYPE_TEXTURE,
            offset + (N * FlattensPixelToIndex(int2(lb_pixel + offset_arr[i]) * N, tex_dimension.x)),
            gradient * neighbor_weights[i], hash);
}
/** Splat the gradient towards a texture resource,
 * but with a trilinear splatting.
 * @param gradient The gradient to splat.
 * @param texcoord The texture coordinate to splat.
 * @param tex_dimension The dimension of the texture.
 * @param offset The offset of the gradient. */
void trilinear_gradient_spatting(
    float gradient,
    float3 texcoord,
    int2 tex_dimension,
    int offset,
    int hash = 0
) {
    // clamp the mip level to the maximum level
    const float max_dimension = log2(max(tex_dimension.x, tex_dimension.y)) - 0.0001;
    const float mip_leel = clamp(texcoord.z, 0.0, max_dimension);
    const int base_level = int(floor(mip_leel));
    const float z = frac(texcoord.z);
    bilinear_gradient_spatting(
        gradient * (1 - z), texcoord.xy, tex_dimension >> base_level,
        offset + mipmap_offset_power2(tex_dimension.x, base_level), hash);
    bilinear_gradient_spatting(
        gradient * (z), texcoord.xy, tex_dimension >> (base_level + 1),
        offset + mipmap_offset_power2(tex_dimension.x, base_level + 1), hash);
}
/** Splat the gradient towards a texture resource,
 * but with a trilinear splatting and using a multi-channel gradient
 * @param gradient The gradient to splat.
 * @param texcoord The texture coordinate to splat.
 * @param tex_dimension The dimension of the texture.
 * @param offset The offset of the gradient. 
 * @param hash The hash of the gradient addition behavior.
 */
void trilinear_gradient_spatting<let N : int>(
    vector<float, N> gradient,
    float3 texcoord,
    int2 tex_dimension,
    int offset,
    int hash = 0
) {
    // clamp the mip level to the maximum level
    const float max_dimension = log2(max(tex_dimension.x, tex_dimension.y)) - 0.0001;
    const float mip_leel = clamp(texcoord.z, 0.0, max_dimension);
    const int base_level = int(floor(mip_leel));
    const float z = frac(texcoord.z);
    bilinear_gradient_spatting(
        gradient * (1 - z), texcoord.xy, tex_dimension >> base_level,
        offset + mipmap_offset_power2(tex_dimension.x, base_level) * N);
    bilinear_gradient_spatting(
        gradient * (z), texcoord.xy, tex_dimension >> (base_level + 1),
        offset + mipmap_offset_power2(tex_dimension.x, base_level + 1) * N);
}
/** Splat the gradient towards a texture resource,
 * but with a trilinear splatting and using a multi-channel gradient
 * @param gradient The gradient to splat.
 * @param texcoord The texture coordinate to splat.
 * @param tex_dimension The dimension of the texture.
 * @param offset The offset of the gradient.
 * @param hash The hash of the gradient addition behavior.
 */
void InterlockedAddGradTex<let N : int>(
    vector<float, N> gradient,
    int texture_idx,
    float3 texcoord,
    int hash = 0
) {
    const int diff_id = DiffableTextureIndices[texture_idx];
    if (diff_id == -1) return; // the texture is not differentiable
    const DiffResourceDesc diff_desc = DiffResourcesDescs[diff_id];
    const int width = (diff_desc.data_extend >> 0) & 0xFFFF;
    const int height = (diff_desc.data_extend >> 16) & 0xFFFF;
    trilinear_gradient_spatting(gradient, texcoord, int2(width, height), diff_desc.data_offset, hash);
}

#endif // !_SRENDERER_COMMON_DIFF_DESCRIPOTR_SET_HEADER_