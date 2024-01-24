#ifndef _SRENDERER_ADDON_PHOTON_MAPPING_HEADER_
#define _SRENDERER_ADDON_PHOTON_MAPPING_HEADER_

// The block size of a tiled texture, which is used in FlattensPixelToIndex
static const uint TEXMEM_BLOCK_SIZE = 16;
/** Flattens a 2D pixel position to a 1D index into a texture,
 * so that neighboring pixels are locally coherent in memory.
 * @param pixelPosition The input 2D pixel position.
 * @param textureSize The size of the texture.
 * @return The flattened index. */
uint FlattensPixelToIndex(int2 pixelPosition, int data_pitch) {
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

void InterloeckedAddFloat3(RWByteAddressBuffer buffer, uint id, float3 color) {
    buffer.InterlockedAddF32(id * sizeof(float3) + 0, color.x);
    buffer.InterlockedAddF32(id * sizeof(float3) + 4, color.y);
    buffer.InterlockedAddF32(id * sizeof(float3) + 8, color.z);
}

float3 WS2PS(float3 positionWS, float4x4 viewProjMat) {
    const float4 positionCS = mul(float4(positionWS, 1.0f), viewProjMat);
    const float3 positionSS = positionCS.xyz / positionCS.w;
    return float3(positionSS.xy * 0.5 + 0.5, positionSS.z);
}

#endif