#ifndef _SRENDERER_ADDON_RESTIR_GI_RUNTIMEPARAMS_HEADER_
#define _SRENDERER_ADDON_RESTIR_GI_RUNTIMEPARAMS_HEADER_

#include "../../../include/common/cpp_compatible.hlsli"

#ifndef RESTIRGI_RESERVOIR_BLOCK_SIZE
#define RESTIRGI_RESERVOIR_BLOCK_SIZE 16
#endif

struct GIResamplingRuntimeParameters {
    uint reservoirArrayPitch;    // number of elements in a whole reservoir array
    uint reservoirBlockRowPitch; // number of elements in a row of reservoir blocks
    uint uniformRandomNumber;
    uint neighborOffsetMask;
};

uint ReservoirPositionToPointer(
    in_ref(GIResamplingRuntimeParameters) params,
    in_ref(uint2) reservoirPosition,
    in_ref(uint) reservoirArrayIndex
) {
    const uint2 blockIdx = reservoirPosition / RESTIRGI_RESERVOIR_BLOCK_SIZE;
    const uint2 positionInBlock = reservoirPosition % RESTIRGI_RESERVOIR_BLOCK_SIZE;
    return reservoirArrayIndex * params.reservoirArrayPitch
        + blockIdx.y * params.reservoirBlockRowPitch
        + blockIdx.x * (RESTIRGI_RESERVOIR_BLOCK_SIZE * RESTIRGI_RESERVOIR_BLOCK_SIZE)
        + positionInBlock.y * RESTIRGI_RESERVOIR_BLOCK_SIZE
        + positionInBlock.x;
}

#endif // _SRENDERER_ADDON_RESTIR_GI_RUNTIMEPARAMS_HEADER_