#ifndef _SRENDERER_PATHGUIDING_ADDON_COMPRESSED_DIRECTIONAL_QUADTREE_HEADER_
#define _SRENDERER_PATHGUIDING_ADDON_COMPRESSED_DIRECTIONAL_QUADTREE_HEADER_

#include "../../../include/common/bit-operation.hlsli"

uint root(uint64_t cdq) {
    return (uint)(cdq >> 60) & 0b1111;
}

uint child(uint64_t cdq, int node_idx, uint child) {
    return 4 * rank(cdq, node_idx) + child;
}

bool is_inner(uint64_t cdq, int node_idx) {
    return (cdq & (1ull << max(63 - node_idx, 0))) != 0;
}

uint find_child_id(float2 uv) {
    return (uv.x < 0.5f ? 0 : 1) + (uv.y < 0.5f ? 0 : 2);
}

float2 zoom_in_coord(float2 uv) {
    return frac(uv * 2.0f);
}

#endif // !_SRENDERER_PATHGUIDING_ADDON_COMPRESSED_DIRECTIONAL_QUADTREE_HEADER_