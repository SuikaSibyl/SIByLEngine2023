#ifndef _SRENDERER_ADDON_VXGUIDING_DIRECTIONAL_HASH_HEADER_
#define _SRENDERER_ADDON_VXGUIDING_DIRECTIONAL_HASH_HEADER_

#include "../../../include/common/octahedral.hlsli"


int hash_directional(float3 dir) {
    const float2 uv = UnitVectorToSignedOctahedron(dir) * 0.5f + 0.5f;
    const int2 xy = clamp(int2(8 * uv), 0, 7);
    return xy.y * 8 + xy.x;
}

float3 unhash_directional(int hash, float2 rand) {
    const int2 xy = int2(hash % 8, hash / 8);
    const float2 uv = (xy + rand) / 8.0f;
    return SignedOctahedronToUnitVector(uv * 2.0f - 1.0f);
}

#ifdef TABULATE_NBIT_FINDING_
// tabulate nth bit set for 8-bit values
groupshared uint8_t nthbit_tabulate[256][8];
// find nth bit set in 32-bit value
uint find_nth_bit_set_tabulate(uint v, uint n) {
    uint p = countbits(v & 0xFFFF);
    uint shift = 0;
    if (p <= n) {
        v >>= 16;
        shift += 16;
        n -= p;
    }
    p = countbits(v & 0xFF);
    if (p <= n) {
        shift += 8;
        v >>= 8;
        n -= p;
    }
    if (n >= 8) return 0; // optional safety, in case n > # of set bits
    return nthbit_tabulate[v & 0xFF][n] + shift;
}
// find nth bit set in 64-bit value
uint find_nth_bit_set_tabulate(uint64_t v, uint n) {
    uint seg = uint(v & 0xFFFFFFFF); // lower32
    const uint p = countbits(seg);
    uint shift = 0;
    if (p <= n) {
        seg = uint(v >> 32);   // upper32
        shift += 32;
        n -= p;
    }
    return find_nth_bit_set_tabulate(v, n) + shift;
}
#endif // TABULATE_NBIT_FINDING_



#endif // !_SRENDERER_ADDON_VXGUIDING_DIRECTIONAL_HASH_HEADER_