#ifndef _SRENDERER_SPACCE_FILLING_CURVE_INCLUDED_
#define _SRENDERER_SPACCE_FILLING_CURVE_INCLUDED_

/***************************************************************************
# Implementation of various space-filling curves.
# 1. Z-order/Morton curve 2D
@ 2. Z-order/Morton curve 3D
 **************************************************************************/

// The following code about 2D Z-Order is adapted from:
// @ref: NVIDIAGameWorks/ RTXDI
// @url: https://github.com/NVIDIAGameWorks/RTXDI/blob/
//       1018431b07fae1ea02ac2002eb7bdee882538d30/rtxdi-sdk/include/rtxdi/RtxdiMath.hlsli#L55

// "Explodes" an integer, i.e. inserts a 0 between each bit.
// Takes inputs up to 16 bit wide.
// For example, 0b11111111 -> 0b1010101010101010
uint IntegerExplode1Bit(uint x) {
    x = (x | (x << 8)) & 0x00FF00FF;
    x = (x | (x << 4)) & 0x0F0F0F0F;
    x = (x | (x << 2)) & 0x33333333;
    x = (x | (x << 1)) & 0x55555555;
    return x;
}

// Reverse of IntegerExplode1Bit, i.e. takes every other bit in the integer and compresses
// those bits into a dense bit firld. Takes 32-bit inputs, produces 16-bit outputs.
// For example, 0b'abcdefgh' -> 0b'0000bdfh'
uint IntegerCompact1Bit(uint x) {
    x = (x & 0x11111111) | ((x & 0x44444444) >> 1);
    x = (x & 0x03030303) | ((x & 0x30303030) >> 2);
    x = (x & 0x000F000F) | ((x & 0x0F000F00) >> 4);
    x = (x & 0x000000FF) | ((x & 0x00FF0000) >> 8);
    return x;
}

// Converts a 2D position to a linear index following a Z-curve pattern.
uint ZCurve2DToMortonCode(uvec2 xy) {
    return IntegerExplode1Bit(xy[0]) | (IntegerExplode1Bit(xy[1]) << 1);
}

// Converts a linear to a 2D position following a Z-curve pattern.
uvec2 MortonCodeToZCurve2D(uint index) {
    return uvec2(IntegerCompact1Bit(index), IntegerCompact1Bit(index >> 1));
}

// The following code about 3D Z-Order is adapted from:
// @ref: Thinking Parallel, Part III: Tree Construction on the GPU
// @url: https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/

// "Explodes" an integer, i.e. inserts two 0-s between each bit.
// Takes inputs up to 10 bit wide.
// For example, 0b11111111 -> 0b001001001001001001001001
uint IntegerExplode2Bit(uint x) {
    x = (x * 0x00010001) & 0xFF0000FF;
    x = (x * 0x00000101) & 0x0F00F00F;
    x = (x * 0x00000011) & 0xC30C30C3;
    x = (x * 0x00000005) & 0x49249249;
    return x;
}

// Converts a 3D position to a linear index following a Z-curve pattern.
uint ZCurve3DToMortonCode(uvec3 xyz) {
    const uint xx = IntegerExplode2Bit(xyz.x);
    const uint yy = IntegerExplode2Bit(xyz.y);
    const uint zz = IntegerExplode2Bit(xyz.z);
    return xx * 4 + yy * 2 + zz;
}

// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
uint ZCurve3DToMortonCode(vec3 xyz) {
    const float x = min(max(xyz.x * 1024.0f, 0.0f), 1023.0f);
    const float y = min(max(xyz.y * 1024.0f, 0.0f), 1023.0f);
    const float z = min(max(xyz.z * 1024.0f, 0.0f), 1023.0f);
    return ZCurve3DToMortonCode(uvec3(uint(x), uint(y), uint(z)));
}

#endif