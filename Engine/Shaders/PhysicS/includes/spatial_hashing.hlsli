#ifndef _PHYSICS_INCLUDES_SPATIAL_HASHING_HEADER_
#define _PHYSICS_INCLUDES_SPATIAL_HASHING_HEADER_

/** Hash integer coords to 1D index in table size. */
uint hashCoords(in int3 icoord, in uint tableSize) {
    const int h = (icoord.x * 92837111) 
                ^ (icoord.y * 689287499) 
                ^ (icoord.z * 283923481);
    return abs(h) % tableSize; 
}

/** Returns the hash cell index containing the given coordinate. */
int intCoord(in float coord, in float inv_spacing) {
    return int(floor(coord * inv_spacing)); } // 1d variant
int2 intCoord(in float2 coord, in float inv_spacing) {
    return int2(floor(coord * inv_spacing));} // 2d variant
int3 intCoord(in float3 coord, in float inv_spacing) {
    return int3(floor(coord * inv_spacing));} // 3d variant

/** */

#endif
