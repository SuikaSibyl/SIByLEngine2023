#ifndef _SRENDERER_COMMON_SPLAT_FILM_HEADER_
#define _SRENDERER_COMMON_SPLAT_FILM_HEADER_

void addSplat(
    in const vec2 pixel,
    in const vec4 color
) {
    imageAtomicAdd(atomicRGB, ivec3(pixel, 0), color.x);
    imageAtomicAdd(atomicRGB, ivec3(pixel, 1), color.y);
    imageAtomicAdd(atomicRGB, ivec3(pixel, 2), color.z);
    imageAtomicAdd(atomicRGB, ivec3(pixel, 3), color.w);
}

#endif