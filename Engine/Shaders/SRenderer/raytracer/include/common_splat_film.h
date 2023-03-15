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

bool insideExclusive(
    in const ivec2 p,
    in const ivec2 b_min,
    in const ivec2 b_max
) {
    return (p.x >= b_min.x && p.x < b_max.x &&
            p.y >= b_min.y && p.y < b_max.y);
}

bool insideExclusive(
    in const ivec2 p,
    in const ivec2 b_max
) {
    const ivec2 b_min = ivec2(0, 0);
    return insideExclusive(p, b_min, b_max);
}

#endif