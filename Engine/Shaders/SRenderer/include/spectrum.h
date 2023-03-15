#ifndef _SRENDERER_COMMON_SPECTRUM_HEADER_
#define _SRENDERER_COMMON_SPECTRUM_HEADER_

float luminance(in const vec3 s) {
    return s.x * float(0.212671) + s.y * float(0.715160) + s.z * float(0.072169);
}

#endif