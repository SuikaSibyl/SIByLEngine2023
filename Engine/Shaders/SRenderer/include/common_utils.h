#ifndef _SRENDERER_COMMON_UTILS_HEADER_
#define _SRENDERER_COMMON_UTILS_HEADER_

vec3 to_world(in mat3 frame, in vec3 v) {
    return frame[0] * v[0] + frame[1] * v[1] + frame[2] * v[2];
    return frame * v;
}

vec3 to_local(in const mat3 frame, in const vec3 v) {
    return vec3(dot(v, frame[0]), dot(v, frame[1]), dot(v, frame[2]));
}

#endif