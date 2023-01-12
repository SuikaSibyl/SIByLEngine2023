#ifndef _SRENDERER_COMMON_SAMPLE_SHAPE_HEADER_
#define _SRENDERER_COMMON_SAMPLE_SHAPE_HEADER_

#include "common_trace.h"

struct SampleQuery {
    // input
    vec3    ref_point;
    uint    geometry_id;
    vec2    uv;         // for selecting a point on a 2D surface
    float   w;          // for selecting triangles
    // output
    vec3    position;
    vec3    normal;
};

struct SamplePdfQuery {
    // input
    vec3    ref_point;
    uint    geometry_id;
    vec3    sample_position;
    vec3    sample_normal;
    // output
    float   pdf;
};

#endif