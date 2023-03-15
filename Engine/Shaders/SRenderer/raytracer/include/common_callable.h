#ifndef _SRENDERER_COMMON_SAMPLE_SHAPE_HEADER_
#define _SRENDERER_COMMON_SAMPLE_SHAPE_HEADER_

struct ShapeSampleQuery {
    // input
    vec3    ref_point;
    uint    geometry_id;
    vec2    uv;         // for selecting a point on a 2D surface
    float   w;          // for selecting triangles
    uvec2    offset;
    uvec2    size;
    // output
    vec3    position;
    vec3    normal;
};

struct ShapeSamplePdfQuery {
    // input
    vec3    ref_point;
    uint    geometry_id;
    vec3    sample_position;
    vec3    sample_normal;
    // output
    float   pdf;
};

bool validRefPoint(in const vec3 ref_point) {
    return !(isnan(ref_point.x) || isnan(ref_point.y) || isnan(ref_point.z));
}

struct BSDFEvalQuery {
    // input
    vec3    dir_in;
    uint    mat_id;
    vec3    dir_out;
    uint    transport_type;     // 0: path tracing // 1: light tracing
    vec3    geometric_normal;
    mat3    frame;
    vec2    uv;
    float   hitFrontface;
    // output
    vec3    bsdf;
};

const uint enum_transport_to_light = 0;
const uint enum_transport_to_view  = 1;

struct BSDFSampleQuery {
    // input
    vec3    dir_in;
    uint    mat_id;
    vec3    geometric_normal;
    mat3    frame;
    vec2    uv;
    vec2    rnd_uv;
    float   rnd_w;
    float   hitFrontface;
    // output
    vec3    dir_out;
};

struct BSDFSamplePDFQuery {
    // input
    vec3    dir_in;
    uint    mat_id;
    vec3    dir_out;
    vec3    geometric_normal;
    mat3    frame;
    vec2    uv;
    float   hitFrontface;
    // output
    float   pdf;
};

#endif