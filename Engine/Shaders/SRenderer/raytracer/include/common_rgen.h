#ifndef _SRENDERER_COMMON_RT_RGEN_HEADER_
#define _SRENDERER_COMMON_RT_RGEN_HEADER_

#include "../../../Utility/random.h"
#include "../../../Utility/sampling.h"
#include "layouts/rgen_layouts.h"

bool intersect(
    in const Ray ray,
    out SurfaceIntersection isect
) {
    traceRayEXT(tlas,              // Top-level acceleration structure
            gl_RayFlagsOpaqueEXT,  // Ray flags, here saying "treat all geometry as opaque"
            0xFF,                  // 8-bit instance mask, here saying "trace against all instances"
            0,                     // SBT record offset
            0,                     // SBT record stride for offset
            0,                     // Miss index
            ray.origin,            // Ray origin
            ray.tMin,              // Minimum t-value
            ray.direction,         // Ray direction
            ray.tMax,              // Maximum t-value
            0);                    // Location of payload

    isect.wo = normalize(-ray.direction);
    isect.position = rPrimaryPld.position;
    isect.lightID = rPrimaryPld.lightID;
    isect.geometric_normal = rPrimaryPld.geometryNormal;
    isect.shading_frame = rPrimaryPld.TBN;
    isect.uv = rPrimaryPld.uv;
    isect.matID = rPrimaryPld.matID;
    isect.lightNormal =  (rPrimaryPld.normalFlipping == 0)
            ? isect.geometric_normal
            : isect.geometric_normal * rPrimaryPld.normalFlipping;
    isect.hitFrontface = rPrimaryPld.hitFrontface;
    return getIntersected(rPrimaryPld.flags);
}

bool intersectEX(
    in const Ray ray,
    out SurfaceIntersection isect,
    inout uint RND
) {
    rPrimaryPld.RND = RND;
    
    traceRayEXT(tlas,              // Top-level acceleration structure
            0,  // Ray flags, here saying "treat all geometry as opaque"
            0xFF,                  // 8-bit instance mask, here saying "trace against all instances"
            0,                     // SBT record offset
            0,                     // SBT record stride for offset
            0,                     // Miss index
            ray.origin,            // Ray origin
            ray.tMin,              // Minimum t-value
            ray.direction,         // Ray direction
            ray.tMax,              // Maximum t-value
            0);                    // Location of payload

    isect.wo = normalize(-ray.direction);
    isect.position = rPrimaryPld.position;
    isect.lightID = rPrimaryPld.lightID;
    isect.geometric_normal = rPrimaryPld.geometryNormal;
    isect.shading_frame = rPrimaryPld.TBN;
    isect.uv = rPrimaryPld.uv;
    isect.matID = rPrimaryPld.matID;
    isect.lightNormal =  (rPrimaryPld.normalFlipping == 0)
            ? isect.geometric_normal
            : isect.geometric_normal * rPrimaryPld.normalFlipping;
    isect.hitFrontface = rPrimaryPld.hitFrontface;
    RND = rPrimaryPld.RND;
    return getIntersected(rPrimaryPld.flags);
}

bool traceOccludeRayEX(
    in Ray ray,
    inout uint RND
) {
    rShadowPld.RND = RND;
    rShadowPld.occluded = false;
    traceRayEXT(tlas,           // Top-level acceleration structure
        0,   // Ray flags, here saying "treat all geometry as opaque"
        0xFF,                   // 8-bit instance mask, here saying "trace against all instances"
        PRIMITIVE_TYPE_COUNT,   // SBT record offset
        0,                      // SBT record stride for offset
        1,                      // Miss index
        ray.origin,             // Ray origin
        ray.tMin,              // Minimum t-value
        ray.direction,          // Ray direction
        ray.tMax,               // Maximum t-value
        1);                     // Location of payload

    RND = rShadowPld.RND;
    return rShadowPld.occluded;
}

bool traceOccludeRay(in Ray ray) {
    rShadowPld.occluded = false;
    traceRayEXT(tlas,           // Top-level acceleration structure
        gl_RayFlagsOpaqueEXT,   // Ray flags, here saying "treat all geometry as opaque"
        0xFF,                   // 8-bit instance mask, here saying "trace against all instances"
        PRIMITIVE_TYPE_COUNT,   // SBT record offset
        0,                      // SBT record stride for offset
        1,                      // Miss index
        ray.origin,             // Ray origin
        ray.tMin,              // Minimum t-value
        ray.direction,          // Ray direction
        ray.tMax,               // Maximum t-value
        1);                     // Location of payload
    return rShadowPld.occluded;
}

vec3 transmittance(
    in SurfaceIntersection isect0,
    in SurfaceIntersection isect1
) {
    const vec3 direction = normalize(isect1.position - isect0.position);

    vec3 offsetDir0 = faceforward(isect0.geometric_normal, -direction, isect0.geometric_normal);
    vec3 offsetDir1 = faceforward(isect1.geometric_normal, +direction, isect1.geometric_normal);
    const vec3 offPos0 = offsetPositionAlongNormal(isect0.position, offsetDir0);
    const vec3 offPos1 = offsetPositionAlongNormal(isect1.position, offsetDir1);

    Ray shadow_ray;
    shadow_ray.origin     = offPos0;
    shadow_ray.tMin       = 0.000;
    shadow_ray.direction  = direction;
    shadow_ray.tMax       = distance(offPos0, offPos1) -0.002;

    bool occluded = traceOccludeRay(shadow_ray);
    return occluded ? vec3(0) : vec3(1);
}

vec3 transmittanceEX(
    in SurfaceIntersection isect0,
    in SurfaceIntersection isect1,
    inout uint RND
) {
    const vec3 direction = normalize(isect1.position - isect0.position);

    vec3 offsetDir0 = faceforward(isect0.geometric_normal, -direction, isect0.geometric_normal);
    vec3 offsetDir1 = faceforward(isect1.geometric_normal, +direction, isect1.geometric_normal);
    const vec3 offPos0 = offsetPositionAlongNormal(isect0.position, offsetDir0);
    const vec3 offPos1 = offsetPositionAlongNormal(isect1.position, offsetDir1);

    Ray shadow_ray;
    shadow_ray.origin     = offPos0;
    shadow_ray.tMin       = 0.000;
    shadow_ray.direction  = direction;
    shadow_ray.tMax       = distance(offPos0, offPos1) -0.002;

    bool occluded = traceOccludeRayEX(shadow_ray, RND);
    return occluded ? vec3(0) : vec3(1);
}
// ********************************************************
//                         BSDF
// ********************************************************

vec3 bsdf_sample(
    in const SurfaceIntersection isect,
    in const vec3 dir_in,
    in const vec2 rnd_uv,
    in const float rnd_w) 
{    
    uint bsdf_type = materials[isect.matID].bsdf_type;

    cBSDFSampleQuery.dir_in = normalize(dir_in);
    cBSDFSampleQuery.mat_id = isect.matID;
    cBSDFSampleQuery.geometric_normal = isect.geometric_normal;
    cBSDFSampleQuery.frame = isect.shading_frame;
    cBSDFSampleQuery.hitFrontface = isect.hitFrontface;
    cBSDFSampleQuery.uv = isect.uv;
    cBSDFSampleQuery.rnd_uv = rnd_uv;
    cBSDFSampleQuery.rnd_w = rnd_w;

    // TODO :: ...
    // executeCallableEXT(BSDF_SAMPLE_IDX(bsdf_type), CALLABLE_LOC_BSDF_SAMPLE);
    executeCallableEXT(BSDF_SAMPLE_IDX(bsdf_type), CALLABLE_LOC_BSDF_SAMPLE);

    return normalize(cBSDFSampleQuery.dir_out);
}

vec3 bsdf_eval(
    in const SurfaceIntersection vertex,
    in const vec3 dir_out,
    in const uint transport_mode
) {
    uint bsdf_type = materials[vertex.matID].bsdf_type;

    cBSDFEvalQuery.dir_in = vertex.wo;
    cBSDFEvalQuery.dir_out = normalize(dir_out);
    cBSDFEvalQuery.mat_id = vertex.matID;
    cBSDFEvalQuery.geometric_normal = vertex.geometric_normal;
    cBSDFEvalQuery.uv = vertex.uv;
    cBSDFEvalQuery.frame = vertex.shading_frame;
    cBSDFEvalQuery.hitFrontface = vertex.hitFrontface;
    cBSDFEvalQuery.transport_mode = transport_mode;
    
    // TODO :: Enable other bsdfs..
    // executeCallableEXT(BSDF_EVAL_IDX(bsdf_type), CALLABLE_LOC_BSDF_EVAL);
    executeCallableEXT(BSDF_EVAL_IDX(bsdf_type), CALLABLE_LOC_BSDF_EVAL);

    return cBSDFEvalQuery.bsdf;
}

vec3 bsdf_eval(
    in const SurfaceIntersection vertex,
    in const vec3 dir_out
) {
    return bsdf_eval(vertex, dir_out, enum_transport_radiance);
}

float bsdf_pdf(
    in const SurfaceIntersection vertex,
    in const vec3 dir_out
) {    
    uint bsdf_type = materials[vertex.matID].bsdf_type;

    cBSDFSamplePDFQuery.dir_in = vertex.wo;
    cBSDFSamplePDFQuery.mat_id = vertex.matID;
    cBSDFSamplePDFQuery.dir_out = normalize(dir_out);
    cBSDFSamplePDFQuery.geometric_normal = vertex.geometric_normal;
    cBSDFSamplePDFQuery.frame = vertex.shading_frame;
    cBSDFSamplePDFQuery.uv = vertex.uv;
    cBSDFSamplePDFQuery.hitFrontface = vertex.hitFrontface;

    // TODO :: ...
    // executeCallableEXT(BSDF_PDF_IDX(bsdf_type), CALLABLE_LOC_BSDF_PDF);
    executeCallableEXT(BSDF_PDF_IDX(bsdf_type), CALLABLE_LOC_BSDF_PDF);

    // Flip the shading frame if it is inconsistent with the geometry normal
    // For Lambertian, we importance sample the cosine hemisphere domain.
    return cBSDFSamplePDFQuery.pdf;
}

float bsdf_pdf(
    in const SurfaceIntersection vertex,
    in const vec3 dir_in,
    in const vec3 dir_out
) {    
    uint bsdf_type = materials[vertex.matID].bsdf_type;

    cBSDFSamplePDFQuery.dir_in = dir_in;
    cBSDFSamplePDFQuery.mat_id = vertex.matID;
    cBSDFSamplePDFQuery.dir_out = dir_out;
    cBSDFSamplePDFQuery.geometric_normal = vertex.geometric_normal;
    cBSDFSamplePDFQuery.frame = vertex.shading_frame;
    cBSDFSamplePDFQuery.uv = vertex.uv;
    cBSDFSamplePDFQuery.hitFrontface = vertex.hitFrontface;

    // TODO :: ...
    // executeCallableEXT(BSDF_PDF_IDX(bsdf_type), CALLABLE_LOC_BSDF_PDF);
    executeCallableEXT(BSDF_PDF_IDX(bsdf_type), CALLABLE_LOC_BSDF_PDF);

    // Flip the shading frame if it is inconsistent with the geometry normal
    // For Lambertian, we importance sample the cosine hemisphere domain.
    return cBSDFSamplePDFQuery.pdf;
}

// ********************************************************
//                         Light
// ********************************************************

struct LightSample {
    vec3 position;
    uint lightID;
    vec3 normal;
    uint lightType;
    vec3 rayDirection;
    vec3 unoffsetPosition;
};

LightSample sampleLight(
    in const vec3 refPoint, 
    in const vec4 rnds
) {
    int lightID = sampleOneLight(rnds.x);
    LightData light = lights[lightID];
    cShapeSampleQuery.ref_point = refPoint;
    cShapeSampleQuery.geometry_id = light.index; // TODO :: flexible ID
    cShapeSampleQuery.uv = vec2(rnds.y, rnds.z);
    cShapeSampleQuery.w = rnds.w;
    cShapeSampleQuery.offset = uvec2(light.sample_dist_offset_cdf_0, light.sample_dist_offset_cdf_1);
    cShapeSampleQuery.size = uvec2(light.sample_dist_size_0, light.sample_dist_size_1);
    executeCallableEXT(SHAPE_SAMPLE_IDX(light.lightType), 0);
    // create light sample
    LightSample lightSample;
    lightSample.position = offsetPositionAlongNormal(cShapeSampleQuery.position, cShapeSampleQuery.normal);
    lightSample.normal = cShapeSampleQuery.normal;
    lightSample.rayDirection = normalize(cShapeSampleQuery.position - cShapeSampleQuery.ref_point);
    lightSample.unoffsetPosition = cShapeSampleQuery.position;
    lightSample.lightID = lightID;
    return lightSample;
}

LightSample sampleLight(in vec3 refPoint, inout uint RNG) {
    vec4 rnds = uniformGenVec4(RNG);
    return sampleLight(refPoint, rnds);
}

float pdf_point_on_light(in vec3 ref_point,
                         in const LightSample lightSample) {
    LightData light = lights[lightSample.lightID];
    float geometry_pdf = 0.f;
    
    cShapeSamplePdfQuery.ref_point = ref_point;
    cShapeSamplePdfQuery.geometry_id = light.index;
    cShapeSamplePdfQuery.sample_position = lightSample.position;
    cShapeSamplePdfQuery.sample_normal = lightSample.normal;
    executeCallableEXT(SHAPE_SAMPLE_PDF_IDX(light.lightType), 1);
    geometry_pdf = cShapeSamplePdfQuery.pdf;

    return light.pmf * geometry_pdf;
}

bool is_light(in SurfaceIntersection vertex) {
    return vertex.lightID != NOT_A_LIGHT;
}

vec3 emission(in uint lightID,
              in vec3 view_dir,
              in float view_footprint,
              in LightSample point_on_light)
{
    if (dot(point_on_light.normal, view_dir) <= 0)
        return vec3(0.f);
    return lights[lightID].intensity;
}

ivec2 uv2pixel(vec2 uv, ivec2 resolution) {
    vec2 pixel_size = vec2(1./resolution.x, 1./resolution.y);
    return ivec2((floor(uv / pixel_size) * pixel_size) * vec2(resolution)) + ivec2(0,1);
    // * should add 1 to y, according to experiment.
    // why? I have not idea though.
}

ivec2 uv2pixel_fix(vec2 uv, ivec2 resolution) {
    vec2 pixel_size = vec2(1./resolution.x, 1./resolution.y);
    return ivec2((floor(uv / pixel_size) * pixel_size) * vec2(resolution));
    // * should add 1 to y, according to experiment.
    // why? I have not idea though.
}

#endif