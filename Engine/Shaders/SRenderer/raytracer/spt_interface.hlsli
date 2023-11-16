#ifndef _SRENDERER_SPT_INTERFACEC_HEADER_
#define _SRENDERER_SPT_INTERFACEC_HEADER_

#include "../include/scene_descriptor_set.hlsli"
#include "../include/common/camera.hlsli"
#include "../include/common/cpp_compatible.hlsli"
#include "../include/common/math.hlsli"
#include "../include/common/random.hlsli"
#include "../include/common/raycast.hlsli"
#include "../include/common/shading.hlsli"
#include "../include/common/packing.hlsli"
#include "../include/raytracer_descriptor_set.hlsli"

#define PRIMITIVE_TRIANGLE 0
#define PRIMITIVE_TRIANGLE_ALPHA 1
#define PRIMITIVE_SPHERE 2
#define PRIMITIVE_SPHERE_ALPHA 3

#define PRIMITIVE_TYPE_COUNT 4

/**********************************************************************
****                    Common Payloads Structures                 ****
**********************************************************************/
/** Geometry hit infomation */
struct GeometryHit {
    float3  position;
    uint    geometryID;
    float3  shadingNormal;
    uint    primitiveID;
    float3  geometryNormal;
    uint    flags;
    float2  barycentric;
    float2  texcoord;
    float4  tangent;
    float3  padding;
    float   lambda;
};

static const uint GeometryHitFlag_HitShift = 0x00;
static const uint GeometryHitFlag_HitMask = 0x01;

static const uint GeometryFlag_FaceForwardShift = 0x01;
static const uint GeometryFlag_FaceForwardMask = 0x01;

static const uint GeometryTypeFlag_HitShift = 0x02;
static const uint GeometryTypeFlag_HitMask = 0xff;


bool HasHit(in_ref(GeometryHit) hit) {
    return ((hit.flags >> GeometryHitFlag_HitShift) & GeometryHitFlag_HitMask) != 0;
}
void SetHit(inout_ref(GeometryHit) hit, bool value) {
    // if not hit, reset the geometryID and primitiveID
    if (!value) { hit.geometryID = uint(-1); hit.primitiveID = uint(-1); }
    hit.flags = (hit.flags & ~(GeometryHitFlag_HitMask << GeometryHitFlag_HitShift)) 
            | (value ? 1 : 0) << GeometryHitFlag_HitShift;
}

bool IsFaceForward(in_ref(GeometryHit) hit) {
    return ((hit.flags >> GeometryFlag_FaceForwardShift) & GeometryFlag_FaceForwardShift) != 0;
}
void SetFaceForward(inout_ref(GeometryHit) hit, bool value) {
    hit.flags = (hit.flags & ~(GeometryFlag_FaceForwardMask << GeometryFlag_FaceForwardShift)) 
            | (value ? 1 : 0) << GeometryFlag_FaceForwardShift;
}

/** Primary Payload Struct */
struct PrimaryPayload {
    GeometryHit hit;
    RandomSamplerState RND;
};

/** Primary Payload Struct */
struct ShadowPayload {
    bool occluded;
    RandomSamplerState RND;
};

GeometryHit CreateGeometryHit(in_ref(ShadingSurface) surface) {
    GeometryHit hit;
    hit.geometryID = 0;
    hit.primitiveID = 0;
    hit.position = surface.worldPos;
    hit.shadingNormal = surface.shadingNormal;
    hit.geometryNormal = surface.geometryNormal;
    hit.barycentric = float2(0);
    hit.texcoord = float2(0);
    hit.tangent = float4(0);
    hit.flags = 0;
    return hit;
}

/**********************************************************************
****                    Common Payloads Structures                 ****
**********************************************************************/

struct SurfaceIntersection {
    float3 position;         // hit position
    uint geometryID;         // geometry ID
    float3 geometric_normal; // surface normal
    uint lightID;            // light ID
    float3 wo;               // negative ray direction
    uint matID;              // material ID
    float2 uv;               // uv for texture fetching
    float uv_screen_size;    // screen uv size
    float mean_curvature;    // for ray differential propagation

    float ray_radius;       // for ray differential propagation
    float3x3 shading_frame; // shading frame
    // vec3 offsetedPosition;
    // vec3 offsetedPositionInv;
    float3 lightNormal;
    float hitFrontface;
};

// offsetPositionAlongNormal shifts a point on a triangle surface so that a
// ray bouncing off the surface with tMin = 0.0 is no longer treated as
// intersecting the surface it originated from.
//
// Here's the old implementation of it we used in earlier chapters:
// vec3 offsetPositionAlongNormal(vec3 worldPosition, vec3 normal)
// {
//   return worldPosition + 0.0001 * normal;
// }
//
// However, this code uses an improved technique by Carsten Wächter and
// Nikolaus Binder from "A Fast and Robust Method for Avoiding
// Self-Intersection" from Ray Tracing Gems (verion 1.7, 2020).
// The normal can be negated if one wants the ray to pass through
// the surface instead.

float3 offsetPositionAlongNormal(float3 worldPosition, float3 normal) {
    // Convert the normal to an integer offset.
    const float int_scale = 256.0f;
    const int3 of_i = int3(int_scale * normal);
    // Offset each component of worldPosition using its binary representation.
    // Handle the sign bits correctly.
    const float3 p_i = float3( //
        asfloat(asint(worldPosition.x) + ((worldPosition.x < 0) ? -of_i.x : of_i.x)),
        asfloat(asint(worldPosition.y) + ((worldPosition.y < 0) ? -of_i.y : of_i.y)),
        asfloat(asint(worldPosition.z) + ((worldPosition.z < 0) ? -of_i.z : of_i.z)));
    // Use a floating-point offset instead for points near (0,0,0), the origin.
    const float origin = 1.0f / 32.0f;
    const float floatScale = 1.0f / 65536.0f;
    return float3( //
        abs(worldPosition.x) < origin ? worldPosition.x + floatScale * normal.x : p_i.x,
        abs(worldPosition.y) < origin ? worldPosition.y + floatScale * normal.y : p_i.y,
        abs(worldPosition.z) < origin ? worldPosition.z + floatScale * normal.z : p_i.z);
}

Ray spawnRay(
    in const SurfaceIntersection isect,
    in const float3 dir
) {
    float3 offsetDir = faceforward(isect.geometric_normal, -dir, isect.geometric_normal);
    float3 offsetedPosition = offsetPositionAlongNormal(isect.position, offsetDir);

    Ray ray;
    ray.origin = offsetedPosition;
    ray.direction = dir;
    ray.tMin = 0.000;
    ray.tMax = k_inf;
    return ray;
}

RayDesc ToRayDesc (in_ref(Ray) ray) {
    RayDesc raydesc = {};
    raydesc.Origin = ray.origin;
    raydesc.Direction = ray.direction;
    raydesc.TMin = ray.tMin;
    raydesc.TMax = ray.tMax;
    return raydesc;
}

struct Attributes {
    float2 uv;
};

// ------------------------------------------------------------
// Query structures for shape sampling
// ------------------------------------------------------------
// Payload structure for shape sample callable.
struct ShapeSampleQuery {
    // input
    float3 ref_point;
    uint geometry_id;
    float2 uv; // for selecting a point on a 2D surface
    float w;   // for selecting triangles
    uint primitive_id;
    uint2 offset;
    uint2 size;
    // output
    float3 position;
    float pdf;
    float3 normal;
};

struct ShapeSamplePdfQuery {
    // input
    float3 ref_point;
    uint geometry_id;
    float3 sample_position;
    float3 sample_normal;
    // output
    float pdf;
};

struct SplitShading {
    float3 diffuse;
    float3 specular;
};

SplitShading add(in_ref(SplitShading) a, in_ref(SplitShading) b) {
    SplitShading result;
    result.diffuse = a.diffuse + b.diffuse;
    result.specular = a.specular + b.specular;
    return result;
}

// ------------------------------------------------------------
// Common structures for various kind of queries
// ------------------------------------------------------------
/** Enum for transport mode. */
static const uint enum_transport_radiance = 0;
static const uint enum_transport_importance = 1;
/** QueryBitfield describes some misc info. */
struct QueryBitfield {
    uint transport_mode; // transport mode
    bool face_forward;   // true: hit front face // false: hit back face
    bool split_query;    // true: split the demodulate diffuse and specular // false: combined BSDF
};
/** Pack the query bitfield into a single uint. */
uint PackQueryBitfield(in_ref(QueryBitfield) bitfield) {
    uint flag = 0;
    flag |= bitfield.transport_mode << 0;
    flag |= bitfield.face_forward << 1;
    flag |= bitfield.split_query << 2;
    return flag;
}
/** Unpack the packed flag into a query bitfield. */
QueryBitfield UnpackQueryBitfield(in_ref(uint) flag) {
    QueryBitfield bitfield;
    bitfield.transport_mode = (flag >> 0) & 0x1;
    bitfield.face_forward = ((flag >> 1) & 0x1) != 0;
    bitfield.split_query = ((flag >> 2) & 0x1) != 0;
    return bitfield;
}

// ------------------------------------------------------------
// Query structures for bsdf evaluation and sampling
// ------------------------------------------------------------
// Payload structure for evaluating the bsdf.
struct BSDFEvalQuery {
    // input
    float3 dir_in; // ----------- 4 floats
    uint mat_id;
    float3 dir_out; // ---------- 4 floats
    uint misc_flag;
    float3 geometric_normal; // - 4 floats
    float2 uv;
    // output
    float3 bsdf;
    // shading frame - 3x3 floats
    float3x3 frame; // shading frame
};
// Payload structure for sampling the bsdf.
struct BSDFSampleQuery {
    // input
    float3 dir_in; // ----------- 4 floats
    uint mat_id;
    float3 geometric_normal; // - 4 floats
    float rnd_w;
    float2 uv; // --------------- 4 floats
    float2 rnd_uv;
    // output
    float3 dir_out; // ---------- 4 floats
    float pdf_out;
    // shading frame - 3x3 floats
    float3x3 frame; // shading frame
    uint misc_flag; // misc flag
};
// Payload structure for getting sample pdf.
struct BSDFSamplePDFQuery {
    // input
    float3 dir_in; // ----------- 4 floats
    uint mat_id;
    float3 dir_out; // ---------- 4 floats
    uint misc_flag;
    float3 geometric_normal; // - 4 floats
    float packedInfo0;
    float2 uv; // --------------- 4 floats
    float packedInfo1;
    // output
    float pdf;
    float3x3 frame;
};
// Payload structure for evaluating the bsdf with differential.
struct BSDFEvalDiffQuery {
    // input
    float3 dir_in; // ----------- 4 floats
    uint mat_id;
    float3 dir_out; // ---------- 4 floats
    uint misc_flag;
    float3 geometric_normal; // - 4 floats
    float lod;
    float3 adjoint_gradient;
    float2 uv;
    // output
    float3 bsdf;
    // shading frame - 3x3 floats
    float3x3 frame; // shading frame
};
bool TraceOccludeRay(
    in_ref(Ray) ray,
    inout_ref(RandomSamplerState) RND,
    RaytracingAccelerationStructure bvh
) {
    ShadowPayload shadowPld;
    shadowPld.RND = RND;
    shadowPld.occluded = false;

    RayDesc raydesc = ToRayDesc(ray);
    TraceRay(bvh, 0, 0xff, PRIMITIVE_TYPE_COUNT, 0, 1, raydesc, shadowPld);

    RND = shadowPld.RND;
    return shadowPld.occluded;
}

Ray SpawnRay(
    in_ref(float3) position,
    in_ref(float3) flat_normal,
    in_ref(float3) dir
) {
    const float3 offsetDir = faceforward(flat_normal, -dir, flat_normal);
    const float3 offsetedPosition = offsetPositionAlongNormal(position, offsetDir);
    Ray ray;
    ray.origin = offsetedPosition;
    ray.direction = dir;
    ray.tMin = 0.000;
    ray.tMax = k_inf;
    return ray;
}

Ray SpawnRay(
    in_ref(GeometryHit) isect,
    in_ref(float3) dir)
{
    const float3 offsetDir = faceforward(isect.geometryNormal, -dir, isect.geometryNormal);
    const float3 offsetedPosition = offsetPositionAlongNormal(isect.position, offsetDir);

    Ray ray;
    ray.origin = offsetedPosition;
    ray.direction = dir;
    ray.tMin = 0.000;
    ray.tMax = k_inf;
    return ray;
}

Ray SpawnRay(
    in_ref(ShadingSurface) surface,
    in_ref(float3) dir)
{
    const float3 offsetDir = faceforward(surface.geometryNormal, -dir, surface.geometryNormal);
    const float3 offsetedPosition = offsetPositionAlongNormal(surface.worldPos, offsetDir);

    Ray ray;
    ray.origin = offsetedPosition;
    ray.direction = dir;
    ray.tMin = 0.000;
    ray.tMax = k_inf;
    return ray;
}

Ray SpawnOcclusionRay(
    in_ref(GeometryHit) isect,
    in_ref(float3) target_pos,
    in_ref(float3) target_norm,
    in_ref(float3) dir)
{
    const float3 offsetDir = faceforward(isect.geometryNormal, -dir, isect.geometryNormal);
    const float3 offsetedPosition = offsetPositionAlongNormal(isect.position, offsetDir);
    const float3 offsetDirTarget = faceforward(target_norm, dir, target_norm);
    const float3 offsetedTarget = offsetPositionAlongNormal(target_pos, offsetDirTarget);
    
    Ray ray;
    ray.origin = offsetedPosition;
    ray.direction = dir;
    ray.tMin = 0.000;
    ray.tMax = length(offsetedTarget - offsetedPosition);
    return ray;
}

Ray SetupVisibilityRay(
    in_ref(ShadingSurface) surface,
    in_ref(float3) samplePosition, 
    float offset = 0.001
) {
    float3 direction = samplePosition - surface.worldPos;
    const float len = length(direction);
    direction /= len;
    Ray ray = SpawnRay(surface, direction);
    ray.tMax = max(offset, len - offset * 2);
    return ray;
}

/*************************************************************************
 *                    Material BSDF Sample Queries                       *
 *************************************************************************
 * In this section, SPT provides a set of functions to evaluate BSDFs.   *
 * These functions are used in various BSDF queries, which handles the   *
 * polymorphic nature of BSDFs. Any BSDF can be evaluated by calling     *
 * EvalBsdf, SampleBsdf, and PdfBsdfSample queries.                      *
 *************************************************************************/

 // Helper macro to define BSDF query indices.
#define BSDF_OFFSET 0
#define BSDF_EVAL_IDX(BSDF_ID) (BSDF_OFFSET + ((BSDF_ID)*3) + 0)
#define BSDF_SAMPLE_IDX(BSDF_ID) (BSDF_OFFSET + ((BSDF_ID)*3) + 1)
#define BSDF_PDF_IDX(BSDF_ID) (BSDF_OFFSET + ((BSDF_ID)*3) + 2)

/**
 * Evaluate the BSDF.
 * @param hit The geometry hit.
 * @param dir_in The direction of the incoming ray.
 * @param dir_out The direction of the outgoing ray.
 * @param transport_mode The transport mode.
 * @return The evaluated BSDF value.
 */
float3 EvalBsdf(
    in_ref(GeometryHit) hit,
    in_ref(float3) dir_in,
    in_ref(float3) dir_out,
    in const uint transport_mode = 0
) {
    uint materialID = geometries[hit.geometryID].materialID;
    BSDFEvalQuery cBSDFEvalQuery;
    uint bsdf_type = materials[materialID].bsdfID;
    cBSDFEvalQuery.dir_in = dir_in;
    cBSDFEvalQuery.dir_out = dir_out;
    cBSDFEvalQuery.mat_id = materialID;
    cBSDFEvalQuery.geometric_normal = hit.geometryNormal;
    cBSDFEvalQuery.uv = hit.texcoord;
    cBSDFEvalQuery.frame = createONB(hit.shadingNormal);
    QueryBitfield flag;
    flag.transport_mode = transport_mode;
    flag.face_forward = IsFaceForward(hit);
    flag.split_query = false;
    cBSDFEvalQuery.misc_flag = PackQueryBitfield(flag);
    CallShader(BSDF_EVAL_IDX(bsdf_type), cBSDFEvalQuery);
    return cBSDFEvalQuery.bsdf;
}

SplitShading EvalBsdfSplit(
    in_ref(GeometryHit) hit,
    in_ref(float3) dir_in,
    in_ref(float3) dir_out,
    in const uint transport_mode = 0,
) {
    uint materialID = geometries[hit.geometryID].materialID;
    BSDFEvalQuery cBSDFEvalQuery;
    uint bsdf_type = materials[materialID].bsdfID;
    cBSDFEvalQuery.dir_in = dir_in;
    cBSDFEvalQuery.dir_out = dir_out;
    cBSDFEvalQuery.mat_id = materialID;
    cBSDFEvalQuery.geometric_normal = hit.geometryNormal;
    cBSDFEvalQuery.uv = hit.texcoord;
    cBSDFEvalQuery.frame = createONB(hit.shadingNormal);
    QueryBitfield flag;
    flag.transport_mode = transport_mode;
    flag.face_forward = IsFaceForward(hit);
    flag.split_query = true;
    cBSDFEvalQuery.misc_flag = PackQueryBitfield(flag);
    CallShader(BSDF_EVAL_IDX(bsdf_type), cBSDFEvalQuery);
    // unpack the result
    SplitShading result;
    result.diffuse = cBSDFEvalQuery.bsdf;
    result.specular = cBSDFEvalQuery.dir_out;
    return result;
}
/**
 * Evaluate the BSDF.
 * @param surface The shading surface.
 * @param dir_in The direction of the incoming ray.
 * @param dir_out The direction of the outgoing ray.
 * @param transport_mode The transport mode.
 * @return The evaluated BSDF value.
 */
float3 EvalBsdf(
    in_ref(ShadingSurface) surface,
    in_ref(float3) dir_in,
    in_ref(float3) dir_out,
    in const uint transport_mode = 0
) {
    BSDFEvalQuery cBSDFEvalQuery;
    uint bsdf_type = surface.bsdfID;
    cBSDFEvalQuery.dir_in = dir_in;
    cBSDFEvalQuery.dir_out = dir_out;
    cBSDFEvalQuery.mat_id = 0xFFFFFFFF;
    cBSDFEvalQuery.geometric_normal = surface.geometryNormal;
    cBSDFEvalQuery.frame = createONB(surface.shadingNormal);
    QueryBitfield flag;
    flag.transport_mode = transport_mode;
    flag.face_forward = surface.faceForward;
    flag.split_query = false;
    cBSDFEvalQuery.misc_flag = PackQueryBitfield(flag);
    // pack shading surface data into query struct
    // we can pack up to 6 floats into the query struct here:
    // uv (2) bsdf (4)
    cBSDFEvalQuery.uv.x = surface.roughness;                             // roughness
    cBSDFEvalQuery.uv.y = asfloat(PackRGBE(surface.specularF0));         // specularF0
    cBSDFEvalQuery.bsdf.x = asfloat(PackRGBE(surface.diffuseAlbedo));    // diffuseAlbedo
    cBSDFEvalQuery.bsdf.y = surface.transmissionFactor;               // eta
    // call shader
    CallShader(BSDF_EVAL_IDX(bsdf_type), cBSDFEvalQuery);
    return cBSDFEvalQuery.bsdf;
}

SplitShading EvalBsdfSplit(
    in_ref(ShadingSurface) surface,
    in_ref(float3) dir_in,
    in_ref(float3) dir_out,
    in const uint transport_mode = 0,
) {
    BSDFEvalQuery cBSDFEvalQuery;
    uint bsdf_type = surface.bsdfID;
    cBSDFEvalQuery.dir_in = dir_in;
    cBSDFEvalQuery.dir_out = dir_out;
    cBSDFEvalQuery.mat_id = 0xFFFFFFFF;
    cBSDFEvalQuery.geometric_normal = surface.geometryNormal;
    cBSDFEvalQuery.frame = createONB(surface.shadingNormal);
    QueryBitfield flag;
    flag.transport_mode = transport_mode;
    flag.face_forward = surface.faceForward;
    flag.split_query = true;
    cBSDFEvalQuery.misc_flag = PackQueryBitfield(flag);
    // pack shading surface data into query struct
    // we can pack up to 6 floats into the query struct here:
    // uv (2) bsdf (4)
    cBSDFEvalQuery.uv.x = surface.roughness;                          // roughness
    cBSDFEvalQuery.uv.y = asfloat(PackRGBE(surface.specularF0));      // specularF0
    cBSDFEvalQuery.bsdf.x = asfloat(PackRGBE(surface.diffuseAlbedo)); // diffuseAlbedo
    cBSDFEvalQuery.bsdf.y = surface.transmissionFactor;               // eta
    // call shader
    CallShader(BSDF_EVAL_IDX(bsdf_type), cBSDFEvalQuery);
    // unpack the result
    SplitShading result;
    result.diffuse = cBSDFEvalQuery.bsdf;
    result.specular = cBSDFEvalQuery.dir_out;
    return result;
}
/**
 * Sample a direction from the BSDF.
 * @param hit The geometry hit.
 * @param dir_in The direction of the incoming ray.
 * @param rand The random number triple to generate sample.
 * @param pdf_out The output pdf (in solid-angle measure).
 * @return The sampled direction.
 */
float3 SampleBsdf(
    in_ref(GeometryHit) hit,
    in_ref(float3) dir_in,
    in_ref(float3) rand,
    out_ref(float) pdf_out,
) {
    uint materialID = geometries[hit.geometryID].materialID;
    uint bsdf_type = materials[materialID].bsdfID;
    BSDFSampleQuery cBSDFSampleQuery;
    cBSDFSampleQuery.dir_in = dir_in;
    cBSDFSampleQuery.mat_id = materialID;
    cBSDFSampleQuery.geometric_normal = hit.geometryNormal;
    cBSDFSampleQuery.uv = hit.texcoord;
    cBSDFSampleQuery.frame = createONB(hit.shadingNormal);
    cBSDFSampleQuery.rnd_uv = rand.xy;
    cBSDFSampleQuery.rnd_w = rand.z;
    QueryBitfield flag;
    flag.transport_mode = 0;
    flag.face_forward = IsFaceForward(hit);
    cBSDFSampleQuery.misc_flag = PackQueryBitfield(flag);
    CallShader(BSDF_SAMPLE_IDX(bsdf_type), cBSDFSampleQuery);
    pdf_out = cBSDFSampleQuery.pdf_out;
    return cBSDFSampleQuery.dir_out;
}

/**
 * Sample a direction from the BSDF.
 * @param surface The shading surface hit.
 * @param dir_in The direction of the incoming ray.
 * @param rand The random number triple to generate sample.
 * @param pdf_out The output pdf (in solid-angle measure).
 * @return The sampled direction.
 */
float3 SampleBsdf(
    in_ref(ShadingSurface) surface,
    in_ref(float3) dir_in,
    in_ref(float3) rand,
    out_ref(float) pdf_out,
) {
    const uint bsdf_type = surface.bsdfID;
    BSDFSampleQuery cBSDFSampleQuery;
    cBSDFSampleQuery.dir_in = dir_in;
    cBSDFSampleQuery.mat_id = 0xFFFFFFFF;
    cBSDFSampleQuery.geometric_normal = surface.geometryNormal;
    cBSDFSampleQuery.frame = createONB(surface.shadingNormal);
    cBSDFSampleQuery.rnd_uv = rand.xy;
    cBSDFSampleQuery.rnd_w = rand.z;
    // pack shading surface data into query struct
    // we can pack up to 6 floats into the query struct here:
    // uv (2) dir_out (3) pdf_out (1)
    cBSDFSampleQuery.uv.x = surface.roughness; // roughness
    cBSDFSampleQuery.uv.y = asfloat(PackRGBE(surface.specularF0)); // specularF0
    cBSDFSampleQuery.dir_out.x = asfloat(PackRGBE(surface.diffuseAlbedo)); // diffuseAlbedo
    cBSDFSampleQuery.dir_out.y = surface.transmissionFactor;               // eta
    QueryBitfield flag;
    flag.transport_mode = 0;
    flag.face_forward = surface.faceForward;
    cBSDFSampleQuery.misc_flag = PackQueryBitfield(flag);
    // call shader
    CallShader(BSDF_SAMPLE_IDX(bsdf_type), cBSDFSampleQuery);
    pdf_out = cBSDFSampleQuery.pdf_out;
    return cBSDFSampleQuery.dir_out;
}

/**
 * The pdf of sampling a direction from the BSDF.
 * @param hit The geometry hit.
 * @param dir_in The direction of the incoming ray.
 * @param dir_out The direction of the outgoing ray.
 * @return The pdf of selecting the sampled direction.
 */
float PdfBsdfSample(
    in_ref(GeometryHit) hit,
    in_ref(float3) dir_in,
    in_ref(float3) dir_out
) {
    uint materialID = geometries[hit.geometryID].materialID;
    uint bsdf_type = materials[materialID].bsdfID;
    BSDFSamplePDFQuery cBSDFSamplePDFQuery;
    cBSDFSamplePDFQuery.dir_in = dir_in;
    cBSDFSamplePDFQuery.dir_out = dir_out;
    cBSDFSamplePDFQuery.mat_id = materialID;
    cBSDFSamplePDFQuery.geometric_normal = hit.geometryNormal;
    cBSDFSamplePDFQuery.uv = hit.texcoord;
    cBSDFSamplePDFQuery.frame = createONB(hit.shadingNormal);
    QueryBitfield flag;
    flag.transport_mode = 0;
    flag.face_forward = IsFaceForward(hit);
    cBSDFSamplePDFQuery.misc_flag = PackQueryBitfield(flag);
    CallShader(BSDF_PDF_IDX(bsdf_type), cBSDFSamplePDFQuery);
    return cBSDFSamplePDFQuery.pdf;
}

/**
 * The pdf of sampling a direction from the BSDF.
 * @param hit The geometry hit.
 * @param dir_in The direction of the incoming ray.
 * @param dir_out The direction of the outgoing ray.
 * @return The pdf of selecting the sampled direction.
 */
float PdfBsdfSample(
    in_ref(ShadingSurface) surface,
    in_ref(float3) dir_in,
    in_ref(float3) dir_out
) {
    const uint bsdf_type = surface.bsdfID;
    BSDFSamplePDFQuery cBSDFSamplePDFQuery;
    cBSDFSamplePDFQuery.dir_in = dir_in;
    cBSDFSamplePDFQuery.dir_out = dir_out;
    cBSDFSamplePDFQuery.geometric_normal = surface.geometryNormal;
    cBSDFSamplePDFQuery.frame = createONB(surface.shadingNormal);
    cBSDFSamplePDFQuery.mat_id = 0xFFFFFFFF;
    // pack shading surface data into query struct
    // we can pack up to 6 floats into the query struct here:
    // uv (2) dir_out (3) pdf_out (1)
    cBSDFSamplePDFQuery.uv.x = surface.roughness;                             // roughness
    cBSDFSamplePDFQuery.uv.y = asfloat(PackRGBE(surface.specularF0));         // specularF0
    cBSDFSamplePDFQuery.packedInfo0 = asfloat(PackRGBE(surface.diffuseAlbedo)); // diffuseAlbedo
    cBSDFSamplePDFQuery.packedInfo1 = surface.transmissionFactor; // eta
    QueryBitfield flag;
    flag.transport_mode = 0;
    flag.face_forward = surface.faceForward;
    cBSDFSamplePDFQuery.misc_flag = PackQueryBitfield(flag);
    // call shader
    CallShader(BSDF_PDF_IDX(bsdf_type), cBSDFSamplePDFQuery);
    return cBSDFSamplePDFQuery.pdf;
}

/**
 * Spawn a ray from a geometry hit with a 
 * random direction sampled from the BSDF.
 * @param hit The geometry hit.
 * @param dir_in The direction of the incoming ray.
 * @param rand The random number triple to generate sample.
 * @param pdf_out The output pdf (in solid-angle measure).
 * @return The spawned ray.
 */
Ray SpawnBsdfRay(
    in_ref(GeometryHit) hit,
    in_ref(float3) dir_in,
    in_ref(float3) rand,
    out_ref(float) pdf_out,
) {
    const float3 bsdf_direction = SampleBsdf(hit, dir_in, rand, pdf_out);
    return SpawnRay(hit, bsdf_direction);
}

/**
 * Spawn a ray from a shading surface with a
 * random direction sampled from the BSDF.
 * @param surface The shading surface.
 * @param dir_in The direction of the incoming ray.
 * @param rand The random number triple to generate sample.
 * @param pdf_out The output pdf (in solid-angle measure).
 * @return The spawned ray.
 */
Ray SpawnBsdfRay(
    in_ref(ShadingSurface) surface,
    in_ref(float3) dir_in,
    in_ref(float3) rand,
    out_ref(float) pdf_out,
) {
    const float3 bsdf_direction = SampleBsdf(surface, dir_in, rand, pdf_out);
    return SpawnRay(surface, bsdf_direction);
}

/**
 * Spawn a ray from a geometry hit 
 * with a random direction sampled from the BSDF.
 * @param hit The geometry hit.
 * @param dir_in The direction of the incoming ray.
 * @param RNG The random number generator.
 * @param pdf_out The output pdf (in solid-angle measure).
 * @return The spawned ray.
 */
Ray SpawnBsdfRay(
    in_ref(GeometryHit) hit,
    in_ref(float3) dir_in,
    inout_ref(RandomSamplerState) RNG,
    out_ref(float) pdf_out,
) {
    const float3 bsdf_rnd = float3(GetNextRandom(RNG), GetNextRandom(RNG), GetNextRandom(RNG));
    return SpawnBsdfRay(hit, dir_in, bsdf_rnd, pdf_out);
}

/**
 * Spawn a ray from a shading surface with a
 * random direction sampled from the BSDF.
 * @param surface The shading surface.
 * @param dir_in The direction of the incoming ray.
 * @param rand The random number triple to generate sample.
 * @param pdf_out The output pdf (in solid-angle measure).
 * @return The spawned ray.
 */
Ray SpawnBsdfRay(
    in_ref(ShadingSurface) surface,
    in_ref(float3) dir_in,
    inout_ref(RandomSamplerState) RNG,
    out_ref(float) pdf_out,
) {
    const float3 bsdf_rnd = float3(GetNextRandom(RNG), GetNextRandom(RNG), GetNextRandom(RNG));
    return SpawnBsdfRay(surface, dir_in, bsdf_rnd, pdf_out);
}
void Intersection(
    in_ref(Ray) ray,
    in_ref(RaytracingAccelerationStructure) bvh,
    inout_ref(PrimaryPayload) payload,
    inout_ref(RandomSamplerState) RNG
) {
    payload.RND = RNG;
    TraceRay(bvh, RAY_FLAG_NONE, 0xff, 0, 0, 0, ToRayDesc(ray), payload);
    RNG = payload.RND;
}

#endif // !_SRENDERER_SPT_INTERFACEC_HEADER_