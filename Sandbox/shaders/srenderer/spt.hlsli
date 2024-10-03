#ifndef _SRENDERER_SPT_HEADER_
#define _SRENDERER_SPT_HEADER_

#include "common/geometry.hlsli"
#include "common/random.hlsli"
#include "common/raycast.hlsli"

/**********************************************************************
****                    Common Payloads Structures                 ****
**********************************************************************/
/** Geometry hit infomation */
struct GeometryHit {
    float3 position;
    uint geometryID;
    float3 shadingNormal;
    uint primitiveID;
    float3 geometryNormal;
    uint flags;
    float2 barycentric;
    float2 texcoord;
    float4 tangent;
    float3 padding;
    float lambda;
};

/** Primary Payload Struct */
struct PrimaryPayload {
    GeometryHit hit;
    RandomSamplerState RND;
};

struct RawPayload {
    int primitiveID;
    int geometryID;
    float2 barycentric;
    bool hasHit;
};

struct MediumHit {
    int mediumID;
};

// ------------------------------------------------------------
// Query structures for bsdf evaluation and sampling
// ------------------------------------------------------------
// Payload structure for evaluating the bsdf.
struct BSDFEvalGeometry {
    // input
    float3 dir_in; // ----------- 4 floats
    float3 dir_out; // ---------- 4 floats
    float3 geometric_normal; // - 4 floats
    // output
    float3 bsdf;
    // shading frame - 3x3 floats
    float3x3 frame; // shading frame
};

static const uint GeometryHitFlag_HitShift = 0x00;
static const uint GeometryHitFlag_HitMask = 0x01;

static const uint GeometryFlag_FaceForwardShift = 0x01;
static const uint GeometryFlag_FaceForwardMask = 0x01;

static const uint GeometryTypeFlag_HitShift = 0x02;
static const uint GeometryTypeFlag_HitMask = 0xff;

bool HasHit(in const GeometryHit hit) { return ((hit.flags >> GeometryHitFlag_HitShift) & GeometryHitFlag_HitMask) != 0; }
void SetHit(inout GeometryHit hit, bool value) {
    // if not hit, reset the geometryID and primitiveID
    if (!value) { hit.geometryID = uint(-1); hit.primitiveID = uint(-1); }
    hit.flags = (hit.flags & ~(GeometryHitFlag_HitMask << GeometryHitFlag_HitShift)) 
            | (value ? 1 : 0) << GeometryHitFlag_HitShift;
}

bool IsFaceForward(in const GeometryHit hit) { return ((hit.flags >> GeometryFlag_FaceForwardShift) & GeometryFlag_FaceForwardShift) != 0; }
void SetFaceForward(inout GeometryHit hit, bool value) {
    hit.flags = (hit.flags & ~(GeometryFlag_FaceForwardMask << GeometryFlag_FaceForwardShift)) 
            | (value ? 1 : 0) << GeometryFlag_FaceForwardShift;
}

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

Ray SpawnRay(float3 pos, float3 dir) {
    Ray ray;
    ray.origin = pos;
    ray.direction = dir;
    ray.tMin = 0.000;
    ray.tMax = 1e6;
    return ray;
}

Ray SpawnVisibilityRay(
    in_ref(GeometryHit) isect,
    in_ref(float3) position)
{
    float3 dir = position - isect.position;
    float distance = length(dir);
    Ray visiblityRay = SpawnRay(isect, dir / distance);
    visiblityRay.tMax = distance - min(0.01, distance * 0.02);
    return visiblityRay;
}

Ray SpawnVisibilityRay(
    in_ref(float3) origin,
    in_ref(float3) target)
{
    float3 dir = target - origin;
    float distance = length(dir);
    Ray visiblityRay = SpawnRay(origin, dir / distance);
    visiblityRay.tMax = distance - min(0.01, distance * 0.02);
    return visiblityRay;
}

BSDFEvalGeometry createBSDFEvalGeometry(
    in const GeometryHit hit,
    in const float3 dir_in,
    in const float3 dir_out,
) {
    BSDFEvalGeometry evalGeometry;
    evalGeometry.dir_in = dir_in;
    evalGeometry.dir_out = dir_out;
    evalGeometry.geometric_normal = hit.geometryNormal;
    evalGeometry.frame = createONB(hit.shadingNormal);
    return evalGeometry;
}

#endif // _SRENDERER_SPT_HEADER_