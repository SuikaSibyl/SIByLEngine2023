#ifndef _SRENDERER_ADDON_SLC_COMMON_HEADER_
#define _SRENDERER_ADDON_SLC_COMMON_HEADER_

#include "../../../include/common/math.hlsli"
#include "../../../include/common/cpp_compatible.hlsli"

#define MAX_CUT_NODES 32

// 48 bytes
struct Node {
    float3 boundMin;
    float intensity;
    float3 boundMax;
    int ID;
    float4 cone; // xyz cone axis, w cone angle
};

struct VizNode {
    float3 boundMin;
    float3 boundMax;
    int level;
    int index;
};

struct BLASInstanceHeader {
    float3x3 rotation;
    float3 translation;
    float scaling;
    float3 emission;
    int nodeOffset;
    int numTreeLevels; // not used
    int numTreeLeafs;
    int emitTexId; // -1 -> no texture
    int BLASId;
};

struct EmissiveVertex {
    float3 position;
    float3 normal;
    float2 texCoord;
};

struct MeshLightInstancePrimtive {
    int indexOffset; // indexoffset in the indinces buffer of mesh lights
    int instanceId;  // points to BLASInstanceHeader
};

inline float GetColorIntensity(float3 color) {
    return color.r + color.g + color.b;
}

inline float4 MergeCones(float4 cone1, float4 cone2) {
    float4 ret;
    float3 axis_a = 0;
    float3 axis_b = 0;

    float angle_a = 0;
    float angle_b = 0;

    if (cone1.w >= cone2.w) {
        axis_a = cone1.xyz;
        axis_b = cone2.xyz;
        angle_a = cone1.w;
        angle_b = cone2.w;
    }
    else {
        axis_a = cone2.xyz;
        axis_b = cone1.xyz;
        angle_a = cone2.w;
        angle_b = cone1.w;
    }

    float cosGamma = max(-1.f, min(1.f, dot(axis_a, axis_b)));
    float gamma = acos(cosGamma);

    if (cosGamma > 0.9999) {
        ret.xyz = axis_a;
        ret.w = min(angle_a /* + gamma*/, k_pi);
        return ret;
    }

    if (cosGamma < -0.9999) {
        ret.xyz = axis_a;
        ret.w = k_pi;
        return ret;
    }

    if (min(gamma + angle_b, k_pi) <= angle_a) {
        ret.xyz = axis_a;
        ret.w = angle_a;
        return ret;
    }

    ret.w = (angle_a + angle_b + gamma) / 2;

    if (ret.w >= k_pi) {
        ret.xyz = axis_a;
        ret.w = k_pi;
        return ret;
    }

    float rot = ret.w - angle_a;
    
    // slerp(axis_a, axis_b, rot / gamma);
    float t = rot / gamma;
    ret.xyz = (sin((1 - t) * gamma) * axis_a + sin(t * gamma) * axis_b) / sin(gamma);
    // need normalization?
    return ret;
}

void Swap(inout int first, inout int second)
{
    int temp = first;
    first = second;
    second = temp;
}

/**
 * Compute the maximum distance along the direction dir to the bounding box bound.
 * @param p The origin of the ray.
 * @param dir The direction of the ray.
 * @param boundMin The min corner of the bounding box.
 * @param boundMax The max corner of the bounding box
 */
inline float MaxDistAlong(float3 p, float3 dir, float3 boundMin, float3 boundMax) {
    float3 dir_p = dir * p;
    float3 mx0 = dir * boundMin - dir_p;
    float3 mx1 = dir * boundMax - dir_p;
    return max(mx0[0], mx1[0]) + max(mx0[1], mx1[1]) + max(mx0[2], mx1[2]);
}

inline void CoordinateSystem_(float3 v1, out float3 v2, out float3 v3)
{
    if (abs(v1.x) > abs(v1.y)) v2 = float3(-v1.z, 0, v1.x) / sqrt(v1.x * v1.x + v1.z * v1.z);
    else v2 = float3(0, v1.z, -v1.y) / sqrt(v1.y * v1.y + v1.z * v1.z);
    v3 = normalize(cross(v1, v2));
}

inline float AbsMinDistAlong(float3 p, float3 dir, float3 boundMin, float3 boundMax)
{
    bool hasPositive = false;
    bool hasNegative = false;
    float a = dot(dir, float3(boundMin.x, boundMin.y, boundMin.z) - p);
    float b = dot(dir, float3(boundMin.x, boundMin.y, boundMax.z) - p);
    float c = dot(dir, float3(boundMin.x, boundMax.y, boundMin.z) - p);
    float d = dot(dir, float3(boundMin.x, boundMax.y, boundMax.z) - p);
    float e = dot(dir, float3(boundMax.x, boundMin.y, boundMin.z) - p);
    float f = dot(dir, float3(boundMax.x, boundMin.y, boundMax.z) - p);
    float g = dot(dir, float3(boundMax.x, boundMax.y, boundMin.z) - p);
    float h = dot(dir, float3(boundMax.x, boundMax.y, boundMax.z) - p);
    hasPositive = a > 0 || b > 0 || c > 0 || d > 0 || e > 0 || f > 0 || g > 0 || h > 0;
    hasNegative = a < 0 || b < 0 || c < 0 || d < 0 || e < 0 || f < 0 || g < 0 || h < 0;
    if (hasPositive && hasNegative) return 0.f;
    else return min(min(min(abs(a), abs(b)), min(abs(c), abs(d))), min(min(abs(e), abs(f)), min(abs(g), abs(h))));
}

inline float GeomTermBound(float3 p, float3 N, float3 boundMin, float3 boundMax)
{
    float nrm_max = MaxDistAlong(p, N, boundMin, boundMax);
    if (nrm_max <= 0) return 0.0f;
    float3 T, B;
    CoordinateSystem_(N, T, B);
    float y_amin = AbsMinDistAlong(p, T, boundMin, boundMax);
    float z_amin = AbsMinDistAlong(p, B, boundMin, boundMax);
    float hyp2 = y_amin * y_amin + z_amin * z_amin + nrm_max * nrm_max;
    return nrm_max * rsqrt(hyp2);
}

inline float GeomTermBoundApproximate(float3 p, float3 N, float3 boundMin, float3 boundMax)
{
    float nrm_max = MaxDistAlong(p, N, boundMin, boundMax);
    if (nrm_max <= 0) return 0.0f;
    float3 d = min(max(p, boundMin), boundMax) - p;
    float3 tng = d - dot(d, N) * N;
    float hyp2 = dot(tng, tng) + nrm_max * nrm_max;
    return nrm_max * rsqrt(hyp2);
}

inline float SquaredDistanceToClosestPoint(float3 p, float3 boundMin, float3 boundMax)
{
    float3 d = min(max(p, boundMin), boundMax) - p;
    return dot(d, d);
}

inline float SquaredDistanceToFarthestPoint(float3 p, float3 boundMin, float3 boundMax)
{
    float3 d = max(abs(boundMin - p), abs(boundMax - p));
    return dot(d, d);
}

inline float WidthSquared(float3 boundMin, float3 boundMax)
{
    float3 d = boundMax - boundMin;
    return d.x * d.x + d.y * d.y + d.z * d.z;
}

inline float normalizedWeights(float l2_0, float l2_1, float intensGeom0, float intensGeom1)
{
    float ww0 = l2_1 * intensGeom0;
    float ww1 = l2_0 * intensGeom1;
    return ww0 / (ww0 + ww1);
};

// https://graphics.stanford.edu/~seander/bithacks.html#IntegerLog
inline uint uintLog2(uint v) {
    uint r; // result of log2(v) will go here
    uint shift;
    r = (v > 0xFFFF) << 4; v >>= r;
    shift = (v > 0xFF) << 3; v >>= shift; r |= shift;
    shift = (v > 0xF) << 2; v >>= shift; r |= shift;
    shift = (v > 0x3) << 1; v >>= shift; r |= shift;
    r |= (v >> 1);
    return r;
}

#endif // _SRENDERER_ADDON_SLC_COMMON_HEADER_