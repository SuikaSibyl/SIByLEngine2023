#ifndef _SRENDERER_ADDON_LTC_INTERFACE_HEADER_
#define _SRENDERER_ADDON_LTC_INTERFACE_HEADER_

#include "../../include/common/cpp_compatible.hlsli"

/***********************************************************************
 * Linear Transformed Cosine Utils
 ***********************************************************************
 * This file contains some utility functions for LTC.
 * The implementation is based on some tutorials:
 * @ref: https://learnopengl.com/Guest-Articles/2022/Area-Lights
 * @ref: https://advances.realtimerendering.com/s2016/s2016_ltc_rnd.pdf
 ***********************************************************************/

/**
 * Integrate the cosine weighted function over an spherical edge.
 * But this is a vector form without project to the plane (dot with the normal)
 * Used for proxy sphere clipping.
 * @param v1: the first end points of the edge.
 * @param v2: the second end points of the edge.
 */
float3 IntegrateEdgeVec(in_ref(float3) v1, in_ref(float3) v2) {
    const float x = dot(v1, v2);
    const float y = abs(x);
    const float a = 0.8543985 + (0.4965155 + 0.0145206 * y) * y;
    const float b = 3.4175940 + (4.1616724 + y) * y;
    const float v = a / b;
    const float theta_sintheta = (x > 0.0) ? v : 0.5 * rsqrt(max(1.0 - x * x, 1e-7)) - v;
    return cross(v1, v2) * theta_sintheta;
}

/**
 * Integrate the cosine weighted function over an spherical edge.
 * @param v1: the first end points of the edge.
 * @param v2: the second end points of the edge.
 * @param N: the normal of the shading point.
 */
float IntegrateEdge(in_ref(float3) v1, in_ref(float3) v2, in_ref(float3) N) {
    const float3 vector_form = IntegrateEdgeVec(v1, v2);
    return dot(vector_form, N);
}

/**
 * Integrate the cosine weighted function over an spherical edge.
 * @param position: the position of the shading point in world space.
 * @param normal: the normal of the shading point in world space.
 * @param view: the view direction of the shading point in world space.
*/

float3 LTCEvaluate_Quad(
    in_ref(Sampler2D<float>) lut,
    in_ref(float3) position, in_ref(float3) normal, in_ref(float3) view,
    in_ref(float3x3) Minv, in_ref(float3) points[4], bool twoSided
) {
    // construct orthonormal basis around N
    float3 T1;
    float3 T2;
    T1 = normalize(view - normal * dot(view, normal));
    T2 = cross(normal, T1);
    // rotate area light in (T1, T2, N) basis
    Minv = Minv * transpose(float3x3(T1, T2, normal));
    // polygon (allocate 4 vertices for clipping)
    float3 L[4];
    // transform polygon from LTC back to origin Do (cosine weighted)
    L[0] = mul(points[0] - position, Minv);
    L[1] = mul(points[1] - position, Minv);
    L[2] = mul(points[2] - position, Minv);
    L[3] = mul(points[3] - position, Minv);
    // use tabulated horizon-clipped sphere
    // check if the shading point is behind the light
    float3 dir = points[0] - position; // LTC space
    float3 lightNormal = cross(points[1] - points[0], points[3] - points[0]);
    bool behind = (dot(dir, lightNormal) < 0.0);
    // cos weighted space
    L[0] = normalize(L[0]);
    L[1] = normalize(L[1]);
    L[2] = normalize(L[2]);
    L[3] = normalize(L[3]);
    // integrate
    float3 vsum = float3(0.0);
    vsum += IntegrateEdgeVec(L[0], L[1]);
    vsum += IntegrateEdgeVec(L[1], L[2]);
    vsum += IntegrateEdgeVec(L[2], L[3]);
    vsum += IntegrateEdgeVec(L[3], L[0]);
    // form factor of the polygon in direction vsum
    float len = length(vsum);
    float z = vsum.z / len;
    if (behind)
        z = -z;
    float2 uv = float2(z * 0.5f + 0.5f, len); // range [0, 1]
    static const float LUT_SIZE = 64.0; // ltc_texture size
    static const float LUT_SCALE = (LUT_SIZE - 1.0) / LUT_SIZE;
    static const float LUT_BIAS = 0.5 / LUT_SIZE;
    uv = uv * LUT_SCALE + LUT_BIAS;
    // Fetch the form factor for horizon clipping
    // float scale = lut.Sample(uv);
    float scale = 1;
    float sum = len * scale;
    if (!behind && !twoSided)
        sum = 0.0;
    // Outgoing radiance (solid angle) for the entire polygon
    float3 Lo_i = float3(sum, sum, sum);
    return Lo_i;
}

/***********************************************************************
 * In this part, we define some utility functions for depth conservatie
 * rasterization. By using hardware conservative rasterization, we can
 * get axises along the image plane become conservative.
 * However, in the depth / dominant axis direction, one pixel invocation
 * may need to fill in multiple voxels.
 * According to "The Basics of GPU Voxelization", it could at most
 * be 3 voxels (shading point, back voxel, rear voxel).
 *
 * To check whether to fill extra voxels, we first do a AABB test.
 * a.k.a. "AABBConditionTest";
 * If the triangle's AABB passes, we do a more accurate test.
 * a.k.a. "EdgeVoxelConditionTest"; (separate axis theorems)
 * If the triangle passes both tests, we fill in the voxels.
 ************************************************************************/

// float IntegrateEdge(in_ref(float3) v1, in_ref(float3) v2) {
//     float cosTheta = dot(v1, v2);
//     float theta = acos(cosTheta);
//     float res = cross(v1, v2).z * ((theta > 0.001) ? theta / sin(theta) : 1.0);
//     return res;
// }

// void ClipQuadToHorizon(inout_ref(float3) L[5], out_ref(int) n) {
//     // detect clipping config
//     int config = 0;
//     if (L[0].z > 0.0) config += 1;
//     if (L[1].z > 0.0) config += 2;
//     if (L[2].z > 0.0) config += 4;
//     if (L[3].z > 0.0) config += 8;
//     // clip
//     n = 0;
//     if (config == 0) {
//         // clip all
//     }
//     else if (config == 1) { // V1 clip V2 V3 V4
//         n = 3;
//         L[1] = -L[1].z * L[0] + L[0].z * L[1];
//         L[2] = -L[3].z * L[0] + L[0].z * L[3];
//     }
//     else if (config == 2) // V2 clip V1 V3 V4
//     {
//         n = 3;
//         L[0] = -L[0].z * L[1] + L[1].z * L[0];
//         L[2] = -L[2].z * L[1] + L[1].z * L[2];
//     }
//     else if (config == 3) // V1 V2 clip V3 V4
//     {
//         n = 4;
//         L[2] = -L[2].z * L[1] + L[1].z * L[2];
//         L[3] = -L[3].z * L[0] + L[0].z * L[3];
//     }
//     else if (config == 4) // V3 clip V1 V2 V4
//     {
//         n = 3;
//         L[0] = -L[3].z * L[2] + L[2].z * L[3];
//         L[1] = -L[1].z * L[2] + L[2].z * L[1];
//     }
//     else if (config == 5) // V1 V3 clip V2 V4) impossible
//     {
//         n = 0;
//     }
//     else if (config == 6) // V2 V3 clip V1 V4
//     {
//         n = 4;
//         L[0] = -L[0].z * L[1] + L[1].z * L[0];
//         L[3] = -L[3].z * L[2] + L[2].z * L[3];
//     }
//     else if (config == 7) // V1 V2 V3 clip V4
//     {
//         n = 5;
//         L[4] = -L[3].z * L[0] + L[0].z * L[3];
//         L[3] = -L[3].z * L[2] + L[2].z * L[3];
//     }
//     else if (config == 8) // V4 clip V1 V2 V3
//     {
//         n = 3;
//         L[0] = -L[0].z * L[3] + L[3].z * L[0];
//         L[1] = -L[2].z * L[3] + L[3].z * L[2];
//         L[2] = L[3];
//     }
//     else if (config == 9) // V1 V4 clip V2 V3
//     {
//         n = 4;
//         L[1] = -L[1].z * L[0] + L[0].z * L[1];
//         L[2] = -L[2].z * L[3] + L[3].z * L[2];
//     }
//     else if (config == 10) // V2 V4 clip V1 V3) impossible
//     {
//         n = 0;
//     }
//     else if (config == 11) // V1 V2 V4 clip V3
//     {
//         n = 5;
//         L[4] = L[3];
//         L[3] = -L[2].z * L[3] + L[3].z * L[2];
//         L[2] = -L[2].z * L[1] + L[1].z * L[2];
//     }
//     else if (config == 12) // V3 V4 clip V1 V2
//     {
//         n = 4;
//         L[1] = -L[1].z * L[2] + L[2].z * L[1];
//         L[0] = -L[0].z * L[3] + L[3].z * L[0];
//     }
//     else if (config == 13) // V1 V3 V4 clip V2
//     {
//         n = 5;
//         L[4] = L[3];
//         L[3] = L[2];
//         L[2] = -L[1].z * L[2] + L[2].z * L[1];
//         L[1] = -L[1].z * L[0] + L[0].z * L[1];
//     }
//     else if (config == 14) // V2 V3 V4 clip V1
//     {
//         n = 5;
//         L[4] = -L[0].z * L[3] + L[3].z * L[0];
//         L[0] = -L[0].z * L[1] + L[1].z * L[0];
//     }
//     else if (config == 15) // V1 V2 V3 V4
//     {
//         n = 4;
//     }

//     if (n == 3)
//         L[3] = L[0];
//     if (n == 4)
//         L[4] = L[0];
// }

// vec3 LTC_Evaluate(
//     vec3 N, vec3 V, vec3 P, mat3 Minv, vec3 points[4], bool twoSided)
// {
//     // construct orthonormal basis around N
//     vec3 T1, T2;
//     T1 = normalize(V - N * dot(V, N));
//     T2 = cross(N, T1);

//     // rotate area light in (T1, T2, N) basis
//     Minv = mul(Minv, transpose(mat3(T1, T2, N)));

//     // polygon (allocate 5 vertices for clipping)
//     vec3 L[5];
//     L[0] = mul(Minv, points[0] - P);
//     L[1] = mul(Minv, points[1] - P);
//     L[2] = mul(Minv, points[2] - P);
//     L[3] = mul(Minv, points[3] - P);

//     int n;
//     ClipQuadToHorizon(L, n);

//     if (n == 0)
//         return vec3(0, 0, 0);

//     // project onto sphere
//     L[0] = normalize(L[0]);
//     L[1] = normalize(L[1]);
//     L[2] = normalize(L[2]);
//     L[3] = normalize(L[3]);
//     L[4] = normalize(L[4]);

//     // integrate
//     float sum = 0.0;

//     sum += IntegrateEdge(L[0], L[1]);
//     sum += IntegrateEdge(L[1], L[2]);
//     sum += IntegrateEdge(L[2], L[3]);
//     if (n >= 4)
//         sum += IntegrateEdge(L[3], L[4]);
//     if (n == 5)
//         sum += IntegrateEdge(L[4], L[0]);

//     sum = twoSided ? abs(sum) : max(0.0, sum);

//     vec3 Lo_i = vec3(sum, sum, sum);

//     return Lo_i;
// }

#endif // !_SRENDERER_ADDON_LTC_INTERFACE_HEADER_