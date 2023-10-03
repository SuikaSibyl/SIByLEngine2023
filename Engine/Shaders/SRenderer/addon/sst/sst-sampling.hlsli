#ifndef _SRENDERER_ADDON_SST_SAMPLING_HEADER_
#define _SRENDERER_ADDON_SST_SAMPLING_HEADER_

#include "../../include/common/gaussian.hlsli"
#include "../../include/common/geometry.hlsli"
#include "../../include/common/random.hlsli"
#include "sst-interface.hlsli"

/** Sample VPL from SST
 */
int TraverseLightTree(
    in const int rootIndex,
    in_ref(float3) sd_p,
    in_ref(float3) sd_n,
    in_ref(float3) sd_v,
    inout_ref(RandomSamplerState) RNG,
    out_ref(float) p,
    in const StructuredBuffer<VPLData> vplData,
) {
    // Root node index
    int parentIdx = rootIndex;
    int currentDepth = 0;

    // Initialize the probability of picking the light
    p = 1.f;
    float r = GetNextRandom(RNG);

    // Get root node.
    VPLData vpl1 = vplData[parentIdx];
    
    while (true) {
        // Approximation good enough? Or did we reach a leaf node?
        if ((vpl1.getEarlyStop() > 0.f) || vpl1.numVPLSubTree <= 0)
            break;

        // Get child nodes
        const int child1Id = vpl1.idChild1;
        const int child2Id = vpl1.idChild2;
        vpl1 = vplData[child1Id];
        VPLData vpl2 = vplData[child2Id];
        // Intensity term: I
        const float I1 = vpl1.getIntensity();
        const float I2 = vpl2.getIntensity();
        // Material term: M
        const float M1 = maxNdotAABB(sd_p, sd_n, vpl1.getAABBMin(), vpl1.getAABBMax());
        const float M2 = maxNdotAABB(sd_p, sd_n, vpl2.getAABBMin(), vpl2.getAABBMax());
        // Geometric term: G
        const float G1 = 1.f; // Omni
        const float G2 = 1.f; // Omni
        // Attenuation term: A
        const float A1 = evalAttenuation(sd_p, vpl1);
        const float A2 = evalAttenuation(sd_p, vpl2);
        // Compute importance weights
        const float w1 = G1 * M1 * A1 * I1;
        const float w2 = G2 * M2 * A2 * I2;
        // Select child node
        if (w1 + w2 > 0) {
            const float p1 = w1 / (w1 + w2);
            if (r <= p1) {
                p = p * p1;
                r = r / p1;
                parentIdx = child1Id;
            } else {
                p = p * (1.f - p1);
                r = (r - p1) / (1.f - p1);
                parentIdx = child2Id;
                vpl1 = vpl2;
            }
            currentDepth++;
        } else {
            // Quit stochastic traversal => dead branch
            p = 0.f;
            return -1;
        }
    }
    return parentIdx;
}

/** Returns normal distributed point on plane bounded by an AABB. */
float3 normalPointOnPlane(
    in const float3 N,
    in const float3 O,
    in const float3 variance,
    in const float3 aabbMin,
    in const float3 aabbMax,
    inout_ref(RandomSamplerState) RNG
) {
    const float2 xy = BoxMuller(GetNextRandomFloat2(RNG), float2(0.f), sqrt(variance.xy));
    const float3x3 R = getRotationMatrixFromAToB(N, float3(0.f, 0.f, 1.f));
    const float3 P = O + R[0] * xy.x + R[1] * xy.y;
    return clamp(P, aabbMin, aabbMax);
}

float evalAttenuation(in_ref(float3) position, in VPLData vpl) {
    float gAttenuationEpsilon = 0.05f;
    return 1.f / max(distance_squared(position, vpl.getPosW()), gAttenuationEpsilon);
}

/** Returns true if point P is within the AABB, else false. */
bool isPointWithinAABB(in const float3 P, in const float3 aabbMin, in const float3 aabbMax) {
    return
        aabbMin.x < P.x && P.x < aabbMax.x &&
           aabbMin.y < P.y && P.y < aabbMax.y &&
           aabbMin.z < P.z && P.z < aabbMax.z;
}

/** Returns an upper bound for the cosine between the normal of point P and an AABB. */
float maxNdotAABB(
    in const float3 P,
    in const float3 N, 
    in const float3 aabbMin,
    in const float3 aabbMax
) {
    // Is point within the AABB? (with some epsilon due to numerical reasons...)
    const float3 epsilon = 0.001f;
    if (isPointWithinAABB(P, aabbMin - epsilon, aabbMax + epsilon))
        return 1.f;

    // Compute the upper cosine bound for the normal of point P to the AABB
    float3 zAxis = float3(0.f, 0.f, 1.f);
    bool flipZ = dot(N, zAxis) < 0.f;
    if (flipZ) zAxis.z = -zAxis.z;
    const float3x3 R = getRotationMatrixFromAToB(N, zAxis);

    // Transform AABB center
    const float3 hS = (aabbMax - aabbMin) * 0.5f;
    const float3 aaBBTCenter = mul(R, (aabbMin + hS) - P);

    const float3 Nx = R[0];
    const float3 Ny = R[1];
    const float3 Nz = R[2];

    float3 aabbTMin = float3(0.f);
    float3 aabbTMax = float3(0.f);

    // Compute each corner of the AABB and perfom min/max to get the extend
    const float3 v1 = Nx * hS.x + Ny * hS.y + Nz * hS.z;
    aabbTMin = min(aabbTMin, v1);
    aabbTMax = max(aabbTMax, v1);

    const float3 v2 = Nx * hS.x + Ny * hS.y + Nz * -hS.z;
    aabbTMin = min(aabbTMin, v2);
    aabbTMax = max(aabbTMax, v2);

    const float3 v3 = Nx * hS.x + Ny * -hS.y + Nz * hS.z;
    aabbTMin = min(aabbTMin, v3);
    aabbTMax = max(aabbTMax, v3);

    const float3 v4 = Nx * hS.x + Ny * -hS.y + Nz * -hS.z;
    aabbTMin = min(aabbTMin, v4);
    aabbTMax = max(aabbTMax, v4);

    const float3 v5 = Nx * -hS.x + Ny * hS.y + Nz * hS.z;
    aabbTMin = min(aabbTMin, v5);
    aabbTMax = max(aabbTMax, v5);

    const float3 v6 = Nx * -hS.x + Ny * hS.y + Nz * -hS.z;
    aabbTMin = min(aabbTMin, v6);
    aabbTMax = max(aabbTMax, v6);

    const float3 v7 = Nx * -hS.x + Ny * -hS.y + Nz * hS.z;
    aabbTMin = min(aabbTMin, v7);
    aabbTMax = max(aabbTMax, v7);

    const float3 v8 = Nx * -hS.x + Ny * -hS.y + Nz * -hS.z;
    aabbTMin = min(aabbTMin, v8);
    aabbTMax = max(aabbTMax, v8);

    aabbTMin += aaBBTCenter;
    aabbTMax += aaBBTCenter;

    const float zMax = flipZ ? -min(aabbTMin.z, aabbTMax.z) : max(aabbTMin.z, aabbTMax.z);
    if (zMax < 0.00001f) // Early out, cos would be <= 0
        return 0.f;

    float xMin, yMin;
    if ((aabbTMin.x < 0.f && aabbTMax.x > 0.f) || (aabbTMax.x < 0.f && aabbTMin.x > 0.f))
        xMin = 0.f;
    else
        xMin = min(abs(aabbTMin.x), abs(aabbTMax.x));

    if ((aabbTMin.y < 0.f && aabbTMax.y > 0.f) || (aabbTMax.y < 0.f && aabbTMin.y > 0.f))
        yMin = 0.f;
    else
        yMin = min(abs(aabbTMin.y), abs(aabbTMax.y));

    return zMax / length(float3(xMin, yMin, zMax)); // Div by zero not possible since zMax > 0
}

#endif // !_SRENDERER_ADDON_SST_SAMPLING_HEADER_