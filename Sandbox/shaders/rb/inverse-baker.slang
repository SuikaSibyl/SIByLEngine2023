#include "../srenderer/rayquery.hlsli"
#include "../srenderer/materials/lambertian.hlsli"
#include "../srenderer/materials/roughplastic.hlsli"

RWStructuredBuffer<float4> u_inverse;

void SetInverseBuffer(int2 pixel, RawPayload payload) {
    const float4 val = float4(
        asfloat(payload.primitiveID),
        asfloat(payload.geometryID),
        payload.barycentric
    );
    u_inverse[(pixel.x + pixel.y * 512)] = val;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void ComputeMain(int3 dtid: SV_DispatchThreadID) {
    // The resolution of the image, which is the same as the launch size:
    const int2 resolution = int2(512, 512);
    const int2 pixel = int2(dtid.xy);
    
    // If the pixel is outside of the image, don't do anything:
    if (any(pixel >= resolution)) return;
    
    for (int i = 0; i < 4; ++i) {
        const Ray primaryRay = generateRay(
            pixel,
            resolution,
            false,
            GPUScene_camera[i]
        );
        RawPayload payload = RawRayQuery(primaryRay);
        if (payload.hasHit) {
            const float2 bary = payload.barycentric;
            const float3 barycentrics = float3(1 - bary.x - bary.y, bary.x, bary.y);
            GeometryHit hit = fetchTrimeshGeometryHit(payload.geometryID, barycentrics, payload.primitiveID);
            int2 coord = int2(hit.texcoord * float2(512, 512));
            SetInverseBuffer(coord, payload);
        }
    }
}