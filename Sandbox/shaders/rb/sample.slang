#include "../srenderer/rayquery.hlsli"

RaytracingAccelerationStructure TexCoordScene_tlas;
RWStructuredBuffer<float> sampleBuffer;
RWStructuredBuffer<float> debugBuffer;

void SetSampleBuffer(float3 sample, int2 pixel) {
    sampleBuffer[(pixel.y * 512 + pixel.x) * 3 + 0] = sample.x;
    sampleBuffer[(pixel.y * 512 + pixel.x) * 3 + 1] = sample.y;
    sampleBuffer[(pixel.y * 512 + pixel.x) * 3 + 2] = sample.z;
}

struct TexcoordSamplePayload {
    float3 position;
    float pdf;
};

TexcoordSamplePayload TexcoordRayQuery(in const Ray ray) {
    // Initialize the payload
    TexcoordSamplePayload payload;
    payload.pdf = 0.f;
    // Apply the ray query
    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(TexCoordScene_tlas, 0, 0xff, ToRayDesc(ray));
    while (q.Proceed()) {
        switch (q.CandidateType()) {
        case CANDIDATE_NON_OPAQUE_TRIANGLE: {
            if (true) {
                q.CommitNonOpaqueTriangleHit();
            }
            if (false) {
                q.Abort();
            }
            break;
        }
        }
    }
    // Process the result
    switch (q.CommittedStatus()) {
    case COMMITTED_TRIANGLE_HIT: {
        // Fetch the hit information
        const int primitiveID = q.CommittedPrimitiveIndex();
        const int geometryID = q.CommittedInstanceIndex() + q.CommittedGeometryIndex();
        const float2 bary = q.CommittedTriangleBarycentrics();
        const float3 barycentrics = float3(1 - bary.x - bary.y, bary.x, bary.y);
        const GeometryData geometry = GPUScene_geometry[geometryID];
        const int3 index = int3(fetchTriangleIndices(geometry, primitiveID));

        float3 vertexPositions[3];
        vertexPositions[0] = fetchVertexPosition(index[0] + int(geometry.vertexOffset));
        vertexPositions[1] = fetchVertexPosition(index[1] + int(geometry.vertexOffset));
        vertexPositions[2] = fetchVertexPosition(index[2] + int(geometry.vertexOffset));

        const float4x4 o2w = ObjectToWorld(geometry);
        const float3 positionOS = interpolate(vertexPositions, barycentrics);
        const float3 positionWS = mul(float4(positionOS, 1.0), o2w).xyz;
        payload.position = positionWS;

        float2 vertexUVs[3];
        vertexUVs[0] = fetchVertexTexCoord(index[0] + int(geometry.vertexOffset));
        vertexUVs[1] = fetchVertexTexCoord(index[1] + int(geometry.vertexOffset));
        vertexUVs[2] = fetchVertexTexCoord(index[2] + int(geometry.vertexOffset));

        // transform vertexPositions to world space
        vertexPositions[0] = mul(float4(vertexPositions[0], 1.0), o2w).xyz;
        vertexPositions[1] = mul(float4(vertexPositions[1], 1.0), o2w).xyz;
        vertexPositions[2] = mul(float4(vertexPositions[2], 1.0), o2w).xyz;

        // compute the 2d area of vertexUVs triangle
        const float texcoord_area = 0.5 * abs((vertexUVs[0].x * (vertexUVs[1].y - vertexUVs[2].y) 
            + vertexUVs[1].x * (vertexUVs[2].y - vertexUVs[0].y) + vertexUVs[2].x * (vertexUVs[0].y - vertexUVs[1].y)));
        // compute the 3d surface area of positionWS triangle
        const float3 edge1 = vertexPositions[1] - vertexPositions[0];
        const float3 edge2 = vertexPositions[2] - vertexPositions[0];
        const float3 normal = cross(edge1, edge2);
        const float area = 0.5 * length(normal);
        payload.pdf = texcoord_area / area;
        
        break;
    }
    case COMMITTED_PROCEDURAL_PRIMITIVE_HIT: {
        break;
    }
    case COMMITTED_NOTHING: {
        break;
    }
    }

    return payload;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void ComputeMain(int3 dtid: SV_DispatchThreadID) {
    const int2 gid = dtid.xy;

    Ray texSampleRay;
    texSampleRay.origin = float3(float2(dtid.x, dtid.y) / 512, -1);
    texSampleRay.direction = float3(0, 0, 1);
    texSampleRay.tMin = 0;
    texSampleRay.tMax = 1000;
    TexcoordSamplePayload samplePayload = TexcoordRayQuery(texSampleRay);
    if (samplePayload.pdf > 0) {
        SetSampleBuffer(samplePayload.position, dtid.xy);
        debugBuffer[gid.y * 512 + gid.x] = samplePayload.pdf;
    }
    else {
        SetSampleBuffer(float3(dtid.x, 0, dtid.y) / float3(512), dtid.xy);
        debugBuffer[gid.y * 512 + gid.x] = samplePayload.pdf;
    }
    // SetSampleBuffer(float3(dtid.x, dtid.y, 0) / float3(512), dtid.xy);
}