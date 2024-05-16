#ifndef _SRENDERER_RAYQUERY_HEADER_
#define _SRENDERER_RAYQUERY_HEADER_

#include "scene-binding.hlsli"

PrimaryPayload PrimaryRayQuery(in const Ray ray) {
    PrimaryPayload payload;
    SetHit(payload.hit, false);
    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(GPUScene_tlas, 0, 0xff, ToRayDesc(ray));
    while (q.Proceed()) {
        switch (q.CandidateType()) {
        case CANDIDATE_NON_OPAQUE_TRIANGLE: {
            if(true) {
                q.CommitNonOpaqueTriangleHit();
            }
            if(false) {
                q.Abort();
            }
            break;
        }
        }
    }
    switch (q.CommittedStatus()) {
    case COMMITTED_TRIANGLE_HIT: {
        // Do hit shading
        const int primitiveID = q.CommittedPrimitiveIndex();
        const int geometryID = q.CommittedInstanceIndex() + q.CommittedGeometryIndex();
        const float2 bary = q.CommittedTriangleBarycentrics();
        const float3 barycentrics = float3(1 - bary.x - bary.y, bary.x, bary.y);
        payload.hit = fetchTrimeshGeometryHit(geometryID, barycentrics, primitiveID, ray);
        SetHit(payload.hit, true);
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

bool VisibilityRayQuery(in const Ray ray) {
    PrimaryPayload payload;
    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(GPUScene_tlas, 0, 0xff, ToRayDesc(ray));
    while (q.Proceed()) {
        switch (q.CandidateType()) {
        case CANDIDATE_NON_OPAQUE_TRIANGLE: {
            q.CommitNonOpaqueTriangleHit();
            break;
        }
        }
    }
    switch (q.CommittedStatus()) {
    case COMMITTED_TRIANGLE_HIT: return false;
    case COMMITTED_PROCEDURAL_PRIMITIVE_HIT: return false;
    case COMMITTED_NOTHING: break;
    }

    return true;
}

RawPayload RawRayQuery(in const Ray ray) {
    RawPayload payload; payload.hasHit = false;
    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(GPUScene_tlas, 0, 0xff, ToRayDesc(ray));
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
    switch (q.CommittedStatus()) {
    case COMMITTED_TRIANGLE_HIT: {
        payload.primitiveID = q.CommittedPrimitiveIndex();
        payload.geometryID = q.CommittedInstanceIndex() + q.CommittedGeometryIndex();
        payload.barycentric = q.CommittedTriangleBarycentrics();
        payload.hasHit = true;
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

#endif // _SRENDERER_RAYQUERY_HEADER_