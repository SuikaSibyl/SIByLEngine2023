#ifndef SRENDERER_RAYTRACER_PLUGIN_TRIMESH_PRIMITIVE_
#define SRENDERER_RAYTRACER_PLUGIN_TRIMESH_PRIMITIVE_

#include "../spt_interface.hlsli"
#include "../spt_implement.hlsli"
#include "../../include/common/math.hlsli"
#include "../../include/common/random.hlsli"
#include "../../include/common/sampling.hlsli"
#include "../../include/common/geometry.hlsli"
#include "../../include/scene_descriptor_set.hlsli"
#include "../../include/common/cpp_compatible.hlsli"

[shader("closesthit")]
void TrimeshClosestHit(
    inout_ref(PrimaryPayload) payload: SV_RayPayload,
    in_ref(Attributes) attrib: SV_IntersectionAttributes)
{
    const uint primitiveID = PrimitiveIndex();
    const uint geometryID = InstanceID() + GeometryIndex();
    const float3 barycentrics = float3(1 - attrib.uv.x - attrib.uv.y, attrib.uv.x, attrib.uv.y);
    
    Ray worldRay;
    worldRay.origin = WorldRayOrigin();
    worldRay.direction = WorldRayDirection();

    payload.hit = fetchTrimeshGeometryHit(
        int(geometryID),
        barycentrics,
        int(primitiveID),
        worldRay);
}

[shader("anyhit")]
void TrimeshAnyHit(inout_ref(PrimaryPayload) payload: SV_RayPayload,
                   in_ref(Attributes) attrib: SV_IntersectionAttributes)
{
    const uint primitiveID = PrimitiveIndex();
    const uint geometryID = InstanceID() + GeometryIndex();
    const float3 barycentrics = float3(1 - attrib.uv.x - attrib.uv.y, attrib.uv.x, attrib.uv.y);

    const GeometryInfo geometry = geometries[geometryID];
    const int3 index = int3(fetchTriangleIndices(geometry, int(primitiveID)));

    float2 vertexUVs[3];
    vertexUVs[0] = fetchVertexTexCoord(index[0] + int(geometry.vertexOffset));
    vertexUVs[1] = fetchVertexTexCoord(index[1] + int(geometry.vertexOffset));
    vertexUVs[2] = fetchVertexTexCoord(index[2] + int(geometry.vertexOffset));
    float2 uv = interpolate(vertexUVs, barycentrics);

    const MaterialInfo material = materials[geometry.materialID];
    const float alpha = textures[material.baseOrDiffuseTextureIndex].SampleLevel(uv, 0).w;

    const float rnd = GetNextRandom(payload.RND);
    if (rnd > alpha) {
        IgnoreHit();
    }
}

[shader("anyhit")]
void TrimeshShadowRayAnyHit(inout_ref(ShadowPayload) payload: SV_RayPayload,
                            in_ref(Attributes) attrib: SV_IntersectionAttributes)
{
    const uint primitiveID = PrimitiveIndex();
    const uint geometryID = InstanceID() + GeometryIndex();
    const float3 barycentrics = float3(1 - attrib.uv.x - attrib.uv.y, attrib.uv.x, attrib.uv.y);

    const GeometryInfo geometry = geometries[geometryID];
    const int3 index = int3(fetchTriangleIndices(geometry, int(primitiveID)));

    float2 vertexUVs[3];
    vertexUVs[0] = fetchVertexTexCoord(index[0] + int(geometry.vertexOffset));
    vertexUVs[1] = fetchVertexTexCoord(index[1] + int(geometry.vertexOffset));
    vertexUVs[2] = fetchVertexTexCoord(index[2] + int(geometry.vertexOffset));
    float2 uv = interpolate(vertexUVs, barycentrics);

    const MaterialInfo material = materials[geometry.materialID];
    const float alpha = textures[material.baseOrDiffuseTextureIndex].SampleLevel(uv, 0).w;

    const float rnd = GetNextRandom(payload.RND);
    if (rnd > alpha) {
        IgnoreHit();
    }
}

#endif // !SRENDERER_RAYTRACER_PLUGIN_TRIMESH_PRIMITIVE_