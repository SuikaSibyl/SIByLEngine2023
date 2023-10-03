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

/**
 * Sample a point on a triangle mesh light.
 * @ref: https://pbr-book.org/3ed-2018/Monte_Carlo_Integration/2D_Sampling_with_Multidimensional_Transformations#SamplingaTriangle
 */
void SampleTrimesh(inout_ref(ShapeSampleQuery) cShapeSampleQuery) {
    // sample a triangle in the geometry
    const uint primitiveID = cShapeSampleQuery.primitive_id;
    const uint geometryID = cShapeSampleQuery.geometry_id;
    const GeometryInfo geometry = geometries[geometryID];
    // Get the indices of the vertices of the triangle
    const int3 index = int3(fetchTriangleIndices(geometry, int(primitiveID)));
    // Get the vertices / normals of the triangle
    const float3 v0 = fetchVertexPosition(index[0] + int(geometry.vertexOffset));
    const float3 v1 = fetchVertexPosition(index[1] + int(geometry.vertexOffset));
    const float3 v2 = fetchVertexPosition(index[2] + int(geometry.vertexOffset));
    const float3 n0 = fetchVertexNormal(index[0] + int(geometry.vertexOffset));
    const float3 n1 = fetchVertexNormal(index[1] + int(geometry.vertexOffset));
    const float3 n2 = fetchVertexNormal(index[2] + int(geometry.vertexOffset));
    // Get the transforms
    const float4x4 o2w = ObjectToWorld(geometry);
    const float4x4 o2wn = ObjectToWorldNormal(geometry);
    // Sample on the triangle
    const float3 e1 = v1 - v0;
    const float3 e2 = v2 - v0;
    const float a = sqrt(clamp(cShapeSampleQuery.uv[0], 0., 1.));
    const float b1 = 1 - a;
    const float b2 = a * cShapeSampleQuery.uv[1];
    const float3 v0_ws = mul(float4(v0, 1), o2w).xyz;
    const float3 v1_ws = mul(float4(v1, 1), o2w).xyz;
    const float3 v2_ws = mul(float4(v2, 1), o2w).xyz;
    const float3 e1_ws = v1_ws - v0_ws;
    const float3 e2_ws = v2_ws - v0_ws;
    float3 geometric_normal = normalize(cross(e1, e2));
    const float area = length(cross(e1_ws, e2_ws)) / 2;
    // Flip the geometric normal to the same side as the shading normal
    const float3 shading_normal = normalize((1 - b1 - b2) * n0 + b1 * n1 + b2 * n2);
    if (dot(geometric_normal, shading_normal) < 0) {
        geometric_normal = -geometric_normal;
    }
    // Output the sample query result
    cShapeSampleQuery.position = mul(float4(v0 + (e1 * b1) + (e2 * b2), 1), o2w).xyz;
    cShapeSampleQuery.normal = normalize(mul(float4(geometric_normal, 0.0), o2wn).xyz);

    cShapeSampleQuery.pdf = 1.f / area;
    return;
}

void PdfTrimesh(inout_ref(ShapeSamplePdfQuery) cShapeSamplePdfQuery) {
    const uint geometryID = cShapeSamplePdfQuery.geometry_id;
    cShapeSamplePdfQuery.pdf = 1.f / geometries[geometryID].surfaceArea;
    return;
}

[shader("callable")]
void SampleTrimeshCallable(inout_ref(ShapeSampleQuery) cShapeSampleQuery) {
    SampleTrimesh(cShapeSampleQuery);
}

[shader("callable")]
void PdfTrimeshCallable(inout_ref(ShapeSamplePdfQuery) cShapeSamplePdfQuery) {
    PdfTrimesh(cShapeSamplePdfQuery);
}

#endif // !SRENDERER_RAYTRACER_PLUGIN_TRIMESH_PRIMITIVE_