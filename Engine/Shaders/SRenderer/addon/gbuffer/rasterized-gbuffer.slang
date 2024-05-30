#include "../../include/common/geometry.hlsli"
#include "../../include/common/math.hlsli"
#include "../../include/common/packing.hlsli"
#include "../../include/common/random.hlsli"
#include "../../include/scene_descriptor_set.hlsli"
#include "../../include/common/raycast.hlsli"
#include "gbuffer_common.hlsli"

[[vk::push_constant]]
cbuffer PushConstants {
    struct PushConstants {
        uint geometry_idx;
    } pConst;
};

struct AssembledVertex { int vertexId : SV_VertexId; };
struct CoarseVertex {
    uint geoID;
};
struct VertexStageOutput {
    CoarseVertex coarseVertex : CoarseVertex;
    float4 sv_position : SV_Position;
};

[shader("vertex")]
VertexStageOutput vertexMain(
    AssembledVertex assembledVertex)
{
    const GeometryInfo geometry = geometries[pConst.geometry_idx];
    const float4x4 o2w = ObjectToWorld(geometry);

    const float3 positionOS = fetchVertexPosition(assembledVertex.vertexId);
    const float3 positionWS = mul(float4(positionOS, 1.0), o2w).xyz;
    const float4 positionCS = mul(float4(positionWS, 1.0f), globalUniform.cameraData.viewProjMat);

    VertexStageOutput output;
    output.coarseVertex.geoID = pConst.geometry_idx;
    output.sv_position = positionCS;
    return output;
}

[shader("fragment")]
void fragmentMain(
    in CoarseVertex coarseVertex: CoarseVertex,
    nointerpolation in int primitiveID: SV_PrimitiveID,
    in bool isFrontFace: SV_IsFrontFace,
    in float3 bary: SV_Barycentrics,
    in float4 svPosition: SV_Position,
    // output
    out float4 o_position: SV_Target0,
    out uint o_diffuseAlbedo: SV_Target1,
    out uint o_specularRough: SV_Target2,
    out uint o_normal: SV_Target3,
    out uint o_geoNormal: SV_Target4,
    out float4 o_emissive: SV_Target5,
    out float4 o_motion: SV_Target6,
    out float3 o_material: SV_Target7)
{
    const GeometryInfo geometry = geometries[coarseVertex.geoID];
    const GeometryInfo prevGeometry = prevGeometries[coarseVertex.geoID];

    const int3 index = int3(fetchTriangleIndices(geometry, primitiveID));
    const MaterialInfo material = materials[geometry.materialID];

    float3 vertexPositions[3];
    vertexPositions[0] = fetchVertexPosition(index[0] + int(geometry.vertexOffset));
    vertexPositions[1] = fetchVertexPosition(index[1] + int(geometry.vertexOffset));
    vertexPositions[2] = fetchVertexPosition(index[2] + int(geometry.vertexOffset));

    const float4x4 o2w = ObjectToWorld(geometry);
    const float3 positionOS = interpolate(vertexPositions, bary);
    const float3 positionWS = mul(float4(positionOS, 1.0), o2w).xyz;

    float3 viewDirection = positionWS - globalUniform.cameraData.posW;
    const float viewDistance = length(viewDirection);
    viewDirection /= viewDistance;

    float2 vertexUVs[3];
    vertexUVs[0] = fetchVertexTexCoord(index[0] + int(geometry.vertexOffset));
    vertexUVs[1] = fetchVertexTexCoord(index[1] + int(geometry.vertexOffset));
    vertexUVs[2] = fetchVertexTexCoord(index[2] + int(geometry.vertexOffset));
    float2 uv = interpolate(vertexUVs, bary);

    float3 objectSpaceFlatNormal = normalize(cross(
        vertexPositions[1] - vertexPositions[0],
        vertexPositions[2] - vertexPositions[0]));

    float4x4 o2wn = ObjectToWorldNormal(geometry);
    float3 flatNormal = normalize(mul(float4(objectSpaceFlatNormal, 0.0), o2wn).xyz);
    bool faceforward = dot(flatNormal, viewDirection) < 0;
    flatNormal = faceforward ? flatNormal : -flatNormal;
    
    float3 normals[3];
    normals[0] = fetchVertexNormal(index[0] + int(geometry.vertexOffset));
    normals[1] = fetchVertexNormal(index[1] + int(geometry.vertexOffset));
    normals[2] = fetchVertexNormal(index[2] + int(geometry.vertexOffset));
    float3 geometryNormalOS = interpolate(normals, bary);
    float3 geometryNormalWS = normalize(mul(float4(geometryNormalOS, 0.0), o2wn).xyz);
    geometryNormalWS = faceforward ? geometryNormalWS : -geometryNormalWS;
    
    float3 tangents[3];
    tangents[0] = fetchVertexTangent(index[0] + int(geometry.vertexOffset));
    tangents[1] = fetchVertexTangent(index[1] + int(geometry.vertexOffset));
    tangents[2] = fetchVertexTangent(index[2] + int(geometry.vertexOffset));
    float3 tangentOS = interpolate(tangents, bary);
    float4 tangentWS = float4(normalize(mul(float4(tangentOS, 0), o2w).xyz), GetOddNegativeScaling(geometry));

    float3 shadingNormal = geometryNormalWS;
    float3 bitangent = cross(geometryNormalWS, tangentWS.xyz) * tangentWS.w;
    float3x3 TBN = float3x3(tangentWS.xyz, bitangent, geometryNormalWS);
    float3 normalMap = float3(0.5, 0.5, 1.0);
    if (material.normalTextureIndex != -1) {
        normalMap = textures[material.normalTextureIndex].Sample(uv).rgb;
        shadingNormal = normalize(mul((normalMap * 2 - 1), TBN));
    }

    // Grab material infomation
    float3 diffuseAlbedo = textures[material.baseOrDiffuseTextureIndex].Sample(uv).rgb;

    float viewDepth = 0;
    float3 motion = getMotionVector(globalUniform.cameraData,
                                    prevGlobalUniform.cameraData,
                                    geometry,
                                    prevGeometry,
                                    positionOS,
                                    positionOS, // no vertex change
                                    viewDepth);

    float3 specularF0 = float3(0);
    float roughness = 1;

    float3 emissiveColor = float3(0);

    const float2 inUV = svPosition.xy - float2(0.5);
    const int2 viewportSize = getViewportSize(globalUniform.cameraData);
    const Ray ray_0 = generateRay(inUV, viewportSize, false, globalUniform.cameraData);
    const Ray ray_x = generateRay(inUV + float2(1, 0), viewportSize, false, globalUniform.cameraData);
    const Ray ray_y = generateRay(inUV + float2(0, 1), viewportSize, false, globalUniform.cameraData);
    const float half_cone_angle = sqrt(1.0 - square(min(dot(ray_0.direction, ray_x.direction), dot(ray_0.direction, ray_y.direction))));
    const float fwidth_depth = computeFWidthDepth(viewDistance * half_cone_angle, viewDirection, flatNormal);

    o_position = float4(positionWS, viewDepth);
    o_diffuseAlbedo = Pack_R11G11B10_UFLOAT(diffuseAlbedo);
    o_specularRough = Pack_R8G8B8A8_Gamma_UFLOAT(float4(specularF0, roughness));
    o_normal = UnitVectorToUnorm32Octahedron(shadingNormal);
    o_geoNormal = UnitVectorToUnorm32Octahedron(flatNormal);
    o_emissive = float4(emissiveColor, viewDistance); // viewDistance is here to enable glass ray tracing on all pixels
    o_motion = float4(motion, fwidth_depth);
    o_material = float3(material.bsdfID, uv);
}