#include "../../include/common/geometry.hlsli"
#include "../../include/scene_descriptor_set.hlsli"
#include "include/vxgi_interface.hlsli"

[[vk::push_constant]]
cbuffer PushConstants {
    struct PushConstants {
        uint geometry_idx;
    } pConst;
};

cbuffer VoxelizerUniform {
    VoxerlizerData gVD;
}

struct AssembledVertex { int vertexId : SV_VertexId; };
struct CoarseVertex {
    float3 posVS;
    // uint geoID;
};
struct VertexStageOutput {
    CoarseVertex coarseVertex : CoarseVertex;
    float4 sv_position : SV_Position;   // [-1,+1]^3
};

[shader("vertex")]
VertexStageOutput vertexMain(
    AssembledVertex assembledVertex)
{
    const GeometryInfo geometry = geometries[pConst.geometry_idx];
    const float4x4 o2w = ObjectToWorld(geometry);

    const float3 positionOS = fetchVertexPosition(assembledVertex.vertexId);
    const float3 positionWS = mul(float4(positionOS, 1.0), o2w).xyz;

    const float3 extent = gVD.aabbMax - gVD.aabbMin;
    const float extentMax = max(extent.x, max(extent.y, extent.z)) * 0.5 + 0.01;
    const float3 center = (gVD.aabbMax + gVD.aabbMin) * 0.5;

    VertexStageOutput output;
    // output.coarseVertex.geoID = pConst.geometry_idx;
    output.sv_position = float4((positionWS.xyz - center) / extentMax, 1.0);
    output.coarseVertex.posVS = output.sv_position.xyz;
    return output;
}

// Get the dominant axis of a triangle, that should be projected to.
int getDominantAxis(in_ref(float3) pos0, in_ref(float3) pos1, in_ref(float3) pos2) {
    const float3 normal = abs(cross(pos1 - pos0, pos2 - pos0));
    return (normal.x > normal.y && normal.x > normal.z) ? 0 : 
			(normal.y > normal.z) ? 1 : 2;
}

float2 project(in_ref(float3) vertex, uint axis) {
    return axis == 0 ? vertex.yz : (axis == 1 ? vertex.xz : vertex.xy);
}

[shader("geometry")]
[instance(1)]
[maxvertexcount(3)]
void geometryMain(
    in VertexStageOutput input[3],
    uint InstanceID: SV_GSInstanceID,
    inout TriangleStream<VertexStageOutput> output)
{
    const int axis = getDominantAxis(input[0].sv_position.xyz,
                                     input[1].sv_position.xyz,
                                     input[2].sv_position.xyz);
    
    for (int i = 0; i < 3; ++i) {
        VertexStageOutput vertex;
        // gl_ViewportIndex = axis;
        vertex.sv_position = float4(project(input[i].sv_position.xyz, axis), 1.0, 1.0);
        vertex.coarseVertex.posVS = (input[i].coarseVertex.posVS.xyz + 1.0) * 0.5;
        // vertex.coarseVertex.geoID = input[i].coarseVertex.geoID;
        output.Append(vertex);
    }
    output.RestartStrip();
}

RWTexture3D<float> uOpacityVox;

[shader("fragment")]
void fragmentMain(
    in CoarseVertex coarseVertex: CoarseVertex,
    nointerpolation in int primitiveID: SV_PrimitiveID,
    in bool isFrontFace: SV_IsFrontFace,
    in float3 bary: SV_Barycentrics)
{
    // const GeometryInfo geometry = geometries[coarseVertex.geoID];
    // const int3 index = int3(fetchTriangleIndices(geometry, primitiveID));

    // MaterialInfo material = materials[geometry.materialID];

    // float3 vertexPositions[3];
    // vertexPositions[0] = fetchVertexPosition(index[0] + int(geometry.vertexOffset));
    // vertexPositions[1] = fetchVertexPosition(index[1] + int(geometry.vertexOffset));
    // vertexPositions[2] = fetchVertexPosition(index[2] + int(geometry.vertexOffset));

    // const float4x4 o2w = ObjectToWorld(geometry);
    // const float3 positionOS = interpolate(vertexPositions, bary);
    // const float3 positionWS = mul(float4(positionOS, 1.0), o2w).xyz;

    // float2 vertexUVs[3];
    // vertexUVs[0] = fetchVertexTexCoord(index[0] + int(geometry.vertexOffset));
    // vertexUVs[1] = fetchVertexTexCoord(index[1] + int(geometry.vertexOffset));
    // vertexUVs[2] = fetchVertexTexCoord(index[2] + int(geometry.vertexOffset));
    // float2 uv = interpolate(vertexUVs, bary);

    // float3 objectSpaceFlatNormal = normalize(cross(
    //     vertexPositions[1] - vertexPositions[0],
    //     vertexPositions[2] - vertexPositions[0]));

    // float4x4 o2wn = ObjectToWorldNormal(geometry);
    // float3 flatNormal = normalize(mul(float4(objectSpaceFlatNormal, 0.0), o2wn).xyz);
    // const float3 viewDirection = positionWS - globalUniform.cameraData.posW;
    // bool faceforward = dot(flatNormal, viewDirection) < 0;
    // flatNormal = faceforward ? flatNormal : -flatNormal;

    // float3 normals[3];
    // normals[0] = fetchVertexNormal(index[0] + int(geometry.vertexOffset));
    // normals[1] = fetchVertexNormal(index[1] + int(geometry.vertexOffset));
    // normals[2] = fetchVertexNormal(index[2] + int(geometry.vertexOffset));
    // float3 geometryNormalOS = interpolate(normals, bary);
    // float3 geometryNormalWS = normalize(mul(float4(geometryNormalOS, 0.0), o2wn).xyz);
    // geometryNormalWS = faceforward ? geometryNormalWS : -geometryNormalWS;

    // float3 tangents[3];
    // tangents[0] = fetchVertexTangent(index[0] + int(geometry.vertexOffset));
    // tangents[1] = fetchVertexTangent(index[1] + int(geometry.vertexOffset));
    // tangents[2] = fetchVertexTangent(index[2] + int(geometry.vertexOffset));
    // float3 tangentOS = interpolate(tangents, bary);
    // float4 tangentWS = float4(normalize(mul(float4(tangentOS, 0), o2w).xyz), geometry.oddNegativeScaling);

    // float3 shadingNormal = geometryNormalWS;
    // float3 bitangent = cross(geometryNormalWS, tangentWS.xyz) * tangentWS.w;
    // float3x3 TBN = float3x3(tangentWS.xyz, bitangent, geometryNormalWS);
    // float3 normalMap = float3(0.5, 0.5, 1.0);
    // if (material.normalTextureIndex != -1) {
    //     normalMap = textures[material.normalTextureIndex].Sample(uv).rgb;
    //     shadingNormal = normalize(mul((normalMap * 2 - 1), TBN));
    // }

    const int3 voxelID = int3(coarseVertex.posVS * gVD.voxelSize);
    uOpacityVox[voxelID] = 1.0;
}
