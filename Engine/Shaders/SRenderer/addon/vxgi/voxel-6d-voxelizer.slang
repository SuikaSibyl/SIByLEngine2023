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
        vertex.sv_position = float4(project(input[i].sv_position.xyz, axis), 1.0, 1.0);
        vertex.coarseVertex.posVS = (input[i].coarseVertex.posVS.xyz + 1.0) * 0.5;
        output.Append(vertex);
    }
    output.RestartStrip();
}

const RWTexture3D<float2> uRadOpaVox6D[6];

[shader("fragment")]
void fragmentMain(
    in CoarseVertex coarseVertex: CoarseVertex,
    nointerpolation in int primitiveID: SV_PrimitiveID,
    in bool isFrontFace: SV_IsFrontFace,
    in float3 bary: SV_Barycentrics)
{
    const int3 voxelID = int3(coarseVertex.posVS * gVD.voxelSize);
    for (int i = 0; i < 6; ++i) {
        uRadOpaVox6D[i][voxelID] = float2(0.0, 1.0);
    }
}
