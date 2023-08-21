#include "../../include/common/geometry.hlsli"
#include "../../include/scene_descriptor_set.hlsli"
#include "../vxgi/include/voxelizer_utils.hlsli"
#include "../vxgi/include/vxgi_interface.hlsli"
#include "testdi_interface.hlsli"

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
    float3 normalWS;
};
struct VertexStageOutput {
    CoarseVertex coarseVertex : CoarseVertex;
    float4 sv_position : SV_Position;   // [-1,+1]^3
};

cbuffer TestDISettingUniform {
    TestDISetting uTestDISetting;
}

[shader("vertex")]
VertexStageOutput vertexMain(
    AssembledVertex assembledVertex)
{
    float2 quad_vert[6] = {
        float2(-1.0, -1.0),
        float2(-1.0, +1.0),
        float2(+1.0, -1.0),
        float2(+1.0, -1.0),
        float2(-1.0, +1.0),
        float2(+1.0, +1.0)
    };

    const float4x4 o2w = uTestDISetting.worldMat;
    const float3 positionOS = float3(quad_vert[assembledVertex.vertexId] * uTestDISetting.pMax, 0.0);
    const float3 positionWS = mul(float4(positionOS, 1.0), o2w).xyz;

    const float3 extent = gVD.aabbMax - gVD.aabbMin;
    const float extentMax = max(extent.x, max(extent.y, extent.z)) * 0.5 + 0.01;
    const float3 center = (gVD.aabbMax + gVD.aabbMin) * 0.5;

    VertexStageOutput output;
    output.sv_position = float4((positionWS.xyz - center) / extentMax, 1.0);
    output.coarseVertex.posVS = output.sv_position.xyz;
    output.coarseVertex.normalWS = mul(uTestDISetting.localMat, float4(0.0, 0.0, 1.0, 0.0)).xyz;
    return output;
}

struct GeometryStageOutput {
    CoarseVertex coarseVertex : CoarseVertex;
    AABB aabb : AABB;
    float3 posVS[3] : VoxelPosition;
    int dominantAxis : DominantAxis;
    float4 sv_position : SV_Position; // [-1,+1]^3
};

[shader("geometry")]
[instance(1)]
[maxvertexcount(3)]
void geometryMain(
    in VertexStageOutput input[3],
    uint InstanceID: SV_GSInstanceID,
    inout TriangleStream<GeometryStageOutput> output)
{
    const int axis = GetDominantAxis(input[0].sv_position.xyz,
                                     input[1].sv_position.xyz,
                                     input[2].sv_position.xyz);
    GeometryStageOutput vertex[3];
    float3 posVS[3];
    for (int i = 0; i < 3; ++i) {
        vertex[i].sv_position = float4(ProjectAlongAxis(input[i].sv_position.xyz, axis), 1.0, 1.0);
        vertex[i].coarseVertex.posVS = (input[i].coarseVertex.posVS.xyz + 1.0) * 0.5;
        vertex[i].coarseVertex.normalWS = input[i].coarseVertex.normalWS;
        vertex[i].dominantAxis = axis;
        posVS[i] = vertex[i].coarseVertex.posVS * gVD.voxelSize;
    }
    AABB aabb = GetTriangleAABB(posVS[0], posVS[1], posVS[2]);
    
    for (int i = 0; i < 3; ++i) {
        vertex[i].aabb = aabb;
        vertex[i].posVS = posVS;
        output.Append(vertex[i]);
    }

    output.RestartStrip();
}

RWTexture3D<float2> uRadopaVox6D[6];   // 64 x 64 x 64

[shader("fragment")]
void fragmentMain(
    in CoarseVertex coarseVertex: CoarseVertex,
    in AABB aabb: AABB,
    in float3 posVS[3]: VoxelPosition,
    in int dominantAxis: DominantAxis,
    nointerpolation in int primitiveID: SV_PrimitiveID,
    in bool isFrontFace: SV_IsFrontFace,
    in float3 bary: SV_Barycentrics)
{
    const int3 voxelID = int3(coarseVertex.posVS * gVD.voxelSize);

    const float3 faceNormals[6] = {
        float3(+1.0, +0.0, +0.0),
        float3(-1.0, +0.0, +0.0),
        float3(+0.0, +1.0, +0.0),
        float3(+0.0, -1.0, +0.0),
        float3(+0.0, +0.0, +1.0),
        float3(+0.0, +0.0, -1.0)
    };
    const float3 normal = coarseVertex.normalWS;
    
    for (int i = 0; i < 6; ++i) {
        const float projLumin = saturate(dot(normal, faceNormals[i]));
        uRadopaVox6D[i][voxelID] = float2(projLumin, 1.0);
    }

    // To be more conservative, we also write to the adjacent voxels in z-axis (if necessary).
    int3 offset = int3(0, 0, 0);
    offset[dominantAxis] = 1;

    // check the voxel rear to the dominant axis
    if (voxelID[dominantAxis] < gVD.voxelSize - 1 && AABBConditionTest(voxelID + offset, aabb)) {
        if (EdgeVoxelConditionTest(posVS, voxelID + offset + float3(0.5))) {
            for (int i = 0; i < 6; ++i) {
                const float projLumin = saturate(dot(normal, faceNormals[i]));
                uRadopaVox6D[i][voxelID + offset] = float2(projLumin, 1.0);
            }
        }
    }

    // check the voxel back to the dominant axis
    if (voxelID[dominantAxis] > 0 && AABBConditionTest(voxelID - offset, aabb)) {
        if (EdgeVoxelConditionTest(posVS, voxelID - offset + float3(0.5))) {
            for (int i = 0; i < 6; ++i) {
                const float projLumin = saturate(dot(normal, faceNormals[i]));
                uRadopaVox6D[i][voxelID - offset] = float2(projLumin, 1.0);
            }
        }
    }
}
