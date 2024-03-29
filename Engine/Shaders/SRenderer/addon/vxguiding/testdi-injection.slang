#include "../../include/common/geometry.hlsli"
#include "../../include/scene_descriptor_set.hlsli"
#include "../vxgi/include/conetrace_utils.hlsli"
#include "../vxgi/include/voxelizer_utils.hlsli"
#include "../vxgi/include/vxgi_interface.hlsli"
#include "include/triangle_clip.hlsli"
#include "include/vxguiding_interface.hlsli"
#include "testdi_interface.hlsli"

[[vk::push_constant]]
cbuffer PushConstants {
    struct PushConstants {
        uint geometry_idx;
    } pConst;
};

cbuffer VoxelConfig {
    bool use_compact;
    bool z_conservative;
    bool clipping;
}

cbuffer VoxelizerUniform {
    VoxerlizerData gVD;
}

struct AssembledVertex { int vertexId : SV_VertexId; };
struct CoarseVertex {
    float3 posVS;
    float3 posWS;
};
struct VertexStageOutput {
    CoarseVertex coarseVertex : CoarseVertex;
    float4 sv_position : SV_Position; // [-1,+1]^3
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
    output.coarseVertex.posWS = positionWS;
    return output;
}

struct GeometryStageOutput {
    nointerpolation AABB aabb : AABB;
    nointerpolation float3 posVS[3] : VoxelPosition;
    nointerpolation float3 posWS[3] : WorldPosition;
    nointerpolation int dominantAxis : DominantAxis;
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
    float3 posWS[3];
    for (int i = 0; i < 3; ++i) {
        vertex[i].sv_position = float4(ProjectAlongAxis(input[i].sv_position.xyz, axis), 1.0, 1.0);
        vertex[i].dominantAxis = axis;
        posVS[i] = (input[i].coarseVertex.posVS.xyz + 1.0) * 0.5 * gVD.voxelSize;
        posWS[i] = input[i].coarseVertex.posWS;
    }
    AABB aabb = GetTriangleAABB(posVS[0], posVS[1], posVS[2]);

    for (int i = 0; i < 3; ++i) {
        vertex[i].aabb = aabb;
        vertex[i].posVS = posVS;
        vertex[i].posWS = posWS;
        output.Append(vertex[i]);
    }

    output.RestartStrip();
}

RWStructuredBuffer<uint> u_pMin;
RWStructuredBuffer<uint> u_pMax;
RWStructuredBuffer<uint> u_pIrradiance;

void TriangleVoxelBoundInjection(
    in float3 triangle[3],
    in int3 voxelID
) {
    float3 posList[9];
    posList[0] = triangle[0];
    posList[1] = triangle[1];
    posList[2] = triangle[2];

    int posCount = 3;
    VoxelTexInfo info = CreateVoxelTexInfo(gVD);
    AABB voxel_bound = VoxelToBound(voxelID, 0, info);
    int voxelIDFlatten = FlatIndex(voxelID, info.volumeDimension);
    
    if (clipping)
        ClipTriangleAgainstAABB(posList, posCount, voxel_bound);

    AABB aabb;
    if (use_compact) {
        aabb = { float3(100), -float3(100) };
        for (int i = 0; i < posCount; ++i) {
            aabb.max = max(aabb.max, posList[i]);
            aabb.min = min(aabb.min, posList[i]);
        }

        aabb.max = max(min(aabb.max, voxel_bound.max), voxel_bound.min);
        aabb.min = min(max(aabb.min, voxel_bound.min), voxel_bound.max);
    }
    else {
        aabb = voxel_bound;
    }
    uint max_uint = 0xffffffff;
    float3 voxel_bound_extend = voxel_bound.max - voxel_bound.min;
    uint3 pMin = uint3(saturate((aabb.min - voxel_bound.min) / voxel_bound_extend) * max_uint);
    uint3 pMax = uint3(saturate((aabb.max - voxel_bound.min) / voxel_bound_extend) * max_uint);

    InterlockedMin(u_pMin[voxelIDFlatten * 4 + 0], pMin.x);
    InterlockedMin(u_pMin[voxelIDFlatten * 4 + 1], pMin.y);
    InterlockedMin(u_pMin[voxelIDFlatten * 4 + 2], pMin.z);

    InterlockedMax(u_pMax[voxelIDFlatten * 4 + 0], pMax.x);
    InterlockedMax(u_pMax[voxelIDFlatten * 4 + 1], pMax.y);
    InterlockedMax(u_pMax[voxelIDFlatten * 4 + 2], pMax.z);

    InterlockedMax(u_pIrradiance[voxelIDFlatten], PackIrradiance(uTestDISetting.intensity));
}

[shader("fragment")]
void fragmentMain(
    nointerpolation in int primitiveID: SV_PrimitiveID,
    nointerpolation in AABB aabb: AABB,
    nointerpolation in float3 posVS[3]: VoxelPosition,
    nointerpolation in float3 posWS[3]: WorldPosition,
    nointerpolation in int dominantAxis: DominantAxis,
    in bool isFrontFace: SV_IsFrontFace,
    in float3 bary: SV_Barycentrics)
{
    const float3 posVSBasry = interpolate(posVS, bary);

    // Insertion center voxel
    const int3 voxelID = int3(posVSBasry);
    TriangleVoxelBoundInjection(posWS, voxelID);

    // To be more conservative, we also write to the adjacent voxels in z-axis (if necessary).
    int3 offset = int3(0, 0, 0);
    offset[dominantAxis] = 1;

    if (z_conservative) {
        // check the voxel rear to the dominant axis
        if (voxelID[dominantAxis] < gVD.voxelSize - 1 && AABBConditionTest(voxelID + offset, aabb)) {
            if (EdgeVoxelConditionTest(posVS, voxelID + offset + float3(0.5))) {
                // Has intersection
                TriangleVoxelBoundInjection(posWS, voxelID + offset);
            }
        }
        
        // check the voxel back to the dominant axis
        if (voxelID[dominantAxis] > 0 && AABBConditionTest(voxelID - offset, aabb)) {
            if (EdgeVoxelConditionTest(posVS, voxelID - offset + float3(0.5))) {
                // Has intersection
                TriangleVoxelBoundInjection(posWS, voxelID - offset);
            }
        }
    }
}