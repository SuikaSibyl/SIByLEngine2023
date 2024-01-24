#include "../../../include/common/geometry.hlsli"
#include "../../../include/scene_descriptor_set.hlsli"
#include "../../vxgi/include/conetrace_utils.hlsli"
#include "../../vxgi/include/voxelizer_utils.hlsli"
#include "../../vxgi/include/vxgi_interface.hlsli"
#include "../include/triangle_clip.hlsli"
#include "../include/vxguiding_interface.hlsli"
#include "common.hlsli"

RWStructuredBuffer<uint> u_hashRecord;
RWStructuredBuffer<uint> u_boundMin;
RWStructuredBuffer<uint> u_boundMax;

[[vk::push_constant]]
cbuffer PushConstants {
    uint geometry_idx;
};

struct AssembledVertex { int vertexId : SV_VertexId; };
struct CoarseVertex {
    float3 posVS;
};
struct VertexStageOutput {
    CoarseVertex coarseVertex : CoarseVertex;
    float4 sv_position : SV_Position;   // [-1,+1]^3
};

[shader("vertex")]
VertexStageOutput vertexMain(
    AssembledVertex assembledVertex)
{
    const GeometryInfo geometry = geometries[geometry_idx];
    const float4x4 o2w = ObjectToWorld(geometry);

    const float3 positionOS = fetchVertexPosition(assembledVertex.vertexId);
    const float3 positionWS = mul(float4(positionOS, 1.0), o2w).xyz;

    const float3 extent = u_config.aabbMax - u_config.aabbMin;
    const float extentMax = max(extent.x, max(extent.y, extent.z)) * 0.5 + 0.01;
    const float3 center = (u_config.aabbMax + u_config.aabbMin) * 0.5;
    
    VertexStageOutput output;
    output.sv_position = float4((positionWS.xyz - center) / extentMax, 1.0);
    output.coarseVertex.posVS = output.sv_position.xyz;
    return output;
}

struct GeometryStageOutput {
    nointerpolation float3 posVS[3] : VoxelPosition;
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
    for (int i = 0; i < 3; ++i) {
        vertex[i].sv_position = float4(ProjectAlongAxis(input[i].sv_position.xyz, axis), 1.0, 1.0);
        vertex[i].dominantAxis = axis;
        posVS[i] = (input[i].coarseVertex.posVS.xyz + 1.0) * 0.5 * u_config.size;
    }
    
    for (int i = 0; i < 3; ++i) {
        // vertex[i].aabb = aabb;
        vertex[i].posVS = posVS;
        output.Append(vertex[i]);
    }

    output.RestartStrip();
}

void TriangleVoxelBoundInjection(
    in float3 triangle[3],
    in int3 voxelID
) {
    int posCount = 3;
    float3 posList[9];
    posList[0] = triangle[0];
    posList[1] = triangle[1];
    posList[2] = triangle[2];

    VoxelTexInfo info = CreateVoxelTexInfo(u_config);
    AABB voxel_bound;
    voxel_bound.min = float3(voxelID);
    voxel_bound.max = float3(voxelID) + float3(1);

    uint32_t bucketID = QueryBucketID(voxelID, u_config);
    uint32_t fingerprint = QueryKey(voxelID);
    int cellID = QueryCellID(bucketID, fingerprint, u_config, u_hashRecord);
    if (cellID == -1) return;
    
    if (u_config.clipping)
        ClipTriangleAgainstAABB(posList, posCount, voxel_bound);
    
    AABB aabb;
    if (u_config.use_compact) {
        aabb = { float3(99999), -float3(99999) };
        for (int i = 0; i < posCount; ++i) {
            aabb.max = max(aabb.max, posList[i]);
            aabb.min = min(aabb.min, posList[i]);
        }
        aabb.max = max(min(aabb.max, voxel_bound.max), voxel_bound.min);
        aabb.min = min(max(aabb.min, voxel_bound.min), voxel_bound.max);
    } else {
        aabb = voxel_bound;
    }
    
    // quantize the result
    uint max_uint = 0xffffffff;
    float3 voxel_bound_extend = voxel_bound.max - voxel_bound.min;
    uint3 pMin = uint3(saturate((aabb.min - voxel_bound.min) / voxel_bound_extend) * max_uint);
    uint3 pMax = uint3(saturate((aabb.max - voxel_bound.min) / voxel_bound_extend) * max_uint);
    
    uint32_t globalID = GetGlobalID(bucketID, cellID, u_config);
    
    // inject the result
    InterlockedMin(u_boundMin[globalID * 4 + 0], pMin.x);
    InterlockedMin(u_boundMin[globalID * 4 + 1], pMin.y);
    InterlockedMin(u_boundMin[globalID * 4 + 2], pMin.z);

    InterlockedMax(u_boundMax[globalID * 4 + 0], pMax.x);
    InterlockedMax(u_boundMax[globalID * 4 + 1], pMax.y);
    InterlockedMax(u_boundMax[globalID * 4 + 2], pMax.z);
}

[shader("fragment")]
void fragmentMain(
    nointerpolation in int primitiveID: SV_PrimitiveID,
    nointerpolation in float3 posVS[3]: VoxelPosition,
    nointerpolation in int dominantAxis: DominantAxis,
    in bool isFrontFace: SV_IsFrontFace,
    in float3 bary: SV_Barycentrics)
{
    AABB aabb = GetTriangleAABB(posVS[0], posVS[1], posVS[2]);

    const float3 posVSBasry = interpolate(posVS, bary);

    // Insertion center voxel
    const int3 voxelID = int3(posVSBasry);
    TriangleVoxelBoundInjection(posVS, voxelID);
    
    // To be more conservative, we also write to the adjacent voxels in z-axis (if necessary).
    int3 offset = int3(0, 0, 0);
    offset[dominantAxis] = 1;

    if (u_config.z_conservative) {
        // check the voxel rear to the dominant axis
        if (voxelID[dominantAxis] < u_config.size - 1 && AABBConditionTest(voxelID + offset, aabb)) {
            if (EdgeVoxelConditionTest(posVS, voxelID + offset + float3(0.5))) {
                // Has intersection
                TriangleVoxelBoundInjection(posVS, voxelID + offset);
            }
        }
        
        // check the voxel back to the dominant axis
        if (voxelID[dominantAxis] > 0 && AABBConditionTest(voxelID - offset, aabb)) {
            if (EdgeVoxelConditionTest(posVS, voxelID - offset + float3(0.5))) {
                // Has intersection
                TriangleVoxelBoundInjection(posVS, voxelID - offset);
            }
        }
    }
}