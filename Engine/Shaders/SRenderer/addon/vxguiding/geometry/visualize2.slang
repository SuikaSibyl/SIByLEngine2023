#include "../../../include/common/camera.hlsli"
#include "../../../include/common/geometry.hlsli"
#include "../../vxgi/include/conetrace_utils.hlsli"
#include "../include/vxguiding_interface.hlsli"
#include "common.hlsli"

const StructuredBuffer<uint4> u_boundMin;
const StructuredBuffer<uint4> u_boundMax;
const StructuredBuffer<uint> u_hashRecord;
const StructuredBuffer<float> u_irradiance;
const StructuredBuffer<uint> u_compactIndex;
const StructuredBuffer<int> u_cluster;
const StructuredBuffer<uint> u_vxCounter;

cbuffer CameraBuffer {
    CameraData gCamera;
};

[[vk::push_constant]]
cbuffer PushConstantBuffer {
    int play_mode;
}

struct AssembledVertex {
    int vertexId : SV_VertexId;
    int instanceId : SV_InstanceID;
};
struct VertexStageOutput {
    float4 sv_position : SV_Position;
    float irradiance : Irradiance;
    nointerpolation uint clusterIdx : ClusterID;
};

[shader("vertex")]
VertexStageOutput vertexMain(
    AssembledVertex assembledVertex)
{
    const float3 cube[36] = {
      float3(-0.5f, -0.5f, -0.5f),
      float3( 0.5f, -0.5f, -0.5f),
      float3( 0.5f,  0.5f, -0.5f),
      float3( 0.5f,  0.5f, -0.5f),
      float3(-0.5f,  0.5f, -0.5f),
      float3(-0.5f, -0.5f, -0.5f),
  
      float3(-0.5f, -0.5f,  0.5f),
      float3( 0.5f, -0.5f,  0.5f),
      float3( 0.5f,  0.5f,  0.5f),
      float3( 0.5f,  0.5f,  0.5f),
      float3(-0.5f,  0.5f,  0.5f),
      float3(-0.5f, -0.5f,  0.5f),
  
      float3(-0.5f,  0.5f,  0.5f),
      float3(-0.5f,  0.5f, -0.5f),
      float3(-0.5f, -0.5f, -0.5f),
      float3(-0.5f, -0.5f, -0.5f),
      float3(-0.5f, -0.5f,  0.5f),
      float3(-0.5f,  0.5f,  0.5f),
  
      float3( 0.5f,  0.5f,  0.5f),
      float3( 0.5f,  0.5f, -0.5f),
      float3( 0.5f, -0.5f, -0.5f),
      float3( 0.5f, -0.5f, -0.5f),
      float3( 0.5f, -0.5f,  0.5f),
      float3( 0.5f,  0.5f,  0.5f),
  
      float3(-0.5f, -0.5f, -0.5f),
      float3( 0.5f, -0.5f, -0.5f),
      float3( 0.5f, -0.5f,  0.5f),
      float3( 0.5f, -0.5f,  0.5f),
      float3(-0.5f, -0.5f,  0.5f),
      float3(-0.5f, -0.5f, -0.5f),
    
      float3(-0.5f,  0.5f, -0.5f),
      float3( 0.5f,  0.5f, -0.5f),
      float3( 0.5f,  0.5f,  0.5f),
      float3( 0.5f,  0.5f,  0.5f),
      float3(-0.5f,  0.5f,  0.5f),
      float3(-0.5f,  0.5f, -0.5f),
    };

    // 12 triangles per cube
    // 3 vertices per triangle
    VoxelTexInfo info = GetVoxelTexInfo();
    // Each instance is a cube
    const int instanceID = assembledVertex.instanceId;
    const int vertexID = assembledVertex.vertexId;
    const uint vx_count = u_vxCounter[0];

    if (instanceID >= vx_count) {
        VertexStageOutput degenerate_output;
        degenerate_output.sv_position = float4(0.0f, 0.0f, 0.0f, 0.0f);
        return degenerate_output;
    }

    const uint flattenID = u_compactIndex[instanceID];
    const uint clusterID = u_cluster[instanceID];
    const float irradiance = u_irradiance[instanceID];
    const int3 voxelID = ReconstructIndex(flattenID, u_config, u_hashRecord);

    AABB voxel_bound = VoxelToBound(voxelID, 0, info);
    AABB compact_bound = UnpackCompactAABB(voxel_bound, u_boundMin[flattenID].xyz, u_boundMax[flattenID].xyz);
    float3 bound_extend = voxel_bound.max - voxel_bound.min;
    
    if (any(voxelID < 0) || any(compact_bound.min > compact_bound.max) || irradiance < 0) {
        VertexStageOutput degenerate_output;
        degenerate_output.sv_position = float4(0.0f, 0.0f, 0.0f, 0.0f);
        return degenerate_output;
    }

    const float3 center = 0.5 * (compact_bound.min + compact_bound.max);
    const float3 extend = compact_bound.max - compact_bound.min + float3(0.005f);

    const float3 positionWS = center + extend * cube[vertexID];
    const float4 positionCS = mul(float4(positionWS, 1.0f), gCamera.viewProjMat);

    VertexStageOutput output;
    output.sv_position = positionCS;
    output.irradiance = irradiance;
    output.clusterIdx = clusterID;
    return output;
}

[shader("fragment")]
float4 fragmentMain(VertexStageOutput vsOut) : SV_TARGET {
    if (play_mode == 0) {
        return float4(float3(vsOut.irradiance), 1.f);
    } else if (play_mode == 1) {
        uint clusterID = murmur3(vsOut.clusterIdx);
        float3 color = UNPACK_FLOAT3_UINT(clusterID);
        return float4(color, 1.f);
    }
    
    return float4(float3(1, 0, 1), 1.0f);
}