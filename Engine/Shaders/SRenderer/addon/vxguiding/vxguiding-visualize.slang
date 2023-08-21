#include "../../include/common/camera.hlsli"
#include "../../include/common/geometry.hlsli"
#include "../vxgi/include/conetrace_utils.hlsli"
#include "include/vxguiding_interface.hlsli"

RWStructuredBuffer<uint4> u_pMin;
RWStructuredBuffer<uint4> u_pMax;

cbuffer CameraBuffer {
    CameraData gCamera;
};

[[vk::push_constant]]
cbuffer PushConstantBuffer {
    float2 resolution;
    float line_width;
}

cbuffer VoxelizerUniform {
    VoxerlizerData gVD;
}

struct AssembledVertex {
    int vertexId : SV_VertexId;
    int instanceId : SV_InstanceID;
};
struct VertexStageOutput {
    float4 sv_position : SV_Position;
};

[shader("vertex")]
VertexStageOutput vertexMain(
    AssembledVertex assembledVertex)
{
    const float3 cube[8] = {
        float3(-0.5, -0.5, -0.5),
        float3(-0.5, -0.5, 0.5),
        float3(-0.5, 0.5, -0.5),
        float3(-0.5, 0.5, 0.5),
        float3(0.5, -0.5, -0.5),
        float3(0.5, -0.5, 0.5),
        float3(0.5, 0.5, -0.5),
        float3(0.5, 0.5, 0.5)
    };

    const int2 edge[12] = {
        int2(0, 1),
        int2(0, 2),
        int2(0, 4),
        int2(1, 3),
        int2(1, 5),
        int2(2, 3),
        int2(2, 6),
        int2(3, 7),
        int2(4, 5),
        int2(4, 6),
        int2(5, 7),
        int2(6, 7)
    };

    const float3 quad[30] = {
        // triangle 0
        float3(0, -0.5, 0),
        float3(0, -0.5, 1),
        float3(0, 0.5, 1),
        // triangle 1
        float3(0, -0.5, 0),
        float3(0, 0.5, 1),
        float3(0, 0.5, 0),
        // triangle 2
        float3(0, 0, 0),
        float3(3.061616997868383e-17, 0.5, 0),
        float3(-0.35355339059327373, 0.3535533905932738, 0),
        // triangle 3
        float3(0, 0, 0),
        float3(-0.35355339059327373, 0.3535533905932738, 0),
        float3(-0.5, 6.123233995736766e-17, 0),
        // triangle 4
        float3(0, 0, 0),
        float3(-0.5, 6.123233995736766e-17, 0),
        float3(-0.35355339059327384, -0.35355339059327373, 0),
        // triangle 5
        float3(0, 0, 0),
        float3(-0.35355339059327384, -0.35355339059327373, 0),
        float3(-9.184850993605148e-17, -0.5, 0),
        // triangle 6
        float3(0, 0, 1),
        float3(-9.184850993605148e-17, -0.5, 1),
        float3(0.3535533905932737, -0.35355339059327384, 1),
        // triangle 7
        float3(0, 0, 1),
        float3(0.3535533905932737, -0.35355339059327384, 1),
        float3(0.5, -1.2246467991473532e-16, 1),
        // triangle 8
        float3(0, 0, 1),
        float3(0.5, -1.2246467991473532e-16, 1),
        float3(0.35355339059327384, 0.3535533905932737, 1),
        // triangle 9
        float3(0, 0, 1),
        float3(0.35355339059327384, 0.3535533905932737, 1),
        float3(1.5308084989341916e-16, 0.5, 1)
    };

    // 12 edges per cube
    // 10 triangles per edge
    // 3 vertices per triangle

    // Each instance is a cube
    const int instanceID = assembledVertex.instanceId;
    // Each instance will have 12 edges
    const int edgeID = assembledVertex.vertexId / 30;
    // Each instance will have 360 (12 * 10 * 3) vertices
    const int vertexID = assembledVertex.vertexId % 30;
    // Map instanceID to 64x64x64 grid
    const int3 voxelID = int3(instanceID % 64, (instanceID / 64) % 64, instanceID / (64 * 64));

    int voxelIDFlatten = FlatIndex(voxelID, 64);

    VoxelTexInfo info = CreateVoxelTexInfo(gVD);
    AABB voxel_bound = VoxelToBound(voxelID, 0, info);
    AABB compact_bound = UnpackCompactAABB(voxel_bound, u_pMin[voxelIDFlatten].xyz, u_pMax[voxelIDFlatten].xyz);
    float3 bound_extend = voxel_bound.max - voxel_bound.min;

    if (any(compact_bound.min > compact_bound.max)) {
        VertexStageOutput degenerate_output;
        degenerate_output.sv_position = float4(0.0f, 0.0f, 0.0f, 0.0f);
        return degenerate_output;
    }
    
    const float3 center = 0.5 * (compact_bound.min + compact_bound.max);
    const float3 extend = compact_bound.max - compact_bound.min;
    
    const float3 vertex0 = center + extend * cube[edge[edgeID].x];
    const float3 vertex1 = center + extend * cube[edge[edgeID].y];

    const float3 position = quad[vertexID];

    // get vertex positions
    const float4 clip0 = mul(float4(vertex0, 1.0f), gCamera.viewProjMat);
    const float4 clip1 = mul(float4(vertex1, 1.0f), gCamera.viewProjMat);

    const float2 screen0 = resolution * (0.5 * clip0.xy / clip0.w + 0.5);
    const float2 screen1 = resolution * (0.5 * clip1.xy / clip1.w + 0.5);

    float2 xBasis = normalize(screen1 - screen0);
    float2 yBasis = float2(-xBasis.y, xBasis.x);
    float2 pt0 = screen0 + line_width * (position.x * xBasis + position.y * yBasis);
    float2 pt1 = screen1 + line_width * (position.x * xBasis + position.y * yBasis);
    float2 pt = lerp(pt0, pt1, position.z);
    float4 clip = lerp(clip0, clip1, position.z);
    
    VertexStageOutput output;
    output.sv_position = float4(clip.w * (2.0 * pt / resolution - 1.0), clip.z, clip.w);
    return output;
}

[shader("fragment")]
float4 fragmentMain(VertexStageOutput vsOut) : SV_TARGET {
    return float4(float3(1, 0, 1), 1.0f);
}
