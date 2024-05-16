#include "srenderer/scene-binding.hlsli"

RWByteAddressBuffer Position;
RWByteAddressBuffer PositionAvg;
RWByteAddressBuffer PositionCount;

[[vk::push_constant]]
cbuffer PushConstants {
    int primitive_num;
};

[shader("compute")]
[numthreads(512, 1, 1)]
void ComputeMain(
    int3 dtid: SV_DispatchThreadID,
    int3 gtid: SV_GroupThreadID,
    int3 gid: SV_GroupID,
    int gi: SV_GroupIndex
) {
    const int primitiveID = dtid.x;
    if (primitiveID >= primitive_num) return;

    const GeometryData geometry = GPUScene_geometry[0];
    const int3 index = int3(fetchTriangleIndices(geometry, primitiveID));

    const int vertexID_0 = index[0] + int(geometry.vertexOffset);
    const int vertexID_1 = index[1] + int(geometry.vertexOffset);
    const int vertexID_2 = index[2] + int(geometry.vertexOffset);

    const float3 vertex_0 = Position.Load<float3>(vertexID_0 * 4 * 3);
    const float3 vertex_1 = Position.Load<float3>(vertexID_1 * 4 * 3);
    const float3 vertex_2 = Position.Load<float3>(vertexID_2 * 4 * 3);

    PositionAvg.InterlockedAddF32(vertexID_0 * 4 * 3 + 0, vertex_1.x + vertex_2.x);
    PositionAvg.InterlockedAddF32(vertexID_0 * 4 * 3 + 4, vertex_1.y + vertex_2.y);
    PositionAvg.InterlockedAddF32(vertexID_0 * 4 * 3 + 8, vertex_1.z + vertex_2.z);

    PositionAvg.InterlockedAddF32(vertexID_1 * 4 * 3 + 0, vertex_0.x + vertex_2.x);
    PositionAvg.InterlockedAddF32(vertexID_1 * 4 * 3 + 4, vertex_0.y + vertex_2.y);
    PositionAvg.InterlockedAddF32(vertexID_1 * 4 * 3 + 8, vertex_0.z + vertex_2.z);

    PositionAvg.InterlockedAddF32(vertexID_2 * 4 * 3 + 0, vertex_0.x + vertex_1.x);
    PositionAvg.InterlockedAddF32(vertexID_2 * 4 * 3 + 4, vertex_0.y + vertex_1.y);
    PositionAvg.InterlockedAddF32(vertexID_2 * 4 * 3 + 8, vertex_0.z + vertex_1.z);
    
    PositionCount.InterlockedAdd(vertexID_0 * 4, 2);
    PositionCount.InterlockedAdd(vertexID_1 * 4, 2);
    PositionCount.InterlockedAdd(vertexID_2 * 4, 2);
}