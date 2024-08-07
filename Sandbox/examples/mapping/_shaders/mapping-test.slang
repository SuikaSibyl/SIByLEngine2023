#include "common/mapping.hlsli"

RWStructuredBuffer<float2> input_buffer;
RWStructuredBuffer<float2> disk_buffer;
RWStructuredBuffer<float4> sphere_buffer;
RWStructuredBuffer<float2> disk2_buffer;
// RWStructuredBuffer<float2> square_buffer;

[shader("compute")]
[numthreads(256, 1, 1)]
void ComputeMain(
    int3 dtid: SV_DispatchThreadID,
    int3 gtid: SV_GroupThreadID,
    int3 gid: SV_GroupID,
    int gi: SV_GroupIndex
) {
    float2 input = input_buffer[dtid.x];
    float2 disk = mapping::square_to_disk_shirley(input);
    disk_buffer[dtid.x] = disk;

    float3 hsph = mapping::disc_to_hemisphere_lambert(disk);
    sphere_buffer[dtid.x] = float4(hsph, 1.0);
    float2 disk2 = mapping::hemisphere_to_disc_lambert(hsph);
    disk2_buffer[dtid.x] = disk2;

    // float2 square = mapping::disk_to_square_shirley(disk2);
    // sphere_buffer[dtid.x] = float4(square, 0.0, 1.0);
}