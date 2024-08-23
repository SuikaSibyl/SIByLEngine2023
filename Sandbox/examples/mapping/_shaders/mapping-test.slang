#include "common/mapping.hlsli"
#include "common/math.hlsli"
#include "common/random.hlsli"
#include "guiding/octmap.hlsli"

RWStructuredBuffer<float2> input_buffer;
RWStructuredBuffer<float2> disk_buffer;
RWStructuredBuffer<float4> sphere_buffer;
RWStructuredBuffer<float2> disk2_buffer;

RWStructuredBuffer<float2> warp_buffer;
RWStructuredBuffer<float4> warp_3d_buffer;
RWByteAddressBuffer octmap_buffer;

[shader("compute")]
[numthreads(256, 1, 1)]
void ComputeMain(
    int3 dtid: SV_DispatchThreadID,
    int3 gtid: SV_GroupThreadID,
    int3 gid: SV_GroupID,
    int gi: SV_GroupIndex
) {
    if (gid.x == 0) {
        float2 input = input_buffer[dtid.x];
        float2 disk = mapping::square_to_disk_shirley(input);
        disk_buffer[dtid.x] = disk;

        float3 hsph = mapping::disc_to_hemisphere_lambert(disk);
        sphere_buffer[dtid.x] = float4(hsph, 1.0);

        float2 disk2 = mapping::hemisphere_to_disc_lambert(hsph);
        float2 square = mapping::disk_to_square_shirley(disk2);
        disk2_buffer[dtid.x] = square;
    }
    
    RandomSamplerState RNG = InitRandomSampler(dtid.x, 0);

    float2 uv = GetNextRandomFloat2(RNG);
    Octmap8x8 octmap;
    octmap.offset = 0;
    float3 uv_pdf = octmap.sample_quadmap(octmap_buffer, uv);
    uv = uv_pdf.xy;

    float2 disk = mapping::square_to_disk_shirley(uv);
    float3 hsph = mapping::disc_to_hemisphere_lambert(disk);

    warp_buffer[dtid.x] = uv;
    warp_3d_buffer[dtid.x] = float4(hsph, 1.0);

    // float2 square = mapping::disk_to_square_shirley(disk2);
    // sphere_buffer[dtid.x] = float4(square, 0.0, 1.0);
}