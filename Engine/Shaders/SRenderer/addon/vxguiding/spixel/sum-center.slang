#include "../../../include/common/cpp_compatible.hlsli"
#include "../../../include/common/octahedral.hlsli"

[[vk::push_constant]]
cbuffer PushConstants {
    int2 map_size;
    int2 img_size;
    int spixel_size;
};

Texture2D<float4> u_input;
RWTexture2D<float4> u_center;

groupshared float3 warp_pos_table[32];
groupshared float3 warp_norm_table[32];
groupshared int warp_num_table[32];

[shader("compute")]
[numthreads(32, 32, 1)]
void ComputeMain(int3 DTid: SV_DispatchThreadID,
                 int Gid: SV_GroupIndex) 
{
    const int2 pixel = DTid.xy;
    const bool oob = any(pixel >= img_size);
    const int2 spixel = pixel / spixel_size;

    int img_x = spixel.x * spixel_size + spixel_size / 2;
    int img_y = spixel.y * spixel_size + spixel_size / 2;
    // modify the center if it is out of image boundary
    img_x = img_x >= img_size.x ? (spixel.x * spixel_size + img_size.x) / 2 : img_x;
    img_y = img_y >= img_size.y ? (spixel.y * spixel_size + img_size.y) / 2 : img_y;
    
    if (all(pixel == int2(img_x, img_y))) {
        u_center[spixel] = u_input[pixel];
    }

    // // load the geometry information of the pixel
    // const float4 color_info = oob ? float4(0) : u_input[pixel];
    // const float3 position = color_info.xyz;
    // const float3 normal = Unorm32OctahedronToUnitVector(asuint(color_info.w));
    // const bool valid = all(!isinf(color_info.xyz)) & (!oob);

    // const int subgroup_id = Gid / 32;

    // const int wave_count = WaveActiveCountBits(valid);
    // if (valid) {
    //     // aggregate the information along the warp first
    //     const float3 wave_position = WaveActiveSum(position);
    //     const float3 wave_normal = WaveActiveSum(normal);
    //     // then aggregate the information within the whole block
    //     if (WaveIsFirstLane()) {
    //         warp_pos_table[subgroup_id] = wave_position;
    //         warp_norm_table[subgroup_id] = wave_normal;
    //         warp_num_table[subgroup_id] = wave_count;
    //     }
    // } else if (wave_count == 0) {
    //     warp_pos_table[subgroup_id] = float3(0);
    //     warp_norm_table[subgroup_id] = float3(0);
    //     warp_num_table[subgroup_id] = 0;
    // }
    
    // GroupMemoryBarrierWithGroupSync();
    
    
    // if (Gid < 32) {
    //     const float3 wave_position = warp_pos_table[Gid];
    //     const float3 wave_normal = warp_norm_table[Gid];
    //     const int warp_number = warp_num_table[Gid];
    //     const float3 block_position = WaveActiveSum(wave_position);
    //     const float3 block_normal = WaveActiveSum(wave_normal);
    //     const int block_number = WaveActiveSum(warp_number);
    //     // finally aggregate the information to the spixel
    //     if (Gid == 0) {
    //         const float3 avg_pos = block_position / block_number;
    //         const float3 avg_norm = normalize(block_normal / block_number);
    //         const float pack_norm = asfloat(UnitVectorToUnorm32Octahedron(avg_norm));
    //         u_center[spixel] = float4(avg_pos, pack_norm);
    //     }
    // }
}