#include "../tinymlp/tinynn-activations.hlsli"

[[vk::push_constant]]
cbuffer PushConstants {
    float3 bounds_min;
    float random_seed;
    float3 bounds_max;
    // float random_seed2;
};

// RWStructuredBuffer<float> u_image;

StructuredBuffer<float> u_position;
StructuredBuffer<float> u_depth;
StructuredBuffer<float> u_rgb_embedding;
StructuredBuffer<float> u_sigma_embedding;
StructuredBuffer<float> u_rgb_embedding_grad;
StructuredBuffer<float> u_sigma_embedding_grad;

RWStructuredBuffer<float> u_rgb;
RWStructuredBuffer<float> u_sigma;
RWStructuredBuffer<float> u_index;

static const int3 resolution = int3(64, 64, 64);

int get_voxel_indices(float3 xyz) {
    if (!(all(xyz > bounds_min) && all(xyz < bounds_max))) {
        xyz = clamp(xyz, bounds_min, bounds_max);
    }
    const float3 grid_size = (bounds_max - bounds_min) / resolution;
    const int3 vox_index = clamp(int3(floor((xyz - bounds_min) / grid_size)), 0, resolution - 1);
    return vox_index[0] * resolution.y * resolution.z + vox_index[1] * resolution.z + vox_index[2];
}

groupshared float shared_buffer[4][65];

[shader("compute")]
[numthreads(256, 1, 1)]
void ComputeMain(
    int3 dtid: SV_DispatchThreadID,
    int3 gtid: SV_GroupThreadID,
    int3 gid: SV_GroupID,
    int gi: SV_GroupIndex
) {
    const float3x3 intrinsic = float3x3(
        875., 0., 400.,
        0., 875., 400.,
        0.0, 0.0, 1.0
    );

    const float3x3 R = float3x3(
        9.4932e-01, 1.4144e-01, -2.8069e-01,
        3.1431e-01, -4.2719e-01, 8.4777e-01,
        -2.6000e-08, -8.9303e-01, -4.5000e-01
    );
    const float3 t = float3(0.8421, -2.5433, 1.3500);


    const int globalID = dtid.x;
    const int rayID = globalID / 64;
    const int sample_index = gi % 64;

    const int x_int = rayID % 800;
    const int y_int = rayID / 800;
    const float x = (x_int + 0.5) / 800 * 2 - 1.f;
    const float y = (y_int + 0.5) / 800 * 2 - 1.f;
    const float z = 1 * 875. / 400.;
    const float3 img_coords = float3(x, y, z);

    const float3 origin = t;
    const float3 direction = normalize(mul(R, img_coords));
    const float depth = 5 * sample_index * 1.f / 63.f;
    const float3 position = origin + direction * depth;

    // const float3 position = float3(u_position[globalID * 3 + 0],
    //     u_position[globalID * 3 + 1], u_position[globalID * 3 + 2]);
    // const float depth = u_depth[globalID];
    // get voxel index
    const int voxel_index = get_voxel_indices(position);
    const int ray_index = gi / 64;
    shared_buffer[ray_index][sample_index] = depth;
    if (sample_index == 0) shared_buffer[ray_index][64] = 1e10;

    float3 rgb = float3(0.f);
    float sigma = 0.0f;
    // rgb = float3(
    //     u_rgb_embedding[voxel_index * 3 + 0],
    //     u_rgb_embedding[voxel_index * 3 + 1],
    //     u_rgb_embedding[voxel_index * 3 + 2]);
    rgb = float3(
        u_sigma_embedding_grad[voxel_index * 3 + 0],
        u_rgb_embedding_grad[voxel_index * 3 + 1],
        u_rgb_embedding_grad[voxel_index * 3 + 2]);
    sigma = LeakyReLU.eval(u_sigma_embedding[voxel_index]);
    rgb = Sigmoid.eval(rgb);
    if (!(all(position >= bounds_min) && all(position <= bounds_max))) {
        sigma = 0.0f;
    }
    
    GroupMemoryBarrierWithGroupSync();
    const float next_depth = shared_buffer[ray_index][sample_index + 1];
    const float delta = next_depth - depth;
    const float alpha = 1.0 - exp(-sigma * delta);
    const float temp = 1. - alpha + 1e-10;
    shared_buffer[ray_index][sample_index + 1] = temp;
    if (sample_index == 0) shared_buffer[ray_index][0] = 1.f;
    GroupMemoryBarrierWithGroupSync();

    float prod = shared_buffer[ray_index][sample_index];
    float cumprod = WavePrefixProduct(prod) * prod;
    if (sample_index == 31) shared_buffer[ray_index][0] = cumprod;
    GroupMemoryBarrierWithGroupSync();
    if (sample_index >= 32) cumprod *= shared_buffer[ray_index][0];
    const float weight = alpha * cumprod;

    const float4 weighted_rgb = float4(weight * rgb, weight);
    const float4 rgb_sum = WaveActiveSum(weighted_rgb);

    GroupMemoryBarrierWithGroupSync();
    if (sample_index < 4) shared_buffer[ray_index][sample_index] = rgb_sum[sample_index];
    GroupMemoryBarrierWithGroupSync();
    if (sample_index == 63) {
        const float4 firstRGBA = float4(shared_buffer[ray_index][0],
                                        shared_buffer[ray_index][1], 
                                        shared_buffer[ray_index][2],
                                        shared_buffer[ray_index][3]);
        const float4 totalRGBA = firstRGBA + rgb_sum;
        float3 rgb_map = totalRGBA.rgb;
        
        // rgb_map += float3(1 - totalRGBA.a);
        
        u_rgb[rayID * 3 + 0] = rgb_map.r;
        u_rgb[rayID * 3 + 1] = rgb_map.g;
        u_rgb[rayID * 3 + 2] = rgb_map.b;
        // u_index[rayID] = 1.f - totalRGBA.a;
    }

    // u_index[globalID] = weight;

    // u_rgb[globalID * 3 + 0] = rgb.r;
    // u_rgb[globalID * 3 + 1] = rgb.g;
    // u_rgb[globalID * 3 + 2] = rgb.b;
    // u_sigma[globalID] = sigma;
}