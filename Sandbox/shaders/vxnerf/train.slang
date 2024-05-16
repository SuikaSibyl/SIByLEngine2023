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
RWByteAddressBuffer u_rgb_embedding_grad;
RWByteAddressBuffer u_sigma_embedding_grad;

RWStructuredBuffer<float> u_rgb;
RWStructuredBuffer<float> u_sigma;
RWStructuredBuffer<float> u_index;
RWStructuredBuffer<float> u_ground_truth;

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

[Differentiable]
static float L2(float3 x, no_diff float3 gt) {
    return dot((x - gt), (x - gt));
}

[BackwardDerivative(LoadRGBSigmaBwd)]
float4 LoadRGBSigma(int index) {
    float sigma = u_sigma_embedding[index];
    float3 rgb = float3(
        u_rgb_embedding[index * 3 + 0],
        u_rgb_embedding[index * 3 + 1],
        u_rgb_embedding[index * 3 + 2]);
    return float4(rgb, sigma);
}

void LoadRGBSigmaBwd(int index, in float4.Differential rgb_sigma_grad) {
    u_rgb_embedding_grad.InterlockedAddF32((index * 3 + 0) * 4, rgb_sigma_grad.r);
    u_rgb_embedding_grad.InterlockedAddF32((index * 3 + 1) * 4, rgb_sigma_grad.g);
    u_rgb_embedding_grad.InterlockedAddF32((index * 3 + 2) * 4, rgb_sigma_grad.b);
    // u_rgb_embedding_grad[index * 3 + 0] = rgb_sigma.x;
    // u_rgb_embedding_grad[index * 3 + 1] = rgb_sigma.y;
    // u_rgb_embedding_grad[index * 3 + 2] = rgb_sigma.z;
    // u_sigma_embedding_grad[index] = rgb_sigma.w;
}

[Differentiable]
float4 process_rgb_sigma(float4 rgb_sigma) {
    return float4(Sigmoid.eval(rgb_sigma.rgb), LeakyReLU.eval(rgb_sigma.w));
}

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
        0.949320773, 0.141438984, -0.280686783,
        0.314308840, -0.427194426, 0.847770553,
        -0.000000026, - 0.893028487, - 0.450000053
    );
    const float3 t = float3(
        0.842060328,
        -2.543311695,
        1.350000105);


    const int globalID = dtid.x;
    const int rayID = globalID / 64;
    const int sample_index = gi % 64;

    const int x_int = rayID % 800;
    const int y_int = rayID / 800;
    const float x = x_int * 2. / 799 - 1.f;
    const float y = y_int * 2. / 799 - 1.f;
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

    float4 rgb_sigma_raw = LoadRGBSigma(voxel_index);
    float4 rgb_sigma = process_rgb_sigma(rgb_sigma_raw);
    float3 rgb = rgb_sigma.rgb;
    float sigma = rgb_sigma.w;
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
        
        rgb_map += float3(1 - totalRGBA.a);
        
        // u_rgb[rayID * 3 + 0] = rgb_map.r;
        // u_rgb[rayID * 3 + 1] = rgb_map.g;
        // u_rgb[rayID * 3 + 2] = rgb_map.b;
        // u_index[rayID] = 1.f - totalRGBA.a;
        const float3 gt = float3(
            u_ground_truth[rayID * 3 + 0],
            u_ground_truth[rayID * 3 + 1],
            u_ground_truth[rayID * 3 + 2]
        );
        
        const float l2 = L2(rgb_map, gt);
        // backward
        
        var rgb_pair = diffPair(rgb_map);
        bwd_diff(L2)(rgb_pair, gt, 1.f);
        shared_buffer[ray_index][0] = rgb_pair.d.r;
        shared_buffer[ray_index][1] = rgb_pair.d.g;
        shared_buffer[ray_index][2] = rgb_pair.d.b;
        // shared_buffer[ray_index][3]
        // u_rgb[rayID * 3 + 0] = rgb_pair.p.r;
        // u_rgb[rayID * 3 + 1] = rgb_pair.p.g;
        // u_rgb[rayID * 3 + 2] = rgb_pair.p.b;
        
        // u_rgb[rayID * 3 + 0] = rgb_pair.d.r;
        // u_rgb[rayID * 3 + 1] = rgb_pair.d.g;
        // u_rgb[rayID * 3 + 2] = rgb_pair.d.b;
    }

    GroupMemoryBarrierWithGroupSync();
    float3.Differential rgb_out_grad = float3.Differential(
        shared_buffer[ray_index][0],
        shared_buffer[ray_index][1],
        shared_buffer[ray_index][2]
    );

    float3.Differential rgb_grad = float3.Differential(weight * rgb_out_grad);
    float.Differential weight_grad = dot(rgb, rgb_out_grad) - dot(rgb_out_grad, float3(1));
    // float.Differential sigma_grad = 
    float4.Differential rgb_sigma_grad = float4.Differential(rgb_grad, 0);
        
    var rgb_sigma_pair = diffPair(rgb_sigma_raw);
    bwd_diff(process_rgb_sigma)(rgb_sigma_pair, rgb_sigma_grad);
        
    rgb_grad = weight_grad;    
    const float3 debug_sum = WaveActiveSum(float3(rgb_grad));
    if (sample_index < 3) shared_buffer[ray_index][sample_index] = debug_sum[sample_index];
    GroupMemoryBarrierWithGroupSync();
    if (sample_index == 63) {
        const float3 debugSum_0 = float3(shared_buffer[ray_index][0],
                                         shared_buffer[ray_index][1],
                                         shared_buffer[ray_index][2]);
        const float3 totalDebugSum = debug_sum + debugSum_0;
            
        u_rgb[rayID * 3 + 0] = totalDebugSum.r;
        u_rgb[rayID * 3 + 1] = totalDebugSum.g;
        u_rgb[rayID * 3 + 2] = totalDebugSum.b;
    }

    // bwd_diff(LoadRGBSigma)(voxel_index, rgb_sigma_pair.d);
    
    // u_index[globalID] = weight;

    // u_rgb[globalID * 3 + 0] = rgb.r;
    // u_rgb[globalID * 3 + 1] = rgb.g;
    // u_rgb[globalID * 3 + 2] = rgb.b;
    // u_sigma[globalID] = sigma;
}