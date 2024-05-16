#include "tinynn-half-mlp.hlsli"
#include "tinynn-featuregrid.hlsli"

RWStructuredBuffer<float4> output_texture;
RWStructuredBuffer<float> error_texture;
Sampler2D<float4> target_texture;

RWStructuredBuffer<float> output_buffer;

[[vk::push_constant]]
cbuffer PushConstants {
    int2 target_resolution;
    float gradient_scale;
};

static const int2 resolution = int2(512, 512);
static const int2 feature_grid = int2(32, 32);

[Differentiable]
float L2Loss(float3 x, no_diff float3 y) {
    return dot((x - y), (x - y));
}

typedef MLPHalf32X32<3, ReLU> MLPModule;

[shader("compute")]
[numthreads(32, 4, 1)]
void ComputeMain(
    int3 dtid: SV_DispatchThreadID,
    int3 gtid: SV_GroupThreadID,
    int3 gctid: SV_GroupID
) {
    const int2 pixel = dtid.xy;
    const float2 uv = (pixel + 0.5f) / resolution;
    const ThreadInfo thread_info = ThreadInfo(gtid.xy, int2(32, 4));
    
    const int layer_stride = 16 * 16 + 16;
    
    HalfFeature<32>.Differential output_grad;
    for (int i = 0; i < 32; i++) {
        output_grad.vals[i] = float16_t(0.0f);
    }
    
    uint param_offset = 0; uint grad_offset = 0;
    MLPModule mlp = MLPModule(param_offset, grad_offset, thread_info);
    TensorView featureGrid = TensorView(param_offset, grad_offset, 14, 14 * 33);

    FeatureGrid2DIndex feature_index = FeatureGrid2DIndex(resolution, pixel, feature_grid);
    HalfFeature<32> feature = computeInterpolatedFeature(featureGrid, feature_index);
    HalfFeature<32> output = MLPModule.forward(mlp, feature);
    
    float3 target_color = target_texture.Load(int3(pixel, 0)).xyz;

    float3 color = float3(output.vals[0], output.vals[1], output.vals[2]);
    
    var color_pair = diffPair(color);
    float loss = L2Loss(color, target_color);
    bwd_diff(L2Loss)(color_pair, target_color, 1);
    const float gradient_scalar = gradient_scale / (512 * 512 * 3);
    output_grad.vals[0] = float16_t(color_pair.d.x * gradient_scalar);
    output_grad.vals[1] = float16_t(color_pair.d.y * gradient_scalar);
    output_grad.vals[2] = float16_t(color_pair.d.z * gradient_scalar);

    float4 debug_color = float4(feature_index.weights, feature_index.cellId);
    for (int i = 0; i < 32; i++) {
        if (isnan(output.vals[i])) {
            debug_color = float4(1.0f, 0.0f, 1.0f, 1.0f);
            // output.vals[i] = float16_t(0.0f);
        } else if (isinf(output.vals[i])) {
            debug_color = float4(0.0f, 1.0f, 1.0f, 1.0f);
        }
    }
    
    error_texture[dtid.x + dtid.y * 512] = loss / (512 * 512 * 3);
    output_texture[dtid.x + dtid.y * 512] = float4(color, 1.0f);
    
    var input_feature_pair = diffPair(feature);
    
    bwd_diff(MLPModule.forward)(mlp, input_feature_pair, output_grad);
    bwd_diff(computeInterpolatedFeature)(featureGrid, feature_index, input_feature_pair.d);
}