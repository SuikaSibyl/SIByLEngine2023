#include "tiny-mpl.hlsli"

RWTexture2D<float4> output_texture;
RWTexture2D<float> error_texture;
// Sampler2D<float4> target_texture; 

// RWStructuredBuffer<float> input_buffer;
RWStructuredBuffer<float> output_buffer;

[[vk::push_constant]]
cbuffer PushConstants {
    int2 target_resolution;
};

static const int2 resolution = int2(128, 128);

[Differentiable]
float L2Loss(float3 x, no_diff float3 y) {
    return dot((x - y), (x - y));
}

[Differentiable]
float relu(float x) {
    return max(0.0f, x);
}

[Differentiable]
float leaky_relu(float x, float alpha = 0.01f) {
    return max(0.0f, x) + 0.1f * min(0.0f, x);
}

[Differentiable]
Feature<16> mlp(
    Linear<16> linear[3],
    Feature<16> feature,
) {
    Feature<16> out_feature = feature;
    for (int i = 0; i < 3; ++i) {
        out_feature = eval(linear[i], out_feature);
        for (int j = 0; j < 16; j++) {
            out_feature.vals[j] = relu(out_feature.vals[j]);
        }
    }
    return out_feature;
}

[BackwardDerivative(InterpolatedFeature_bwd)]
Feature<16> computeInterpolatedFeature(no_diff TensorView featureGrid, no_diff uint2 frameDim, no_diff int2 pixelId) {
    // Compute pixel block covered by each grid cell.
    uint dim0 = 32 - 1;
    uint dim1 = 32 - 1;
    float2 cellSize = float2(frameDim.x * 1.f / dim0, frameDim.y * 1.f / dim1);
    uint2 cellId = uint2(floor(pixelId / cellSize));
    // Compute the interpolation weights.
    float2 weights = (float2(pixelId) / cellSize) - float2(cellId);
    // Compute the interpolated feature.
    Feature<16> feature;
    [ForceUnroll]
    for (int i = 0; i < 16 - 2; i++) {
        feature.vals[i] = featureGrid.load_prim(cellId.x, cellId.y, i) * (1 - weights.x) * (1 - weights.y) +
                          featureGrid.load_prim(cellId.x + 1, cellId.y, i) * weights.x * (1 - weights.y) +
                          featureGrid.load_prim(cellId.x, cellId.y + 1, i) * (1 - weights.x) * weights.y +
                          featureGrid.load_prim(cellId.x + 1, cellId.y + 1, i) * weights.x * weights.y;
    }
    feature.vals[16 - 2] = weights.x;
    feature.vals[16 - 1] = weights.y;
    return feature;
}

void InterpolatedFeature_bwd(no_diff TensorView featureGrid, no_diff uint2 frameDim, no_diff int2 pixelId, Feature<16>.Differential output_grad) {
    // Compute pixel block covered by each grid cell.
    uint dim0 = 32 - 1;
    uint dim1 = 32 - 1;
    float2 cellSize = float2(frameDim.x * 1.f / dim0, frameDim.y * 1.f / dim1);
    uint2 cellId = uint2(floor(pixelId / cellSize));
    // Compute the interpolation weights.
    float2 weights = (float2(pixelId) / cellSize) - float2(cellId);
    // Compute the interpolated feature.
    [ForceUnroll]
    for (int i = 0; i < 16 - 2; i++) {
        featureGrid.interlocked_add_grad(cellId.x, cellId.y, i, output_grad.vals[i] * (1 - weights.x) * (1 - weights.y));
        featureGrid.interlocked_add_grad(cellId.x + 1, cellId.y, i, output_grad.vals[i] * weights.x * (1 - weights.y));
        featureGrid.interlocked_add_grad(cellId.x, cellId.y + 1, i, output_grad.vals[i] * (1 - weights.x) * weights.y);
        featureGrid.interlocked_add_grad(cellId.x + 1, cellId.y + 1, i, output_grad.vals[i] * weights.x * weights.y);
    }
}

[shader("compute")]
[numthreads(32, 4, 1)]
void ComputeMain(
    int3 dtid: SV_DispatchThreadID,
    int3 gtid: SV_GroupThreadID,
    int3 gctid: SV_GroupID
) {
    float2 uv = (dtid.xy + 0.5f) / resolution;
    int2 pixel = dtid.xy;

    ThreadInfo thread_info = ThreadInfo(gtid.xy, int2(32, 4));
    
    const int layer_stride = 16 * 16 + 16;

    TensorView weights[3] = {
        TensorView(0 * layer_stride, 0 * layer_stride, 16, 16),
        TensorView(1 * layer_stride, 1 * layer_stride, 16, 16),
        TensorView(2 * layer_stride, 2 * layer_stride, 16, 16),
    };
    TensorView bias[3] = {
        TensorView(0 * layer_stride + 16 * 16, 0 * layer_stride + 16 * 16, 16, 16),
        TensorView(1 * layer_stride + 16 * 16, 1 * layer_stride + 16 * 16, 16, 16),
        TensorView(2 * layer_stride + 16 * 16, 2 * layer_stride + 16 * 16, 16, 16),
    };
    
    // TensorView weights = TensorView(param_buffer, param_diff_buffer, 0 * layer_stride, 0 * layer_stride, 16);
    // TensorView bias = TensorView(param_buffer, param_diff_buffer, 0 * layer_stride + 16 * 16, 0 * layer_stride + 16 * 16, 16);
    
    Feature<16>.Differential output_grad;
    for (int i = 0; i < 16; i++) {
        output_grad.vals[i] = 0.0f;
    }

    TensorView featureGrid = TensorView(3 * layer_stride, 3 * layer_stride, 32, 32 * 32);

    Linear<16> linear[3] = {
        Linear<16>(weights[0], bias[0], thread_info),
        Linear<16>(weights[1], bias[1], thread_info),
        Linear<16>(weights[2], bias[2], thread_info),
    };

    Feature<16> feature = computeInterpolatedFeature(featureGrid, resolution, pixel);
    feature.vals[16 - 2] = uv.x;
    feature.vals[16 - 1] = uv.y;
    Feature<16> out_feature = mlp(linear, feature);

    int2 target_pixel = int2(uv * target_resolution);
    // float3 target_color = target_texture.Sample(uv).xyz;
    float3 target_color = float3(uv, 1.0f);
    
    float3 color = float3(out_feature.vals[0], out_feature.vals[1], out_feature.vals[2]);
    
    var color_pair = diffPair(color);
    float loss = L2Loss(color, target_color);
    bwd_diff(L2Loss)(color_pair, target_color, 1);
    output_grad.vals[0] = color_pair.d.x;
    output_grad.vals[1] = color_pair.d.y;
    output_grad.vals[2] = color_pair.d.z;
    
    output_texture[dtid.xy] = float4(color, 1.0f);
    error_texture[dtid.xy] = loss;
    
    var feature_pair = diffPair(feature);
    bwd_diff(mlp)(linear, feature_pair, output_grad);
    bwd_diff(computeInterpolatedFeature)(featureGrid, resolution, pixel, feature_pair.d);
}