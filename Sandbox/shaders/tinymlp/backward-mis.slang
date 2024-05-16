#include "tinynn-half-mlp.hlsli"
#include "tinynn-featuregrid.hlsli"

StructuredBuffer<float> grad_texture;
const StructuredBuffer<int> index_array;

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
    int3 gctid: SV_GroupID,
    int gidx: SV_GroupIndex
) {
    const int globalIDX = gctid.x * 128 + gidx;
    const int indexIDX = index_array[globalIDX];
    const bool invalidIDX = indexIDX == -1;
    int2 pixelIDX = invalidIDX ? int2(0, 0) : int2(indexIDX >> 16, indexIDX & 0xFFFF);
    
    const ThreadInfo thread_info = ThreadInfo(gtid.xy, int2(32, 4));
        
    uint param_offset = 0; uint grad_offset = 0;
    MLPModule mlp = MLPModule(param_offset, grad_offset, thread_info);
    TensorView featureGrid = TensorView(param_offset, grad_offset, 14, 14 * 33);

    FeatureGrid2DIndex feature_index = FeatureGrid2DIndex(resolution, pixelIDX, feature_grid);
    HalfFeature<32> feature = computeInterpolatedFeature(featureGrid, feature_index);
    HalfFeature<32> output = MLPModule.forward(mlp, feature);
    
    HalfFeature<32>.Differential output_grad;
    for (int i = 0; i < 32; i++) {
        output_grad.vals[i] = float16_t(0.0f);
    }
    
    const float grad_x = grad_texture[globalIDX * 3 + 0];
    const float grad_y = grad_texture[globalIDX * 3 + 1];
    const float grad_z = grad_texture[globalIDX * 3 + 2];
    
    const float gradient_scalar = gradient_scale / (512 * 512 * 3);
    output_grad.vals[0] = float16_t(grad_x * gradient_scalar);
    output_grad.vals[1] = float16_t(grad_y * gradient_scalar);
    output_grad.vals[2] = float16_t(grad_z * gradient_scalar);
    
    var input_feature_pair = diffPair(feature);
    
    float scalar = 1.f;
    if (invalidIDX) {
        scalar = 0.f;
    } else {
        scalar = ((512.f*512.f)/(528*528)) / 4;
    }
    
    bwd_diff(MLPModule.forward)(mlp, input_feature_pair, scalar, output_grad);
    // bwd_diff(computeInterpolatedFeature)(featureGrid, feature_index, input_feature_pair.d);

    int2 cellSampled = int2((globalIDX % 528) / 16, (globalIDX / 528) / 16);
    if (invalidIDX) pixelIDX = int2(-1, -1);

    bwd_diff(computeInterpolatedFeature_MIS)(featureGrid, feature_index, pixelIDX, input_feature_pair.d);
}