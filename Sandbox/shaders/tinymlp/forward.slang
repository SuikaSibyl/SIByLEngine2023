#include "tinynn-half-mlp.hlsli"
#include "tinynn-featuregrid.hlsli"

RWStructuredBuffer<float> output_texture;

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
    const ThreadInfo thread_info = ThreadInfo(gtid.xy, int2(32, 4));
    
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
    
    float3 color = float3(output.vals[0], output.vals[1], output.vals[2]);
    output_texture[(dtid.x + dtid.y * 512) * 3 + 0] = output.vals[0];
    output_texture[(dtid.x + dtid.y * 512) * 3 + 1] = output.vals[1];
    output_texture[(dtid.x + dtid.y * 512) * 3 + 2] = output.vals[2];
}