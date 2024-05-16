RWStructuredBuffer<float> param_buffer;
RWStructuredBuffer<float> param_diff_buffer;

[[vk::push_constant]]
cbuffer PushConstants {
    int random_seed;
    bool reweight;
    float learning_rate;
};

#include "../../include/common/random.hlsli"
#include "../../include/common/gaussian.hlsli"

float gaussian(inout RandomSamplerState RNG) {
    return BoxMuller(GetNextRandomFloat2(RNG)).x;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void ComputeMain(
    int3 dtid: SV_DispatchThreadID,
    int3 gtid: SV_GroupThreadID,
    int3 gctid: SV_GroupID
) {
    // For each layer, we have 16 * 16 + 16 parameters
    // in total we get 3 layers.
    uint param_count = (16 * 16 + 16) * 3;
    uint param_stride = 16 * 16 + 16;
    const uint index = dtid.x;
    if(index >= param_count) return;

    if (!reweight) {
        param_buffer[index] -= learning_rate * param_diff_buffer[index] / (128 * 128);
        param_diff_buffer[index] = 0.f;
        return;
    }
    
    RandomSamplerState RNG = InitRandomSampler(dtid.x, random_seed);

    if(index % param_stride < 16 * 16) { // is a weight
        param_buffer[index] = gaussian(RNG) * 0.01;
    } else { // is a bias
        param_buffer[index] = 0.f;
    }
    // Also clear the diff buffer
    param_diff_buffer[index] = 0.f;
}