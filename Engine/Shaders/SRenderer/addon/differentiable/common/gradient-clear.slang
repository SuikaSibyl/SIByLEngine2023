#include "../../../include/diff_descriptor_set.hlsli"

[[vk::push_constant]]
cbuffer PushConstants {
    uint param_size; // size in bytes
};

// Error sum buffer
RWByteAddressBuffer u_error_sum;

[shader("compute")]
[numthreads(256, 1, 1)]
void ComputeMain(int3 dtid: SV_DispatchThreadID) {
    if (dtid.x == 0) u_error_sum.Store<float>(0, 0.f);
    if (any(dtid.x >= param_size)) return;
    // Clear the parameter gradients
    SetGrad(dtid.x, 0.f);
    // Also clear the error sum
}