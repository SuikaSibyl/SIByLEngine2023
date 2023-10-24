#include "../../../include/diff_descriptor_set.hlsli"

[[vk::push_constant]]
cbuffer PushConstants {
    uint param_size; // size in bytes
};

[shader("compute")]
[numthreads(256, 1, 1)]
void ComputeMain(int3 dtid: SV_DispatchThreadID) {
    if (any(dtid.x >= param_size)) return;
    SetGrad(dtid.x, 0.f);
}