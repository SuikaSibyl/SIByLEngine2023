[[vk::push_constant]]
cbuffer PushConstants {
    int data_num;
};

const StructuredBuffer<float> u_prefixsum;
RWStructuredBuffer<float> u_pmf;

[shader("compute")]
[numthreads(512, 1, 1)]
void ComputeMain(
    int3 DTid: SV_DispatchThreadID,
    int3 GTid: SV_GroupThreadID,
    int3 Gid: SV_GroupID,
) {
    const int global_id = DTid.x;
    if (global_id >= data_num) return;
    const float data = u_prefixsum[global_id];
    const float sum = u_prefixsum[data_num];
    const float pmf = data / sum;
    u_pmf[global_id] = pmf;
}