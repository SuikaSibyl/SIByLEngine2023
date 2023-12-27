[[vk::push_constant]]
cbuffer PushConstants {
    int data_num;
};

RWStructuredBuffer<float> u_wg_aggregate;
RWStructuredBuffer<float> u_wg_prefixsum;
RWStructuredBuffer<float> u_prefix_sum;

[shader("compute")]
[numthreads(512, 1, 1)]
void ComputeMain(
    int3 DTid: SV_DispatchThreadID,
    int3 GTid: SV_GroupThreadID,
    int3 Gid: SV_GroupID,
) {
    const int global_id = DTid.x;
    const int workgroup_id = Gid.x;
    if (global_id >= data_num) return;

    if (GTid.x == 0) {
        u_wg_aggregate[workgroup_id] = asfloat(2139095039u);
        u_wg_prefixsum[workgroup_id] = asfloat(2139095039u);
    }

    u_prefix_sum[global_id] = 0.f;
}