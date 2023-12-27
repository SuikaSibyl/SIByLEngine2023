[[vk::push_constant]]
cbuffer PushConstants {
    int data_num;
};

StructuredBuffer<float> u_data;
volatile RWStructuredBuffer<float> u_wg_aggregate;
volatile RWStructuredBuffer<float> u_wg_prefixsum;
RWStructuredBuffer<float> u_prefix_sum;

groupshared float s_subgroup_prefixsum[33];

float DecoupledLookBack(int thread_id, int workgroup_id, float workgroup_aggregate) {
    float exclusive_prefix = 0;
    // Do decoupled Look-back
    for (int precursor = workgroup_id - 1; precursor >= 0; precursor--) {
        const float precursor_prefix = u_wg_prefixsum[precursor];
        if (precursor_prefix <= 0) {
            exclusive_prefix += -precursor_prefix;
            break;
        } else {
            // stall until aggregate is prepared
            float precursor_aggregate = u_wg_aggregate[precursor];
            while (precursor_aggregate > 0) { precursor_aggregate = u_wg_aggregate[precursor]; }
            exclusive_prefix += -precursor_aggregate;
        }
    }
    u_wg_prefixsum[workgroup_id] = -(exclusive_prefix + workgroup_aggregate);
    return exclusive_prefix;
}

[shader("compute")]
[numthreads(1024, 1, 1)]
void ComputeMain(
    int3 DTid: SV_DispatchThreadID,
    int3 GTid: SV_GroupThreadID,
    int3 Gid: SV_GroupID,
) {
    const int global_id = DTid.x;
    const int thread_id = GTid.x;
    const int workgroup_id = Gid.x;
    const int subgroup_id = thread_id / 32;

    const float data = (global_id < data_num) ? u_data[global_id] : 0;
    const float subgroup_prefixsum = WavePrefixSum(data);
    if (thread_id % 32 == 31) {
        const float subgroup_aggregate = subgroup_prefixsum + data;
        s_subgroup_prefixsum[subgroup_id] = subgroup_aggregate;
    }

    GroupMemoryBarrierWithGroupSync();

    if (subgroup_id == 0) {
        const float data = s_subgroup_prefixsum[thread_id];
        const float subgroup_prefixsum = WavePrefixSum(data);
        s_subgroup_prefixsum[thread_id] = subgroup_prefixsum;
        if (thread_id == 31) {
            s_subgroup_prefixsum[32] = subgroup_prefixsum + data;
        }
    }

    if (thread_id == 0) {
        const float workgroup_aggregate = s_subgroup_prefixsum[32];
        u_wg_aggregate[workgroup_id] = -workgroup_aggregate;
        const float exclusive_prefix = DecoupledLookBack(thread_id, workgroup_id, workgroup_aggregate);
        s_subgroup_prefixsum[32] = exclusive_prefix;
    }

    GroupMemoryBarrierWithGroupSync();

    const float total_prefix_sum = subgroup_prefixsum
        + s_subgroup_prefixsum[subgroup_id]
        + s_subgroup_prefixsum[32];
        
    if (global_id < data_num)
        u_prefix_sum[global_id] = total_prefix_sum;
    
    if (global_id == data_num - 1)
        u_prefix_sum[data_num] = total_prefix_sum + data;
}