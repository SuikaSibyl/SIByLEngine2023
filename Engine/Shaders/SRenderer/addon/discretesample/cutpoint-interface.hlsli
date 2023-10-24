#include "../../include/common/cpp_compatible.hlsli"

/**
 * Sample the discrete distribution using the cutpoint method.
 * @param n_partitions The number of partitions in the distribution.
 */
int cutpoint_sampling(
    in_ref(uint) n_partitions,
    in_ref(float) random_sample,
    in_ref(StructuredBuffer<int>) prefix_sum,
    in_ref(StructuredBuffer<int>) cutpoint_table,
) {
    const int g = int(floor(random_sample * n_partitions));
    const int j = cutpoint_table[g];

    
}