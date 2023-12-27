#ifndef _SRENDERER_ADDON_RADIXFOREST_SAMPLING_HLSLI_HEADER_
#define _SRENDERER_ADDON_RADIXFOREST_SAMPLING_HLSLI_HEADER_

struct RadixTreeNode {
    int child[2];
};

int sample_radix_forset(
    float xi,
    int partition_num,
    StructuredBuffer<int> forset_table,
    StructuredBuffer<float> pmf_list,
    StructuredBuffer<RadixTreeNode> nodes
) {
    const float g = floor(xi * partition_num);
    int j = forset_table[int(g)];
    while (firstbithigh(j) != 1) {
        j = (xi < pmf_list[j]) 
            ? nodes[j].child[0] 
            : nodes[j].child[1];
    }
    return ~j;
}

#endif // !_SRENDERER_ADDON_RADIXFOREST_SAMPLING_HLSLI_HEADER_