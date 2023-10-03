#ifndef _SRENDERER_VXGUIDING_MCRS_CLUSTER_COMMON_HEADER_
#define _SRENDERER_VXGUIDING_MCRS_CLUSTER_COMMON_HEADER_

#define DESC_TYPE uint4

#if DESC_TYPE == uint4
struct svoxel_info {
    DESC_TYPE desc_info;// descriptor info
    float3 center;      // center (avg pixel pos) of the supervoxel
    int no_voxels;      // number of voxels in the supervoxel
    __init(DESC_TYPE desc, float3 cent, int num) {
        desc_info = desc;
        center = cent;
        no_voxels = num;
    }
};
#endif

// float ComputeDistance(
//     in DESC_TYPE a, 
//     in DESC_TYPE b,
//     in float3 a_center,
//     in float3 b_center,
//     float weight = 1.0f
// ) {
//     const uint4 diff = a ^ b;
//     const float dist_desc = countbits(diff.x) + countbits(diff.y)
//      + countbits(diff.z) + countbits(diff.w);
//     return dist_desc;
// }

float ComputeLength(in DESC_TYPE vec) {
    return countbits(vec.x) + countbits(vec.y) + countbits(vec.z) + countbits(vec.w);
}

float ComputeDistance(in DESC_TYPE a, in DESC_TYPE b, bool extra = false) {
    const uint4 diff = a ^ b;
    const float diff_length = ComputeLength(diff);
    if (extra) {
        return diff_length * diff_length * ComputeLength(a) * ComputeLength(b);
    } else {
        return diff_length;
    }
}

#endif // _SRENDERER_VXGUIDING_MCRS_CLUSTER_COMMON_HEADER_