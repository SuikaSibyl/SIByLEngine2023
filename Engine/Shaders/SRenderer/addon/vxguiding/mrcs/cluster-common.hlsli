#ifndef _SRENDERER_VXGUIDING_MCRS_CLUSTER_COMMON_HEADER_
#define _SRENDERER_VXGUIDING_MCRS_CLUSTER_COMMON_HEADER_

#ifndef DESC_TYPE
#define DESC_TYPE uint4
#endif

// #if DESC_TYPE == uint4
struct svoxel_info {
    DESC_TYPE desc_info;// descriptor info
    float3 center;      // center (avg pixel pos) of the supervoxel
    float intensity;    // number of voxels in the supervoxel
    __init(DESC_TYPE desc, float3 cent, float intens) {
        desc_info = desc;
        center = cent;
        intensity = intens;
    }
};
// #endif

float ComputeLength(in DESC_TYPE vec) {
    return countbits(vec.x) + countbits(vec.y) + countbits(vec.z) + countbits(vec.w);
}

float ComputeDistance(in DESC_TYPE a, in DESC_TYPE b, bool extra = false,
                      float3 a_pos = float3(0), float3 b_pos = float3(0),
                      float a_intensity = 0.f, float b_intensity = 0.f,
                      float position_weight = 0.0f,
                      float intensity_weight = 0.01f) {
    const uint4 diff = a ^ b;
    const float diff_length = ComputeLength(diff);
    if (extra) {
        return diff_length + position_weight * distance(a_pos, b_pos) 
            + intensity_weight * distance(a_intensity, b_intensity);
    } else {
        return diff_length;
    }
}

#endif // _SRENDERER_VXGUIDING_MCRS_CLUSTER_COMMON_HEADER_