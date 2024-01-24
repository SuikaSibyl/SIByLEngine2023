#include "../../../include/diff_descriptor_set.hlsli"
#include "../../../include/common/data_type.hlsli"

[[vk::push_constant]]
cbuffer PushConstants { 
    int2 image_size;
    uint resource_id;
    float learning_rate;
};

RWTexture2D<DATA_TYPE> u_resource;

[shader("compute")]
[numthreads(16, 16, 1)]
void ComputeMain(int3 dtid: SV_DispatchThreadID) {
    const int2 pixel = dtid.xy;
    const DiffResourceDesc desc = DiffResourcesDescs[resource_id];
    const uint width = desc.data_extend & 0xFFFF;
    const uint data_mask = get_data_mask(desc);
    const int valid_component = countbits(data_mask);
    const int component = sizeof(DATA_TYPE) / sizeof(float);
    DATA_TYPE gradient = DATA_TYPE(0.0f);
    
    // Fetch the gradient of each differentiable component
    const float2 uv = (pixel + 0.5f) / image_size;
    const int mip_max = 1 + int(floor(log2(width)));
    // texcoord *tex_dimension
    for (int mip = 0; mip < mip_max; mip++) {
        const int mip_size = width >> mip;
        const float weight = 1.0f / (1 << (2 * mip));
        const int2 mip_pixel = int2(floor(uv * mip_size));
        int offset = desc.data_offset + (FlattensPixelToIndex(mip_pixel, mip_size)) * valid_component;
        offset += mipmap_offset_power2(width, mip) * valid_component;
        for (int i = 0; i < component; ++i) {
            const bool valid_channel = (((data_mask >> i) & 0b1) != 0);
            if (valid_channel) {
                const float comp_gradient = GetGrad(offset) * weight;
#if DATA_COMPONENT == 1 // if it is a single-component data
                gradient += comp_gradient;
#else // otherwise it is a multi-component data
                gradient[i] += comp_gradient;
#endif
                offset += 1;
            }
        }
    }
    
    DATA_TYPE loaded = u_resource[pixel];
    for (int i = 0; i < DATA_COMPONENT; ++i) {
        if (loaded[i] < 0) {
            loaded[i] = 0.01f;
        }
        if (loaded[i] > 1) {
            loaded[i] = 0.99f;
        }
    }
    gradient = clamp(gradient, -10, 10);
    u_resource[pixel] = loaded - gradient * learning_rate;
}