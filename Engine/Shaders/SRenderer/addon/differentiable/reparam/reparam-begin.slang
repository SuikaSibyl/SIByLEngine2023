[[vk::push_constant]]
cbuffer PushConstants {
    float learning_rate;
    bool initialize;
    float init_value_x;
    float init_value_y;
    float init_value_z;
};

RWByteAddressBuffer u_buffer;

[shader("compute")]
[numthreads(16, 1, 1)]
void ComputeMain(int3 dtid: SV_DispatchThreadID) {
    if (dtid.x == 0) {
        if (initialize) {
            u_buffer.Store<float3>(0, float3(init_value_x, init_value_y, init_value_z));
            u_buffer.Store<float3>(16, float3(0.0f));
        } else {
            float3 y = u_buffer.Load<float3>(0);
            float3 gradient = u_buffer.Load<float3>(16);
            y = y - learning_rate * gradient;
            u_buffer.Store<float3>(0, y);
            u_buffer.Store<float3>(16, float3(0.0f));
        }
    }
}