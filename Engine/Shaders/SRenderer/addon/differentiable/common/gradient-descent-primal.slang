#include "optimizer.hlsli"

RWStructuredBuffer<float> u_parameter_buffer;
RWStructuredBuffer<float> u_gradient_buffer;
RWStructuredBuffer<float> u_auxiliary_buffer;

[[vk::push_constant]]
cbuffer PushConstants {
    OptimizerParam optimizer;
    int param_size;
    int grad_offset;
    int step;
};

[shader("compute")]
[numthreads(256, 1, 1)]
void ComputeMain(int3 dtid: SV_DispatchThreadID) {
    const int tid = dtid.x;
    if (tid >= param_size) return;

    const float theta = u_parameter_buffer[tid];
    const float df_dtheta = u_gradient_buffer[grad_offset + tid];

    if (optimizer.type == OptimizerType::SGD) {
        float b_t = u_auxiliary_buffer[grad_offset + tid];
        if (step == 0) b_t = 0.f;
        const float theta_new = sgd_optimizer(theta, df_dtheta, b_t, optimizer);
        u_auxiliary_buffer[grad_offset + tid] = b_t;
        u_parameter_buffer[tid] = theta_new;
    } else if (optimizer.type == OptimizerType::ADAM) {
        float m_t = u_auxiliary_buffer[2 * (grad_offset + tid) + 0];
        float v_t = u_auxiliary_buffer[2 * (grad_offset + tid) + 1];
        if (step == 0) { m_t = 0; v_t = 0; }
        const float theta_new = adam_optimizer(theta, df_dtheta, m_t, v_t, optimizer);
        u_auxiliary_buffer[2 * (grad_offset + tid) + 0] = m_t;
        u_auxiliary_buffer[2 * (grad_offset + tid) + 1] = v_t;
        u_parameter_buffer[tid] = theta_new;
    }
}