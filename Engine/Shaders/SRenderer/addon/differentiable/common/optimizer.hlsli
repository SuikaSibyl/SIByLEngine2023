#ifndef _SRENDERER_ADDON_DIFFERENTIABLE_OPTIMIZER_HEADER_
#define _SRENDERER_ADDON_DIFFERENTIABLE_OPTIMIZER_HEADER_

enum OptimizerType {
    SGD = 0,
    ADAM = 1,
};

struct OptimizerParam {
    OptimizerType type;
    int t;  // iteration index
    float lr;       // |    SGD    |   Adam   |
    float param_0;  // |  momentum |  beta_1  |
    float param_1;  // | dampening |  beta_2  |
    float param_2;  // |           |  epsilon |
    float param_3;  // |
    float param_4;  // |
};

float sgd_optimizer(
    float theta,
    float df_dtheta,
    inout float b_t,
    in OptimizerParam constant,
) {
    const int t = constant.t;
    const float momentum = constant.param_0;
    const float dampening = constant.param_1;
    float g_t = df_dtheta;
    b_t = (momentum != 0 && t > 1)
              ? momentum * b_t + (1 - dampening) * g_t
              : g_t;
    g_t = b_t;
    return theta - constant.lr * g_t;
}

float adam_optimizer(
    float theta,
    float g_t,
    inout float m_t,
    inout float v_t,
    in OptimizerParam constant,
) {
    const int t = constant.t + 1;
    const float beta_1 = constant.param_0;
    const float beta_2 = constant.param_1;
    const float epsilon = constant.param_2;
    m_t = beta_1 * m_t + (1 - beta_1) * g_t;
    v_t = beta_2 * v_t + (1 - beta_2) * g_t * g_t;
    const float m_t_hat = m_t / (1 - pow(beta_1, t));
    const float v_t_hat = v_t / (1 - pow(beta_2, t));
    const float theta_new = theta - constant.lr * m_t_hat / (sqrt(v_t_hat) + epsilon);
    return theta_new;
}

#endif // _SRENDERER_ADDON_DIFFERENTIABLE_OPTIMIZER_HEADER_