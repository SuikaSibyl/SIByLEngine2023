// -------------- input --------------
[[vk::push_constant]]
cbuffer PushConstants {
    int2 size;
    uint objective; // 0: L2
};
const Texture2D<float4> u_input;        // input image
const Sampler2D<float4> u_reference;    // reference image
// -------------- output --------------
RWTexture2D<float4> u_deltaY;        // delta y

// L2 loss
[Differentiable]
float L2Loss(float3 input, no_diff float3 reference) {
    float3 diff = input - reference;
    return dot(diff, diff);
}

// differentiate the L2 loss
float3 L2Diff(float3 input, float3 reference) {
    var input_pair = diffPair(input);
    bwd_diff(L2Loss)(input_pair, reference, 1.0);
    return input_pair.d;
}

/**
 * We differentiate the objective at y with AD.
 * Getting the gradient δy known as adjoint rendering
 */
[shader("compute")]
[numthreads(16, 16, 1)]
void ComputeMain(int3 dtid: SV_DispatchThreadID) {
    const int2 pixel = dtid.xy;
    if (any(pixel >= size)) return;
    const float3 input = u_input[pixel].xyz;
    const float3 reference = u_reference[pixel].xyz;
    float3 deltaY = float3(0, 0, 0);
    if (objective == 0) { // L2
        deltaY = L2Diff(input, reference);
    }
    u_deltaY[pixel] = float4(deltaY, 1);
}