#ifndef _SRENDERER_ADDON_HALF_TINYNN_MLP_HLSLI_HEADER_
#define _SRENDERER_ADDON_HALF_TINYNN_MLP_HLSLI_HEADER_

#include "tinynn-activations.hlsli"
#include "tinynn-half-linear.hlsli"

struct MLPHalf<let N : int, let C : int> {
    typedef HalfFeature<C> Input;
    typedef HalfFeature<C> Output;
};

struct MLPHalf16X16<let N:int, Act:IActivationFn> : MLPHalf<N, 16> {
    LinearHalf16X16 linears[N];

    __init(inout uint offset_prim, inout uint offset_grad, ThreadInfo threadInfo) {
        [ForceUnroll] for (int i = 0; i < N; i++)
        linears[i] = LinearHalf16X16(offset_prim, offset_grad, threadInfo); }

    [Differentiable]
    Output _foward(Input input) {
        HalfFeature<16> out_feature = input;
        [ForceUnroll] for (int i = 0; i < N; i++) {
            out_feature = LinearHalf16X16.eval(linears[i], out_feature);
            [ForceUnroll] for (int j = 0; j < 16; j++)
            out_feature.vals[j] = Act.eval(out_feature.vals[j]); }
        return out_feature; }

    [Differentiable]
    static Output forward(no_diff MLPHalf16X16<N, Act> mlp, Input input) {
        return mlp._foward(input); }
};

struct MLPHalf32X32<let N:int, Act:IActivationFn> : MLPHalf<N, 32> {
    LinearHalf32X32 linears[N];
    
    __init(inout uint offset_prim, inout uint offset_grad, ThreadInfo threadInfo) {
        [ForceUnroll] for (int i = 0; i < N; i++)
        linears[i] = LinearHalf32X32(offset_prim, offset_grad, threadInfo); }

    [Differentiable]
    Output _foward(Input input) {
        HalfFeature<32> out_feature = input;
        [ForceUnroll] for (int i = 0; i < N; i++) {
            out_feature = LinearHalf32X32.eval(linears[i], out_feature);
            [ForceUnroll] for (int j = 0; j < 32; j++)
            out_feature.vals[j] = Act.eval(out_feature.vals[j]); }
        return out_feature; }
    
    [Differentiable]
    static Output forward(no_diff MLPHalf32X32<N, Act> mlp, Input input) {
        return mlp._foward(input); }
};

#endif // !_SRENDERER_ADDON_HALF_TINYNN_LINEAR_HLSLI_HEADER_