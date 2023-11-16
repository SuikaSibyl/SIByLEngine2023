#ifndef _SRENDERER_ADDON_DIFFERENTIABLE_GRADIENT_UTILS_HEADER_
#define _SRENDERER_ADDON_DIFFERENTIABLE_GRADIENT_UTILS_HEADER_

#include "../../../include/diff_descriptor_set.hlsli"

/**
 * Differentiable IO wrapper for global memory access.
 * As slang AD doesn't support global memory AD, for BP,
 * we need to wrap all global memory access into this struct.
 */
struct DiffIOWrapper {
    uint base_offset;   // base offset of the gradient buffer
    uint hash_idx;      // hash index of the gradient buffer
    /** initialize the wrapper */
    __init(uint _base_offset, uint _hash_idx) {
        this.base_offset = _base_offset;
        this.hash_idx = _hash_idx;
    }
    /**
     * Get the gradient of a float value.
     * @param val: the value loaded from global memory
     * @param offset: the offset of the value since the base offset
     */
    [ForwardDerivative(__fwd_d_getFloat)]
    [BackwardDerivative(__bwd_d_getFloat)]
    float getFloat(float val, uint offset) { return val; }
    /** forward derivative, return the differential pair
     * should propagate the differential pair (?) */
    [TreatAsDifferentiable]
    DifferentialPair<float> __fwd_d_getFloat(DifferentialPair<float> dpVal, uint offset) {
        return dpVal; // should propagate the differential pair (?)
    }
    /** backward derivative, return the differential pair
     * accumulate the gradient to the gradient buffer */
    [TreatAsDifferentiable]
    void __bwd_d_getFloat(inout DifferentialPair<float> dpVal, uint offset, float dOut) {
        offset += base_offset;
        gSceneGradients.atomicAddGrad(offset, hash_idx, dOut);
    }
    
    [ForwardDerivative(__fwd_d_getFloats)]
    [BackwardDerivative(__bwd_d_getFloats)]
    __generic<let N : int> vector<float, N> getFloat(vector<float, N> val, uint offset) { return val; }

    [TreatAsDifferentiable]
    __generic<let N : int> DifferentialPair<vector<float, N>> __fwd_d_getFloats(DifferentialPair<vector<float, N>> dpVal, uint offset)
    {
        vector<float, N> d;
        [ForceUnroll]
        for (int i = 0; i < N; i++)
            d[i] = 0.f;
        
#if DIFF_MODE == 2 // ForwardDiffDebug
        if (gDiffDebug.varType == DiffVariableType::Material && base_offset == gDiffDebug.id.x * 20 && offset == gDiffDebug.offset)
        {
            [ForceUnroll]
            for (int i = 0; i < N; i++)
                d[i] = gDiffDebug.grad[i];
        }
#endif

        return diffPair(dpVal.p, d);
    }

    [TreatAsDifferentiable]
    __generic<let N : int> void __bwd_d_getFloats(inout DifferentialPair<vector<float, N>> dpVal, uint offset, vector<float, N> dOut)
    {
        offset += base_offset;
        [ForceUnroll]
        for (uint i = 0; i < N; i++)
            gSceneGradients.atomicAddGrad(gradType, offset + i, hash_idx, dOut[i]);
    }

    [Differentiable]
    float getFloat(float val, const MaterialParamLayoutEntry entry) { return getFloat(val, entry.offset); }

    [Differentiable]
    __generic<let N : int> vector<float, N> getFloat(vector<float, N> val, const MaterialParamLayoutEntry entry)
    {
        return getFloat(val, entry.offset);
    }
};

#endif // _SRENDERER_ADDON_DIFFERENTIABLE_GRADIENT_UTILS_HEADER_