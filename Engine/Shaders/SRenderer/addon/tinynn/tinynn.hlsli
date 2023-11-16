#ifndef _SRENDERER_ADDON_TINYNN_COMMON_HLSLI_HEADER_
#define _SRENDERER_ADDON_TINYNN_COMMON_HLSLI_HEADER_

RWStructuredBuffer<float> _primal_buffer;
RWByteAddressBuffer _gradient_buffer;

struct ThreadInfo {
    int2 thread_idx;
    int2 block_dim;
    // constructor
    __init() { // default constructor
        thread_idx = int2(0, 0);
        block_dim = int2(0, 0); }
    __init(int2 thread_idx, int2 block_dim) {
        this.thread_idx = thread_idx;
        this.block_dim = block_dim; }
};

struct TensorView {
    uint offset_prim;
    uint offset_grad;
    uint stride;
    uint pitch;

    [BackwardDerivative(load_prim_idx1_bwd)]
    float load_prim(int x) { return _primal_buffer[offset_prim + x]; }
    [BackwardDerivative(load_prim_idx2_bwd)]
    float load_prim(int x, int y) { return load_prim(x + y * stride); }
    [BackwardDerivative(load_prim_idx3_bwd)]
    float load_prim(int x, int y, int z) { return load_prim(x + y * stride + z * pitch); }
    
    void interlocked_add_grad(int x, float val) { _gradient_buffer.InterlockedAddF32((offset_grad + x) * sizeof(float), val); }
    void interlocked_add_grad(int x, int y, float val) { interlocked_add_grad(x + y * stride, val); }
    void interlocked_add_grad(int x, int y, int z, float val) { interlocked_add_grad(x + y * stride + z * pitch, val); }

    void load_prim_idx1_bwd(int x, float.Differential val) { interlocked_add_grad(x, val); }
    void load_prim_idx2_bwd(int x, int y, float.Differential val) { interlocked_add_grad(x, y, val); }
    void load_prim_idx3_bwd(int x, int y, int z, float.Differential val) { interlocked_add_grad(x, y, z, val); }
    
    int indexing(int x, int y, int z) {
        return offset_prim + x + y * stride + z * pitch;
    }

    // constructor
    __init() { // default constructor
        offset_prim = 0;
        offset_grad = 0;
        stride = 0;
        pitch = 0; }
    __init(uint offset_prim, uint offset_grad, uint stride, uint pitch
    ) { this.offset_prim = offset_prim;
        this.offset_grad = offset_grad;
        this.stride = stride;
        this.pitch = pitch; }
};

#endif // !_SRENDERER_ADDON_TINYNN_COMMON_HLSLI_HEADER_