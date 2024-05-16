#ifndef _SRENDERER_ADDON_TINYNN_COMMON_HLSLI_HEADER_
#define _SRENDERER_ADDON_TINYNN_COMMON_HLSLI_HEADER_

RWStructuredBuffer<float16_t> _primal_buffer;
RWByteAddressBuffer _gradient_buffer;
RWByteAddressBuffer _gradcell_buffer;

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
    float load_prim(int x, int y) { return load_prim(x * stride + y); }
    [BackwardDerivative(load_prim_idx3_bwd)]
    float load_prim(int x, int y, int z) { return load_prim(x * stride + y * pitch + z); }

    [BackwardDerivative(load_prim_gscalar_idx1_bwd)]
    float load_prim_gscalar(int x, no_diff float scalar = 1.f) { return _primal_buffer[offset_prim + x]; }
    [BackwardDerivative(load_prim_gscalar_idx2_bwd)]
    float load_prim_gscalar(int x, int y, no_diff float scalar = 1.f) { return load_prim(x * stride + y); }
    [BackwardDerivative(load_prim_gscalar_idx3_bwd)]
    float load_prim_gscalar(int x, int y, int z, no_diff float scalar = 1.f) { return load_prim(x * stride + y * pitch + z); }
    
    /**
     * load primal value with explicit gradient index
     * @param x primal index
     * @param grad_idx gradient index
     * when grad_idx >= 0, the gradient will be accumulated to the specified index,
     * when grad_idx == -1, the gradient will be accumulated to the primal index,
     * when grad_idx < 0, the gradient will be ignored.
     */
    [BackwardDerivative(load_prim_grad_id_idx1_bwd)]
    float load_prim_grad_id(int x, int grad_idx = -1) { return _primal_buffer[offset_prim + x]; }
    [BackwardDerivative(load_prim_grad_id_idx2_bwd)]
    float load_prim_grad_id(int x, int y, int grad_idx = -1) { return load_prim(x * stride + y); }
    [BackwardDerivative(load_prim_grad_id_idx3_bwd)]
    float load_prim_grad_id(int x, int y, int z, int grad_idx = -1) { return load_prim(x * stride + y * pitch + z); }

    [BackwardDerivative(load_prim_gradid_idx1_bwd)]
    float load_prim_gradid(int x, int grad_idx = -1) { return _primal_buffer[offset_prim + x]; }
    [BackwardDerivative(load_prim_gradid_idx2_bwd)]
    float load_prim_gradid(int x, int y, int grad_idx = -1) { return load_prim(x * stride + y); }
    [BackwardDerivative(load_prim_gradid_idx3_bwd)]
    float load_prim_gradid(int x, int y, int z, int grad_idx = -1) { return load_prim(x * stride + y * pitch + z); }
    
    void interlocked_add_grad(int x, float val) { _gradient_buffer.InterlockedAddF32((offset_grad + x) * 4, val); }
    void interlocked_add_grad(int x, int y, float val) { interlocked_add_grad(x * stride + y, val); }
    void interlocked_add_grad(int x, int y, int z, float val) { interlocked_add_grad(x * stride + y * pitch + z, val); }
    
    void load_prim_idx1_bwd(int x, float.Differential val) { interlocked_add_grad(x, val); }
    void load_prim_idx2_bwd(int x, int y, float.Differential val) { interlocked_add_grad(x, y, val); }
    void load_prim_idx3_bwd(int x, int y, int z, float.Differential val) { interlocked_add_grad(x, y, z, val); }

    void load_prim_gscalar_idx1_bwd(int x, no_diff float scalar, float.Differential val) { interlocked_add_grad(x, val * scalar); }
    void load_prim_gscalar_idx2_bwd(int x, int y, no_diff float scalar, float.Differential val) { interlocked_add_grad(x, y, val * scalar); }
    void load_prim_gscalar_idx3_bwd(int x, int y, int z, no_diff float scalar, float.Differential val) { interlocked_add_grad(x, y, z, val * scalar); }
    
    void interlocked_add_gradcell(int x, float val) { _gradcell_buffer.InterlockedAddF32(x * 4, val); }
    void interlocked_add_gradcell(int x, int y, float val) { interlocked_add_grad(x * stride + y, val); }
    void interlocked_add_gradcell(int x, int y, int z, float val) { interlocked_add_grad(x * stride + y * pitch + z, val); }

    void load_prim_gradid_idx1_bwd(int x, int grad_idx, float.Differential val) {
        if (grad_idx == -1) interlocked_add_grad(x, val);
        else if (grad_idx < 0) return;
        else interlocked_add_grad(grad_idx, val); }
    void load_prim_gradid_idx2_bwd(int x, int y, int grad_idx, float.Differential val) {
        if (grad_idx == -1) interlocked_add_grad(x, y, val);
        else if (grad_idx < 0) return;
        else interlocked_add_grad(grad_idx, val); }
    void load_prim_gradid_idx3_bwd(int x, int y, int z, int grad_idx, float.Differential val) {
        if (grad_idx == -1) interlocked_add_grad(x, y, z, val);
        else if (grad_idx < 0) return;
        else interlocked_add_grad(grad_idx, val); }

    void load_prim_grad_id_idx1_bwd(int x, int grad_idx, float.Differential val) {
        if (grad_idx == -1) interlocked_add_gradcell(x, val);
        else if (grad_idx < 0) return;
        else interlocked_add_gradcell(grad_idx, val); }
    void load_prim_grad_id_idx2_bwd(int x, int y, int grad_idx, float.Differential val) {
        if (grad_idx == -1) interlocked_add_gradcell(x, y, val);
        else if (grad_idx < 0) return;
        else interlocked_add_gradcell(grad_idx, val); }
    void load_prim_grad_id_idx3_bwd(int x, int y, int z, int grad_idx, float.Differential val) {
        if (grad_idx == -1) interlocked_add_gradcell(x, y, z, val);
        else if (grad_idx < 0) return;
        else interlocked_add_gradcell(grad_idx, val); }
    
    // constructor
    __init() { // default constructor
        offset_prim = 0;
        offset_grad = 0;
        stride = 0;
        pitch = 0; }
    __init(uint offset_prim, uint offset_grad, uint stride, uint pitch = 0
    ) { this.offset_prim = offset_prim;
        this.offset_grad = offset_grad;
        this.stride = stride;
        this.pitch = pitch; }
};

// All half linear layers use the half precision,
// Thus the input and output are also half precision.
struct HalfFeature<let C : int> : IDifferentiable {
    float16_t vals[C];
};

#endif // !_SRENDERER_ADDON_TINYNN_COMMON_HLSLI_HEADER_