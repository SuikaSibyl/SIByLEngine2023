#include "matmul.hlsli"
#include "tinynn.hlsli"

static const int g_warp_size = 32;
static const int g_num_threads_per_block = 256;
static const int g_num_warps_per_block = g_num_threads_per_block / g_warp_size;

__generic<let C : int> 
struct Feature: IDifferentiable {
    float vals[C];
}

/*
 * An implementation of a CxC linear layer that is designed to be
 * used 'inline' within a larger kernel. 
 */
struct Linear<let C : int> {
    typedef Feature<C> Input;
    typedef Feature<C> Output;
    typedef uint SharedMemRef;

    no_diff ThreadInfo threadInfo;
    no_diff TensorView weights_view;
    no_diff TensorView bias_view;

    __init() {
        this.threadInfo = ThreadInfo();
        this.weights_view = TensorView();
        this.bias_view = TensorView();
    }
    __init(TensorView weights_view, TensorView bias_view, ThreadInfo threadInfo) {
        this.threadInfo = threadInfo;
        this.weights_view = weights_view;
        this.bias_view = bias_view;
    }

    uint calcOffset<let N : int>() {
        return uint(((threadInfo.thread_idx.x / g_warp_size) + threadInfo.thread_idx.y *
            (threadInfo.block_dim.x * 1.0 / g_warp_size)) * N);
    }

    // Get the weight buffer for the current warp.
    SharedMemRef wtBufferForCurrentWarp() { return calcOffset<C * C>(); }
    // Get the input buffer for the current warp.
    SharedMemRef inpBufferForCurrentWarp() { return calcOffset<g_warp_size * C>(); }
    // Get the output buffer for the current warp.
    SharedMemRef outBufferForCurrentWarp() { return calcOffset<g_warp_size * C>(); }

    // move the weights from global memory to shared memory.
    SharedMemRef moveWeightsToSharedMem<let colMajor: bool>() {
        SharedMemRef wtPtr = wtBufferForCurrentWarp();
        // Copy weights to shared memory.
        int2 threadIdx = threadInfo.thread_idx;
        [ForceUnroll]
        for (uint i = 0; i < C; i += g_warp_size) {
            [ForceUnroll]
            for (uint j = 0; j < C; j++) {
                var threadIdInWarp = threadIdx.x % g_warp_size;
                if ((i + threadIdInWarp) >= C) continue;
                if (colMajor) inline_set_weights_element(wtPtr + (i + threadIdInWarp) * C + j,
                                               float16_t(weights_view.load_prim(i + threadIdInWarp, j)));
                else inline_set_weights_element(wtPtr + j * C + (i + threadIdInWarp),
                                               float16_t(weights_view.load_prim(i + threadIdInWarp, j)));
            }
        }
        return wtPtr;
    }
    
    SharedMemRef storeArray_input<let N : int, let colMajor : bool>(SharedMemRef memptr, float input[N]) {
        const uint threadIdInWarp = threadInfo.thread_idx.x % g_warp_size;
        // Each thread in the warp will move N contiguous elements to their corresponding shared memory.
        if (!colMajor) {
            [ForceUnroll]
            for (int i = 0; i < N; i++)
                inline_set_input_element(memptr + (threadIdInWarp * N + i), float16_t(input[i]));
        } else {
            [ForceUnroll]
            for (int i = 0; i < N; i++)
                inline_set_input_element(memptr + (i * g_warp_size + threadIdInWarp), float16_t(input[i]));
        }
        return memptr;
    }

    SharedMemRef storeArray_weights<let N : int, let colMajor : bool>(SharedMemRef memptr, float input[N]) {
        const uint threadIdInWarp = threadInfo.thread_idx.x % g_warp_size;
        // Each thread in the warp will move N contiguous elements to their corresponding shared memory.
        if (!colMajor) {
            [ForceUnroll]
            for (int i = 0; i < N; i++)
                inline_set_weights_element(memptr + (threadIdInWarp * N + i), float16_t(input[i]));
        } else {
            [ForceUnroll]
            for (int i = 0; i < N; i++)
                inline_set_weights_element(memptr + (i * g_warp_size + threadIdInWarp), float16_t(input[i]));
        }
        return memptr;
    }

    void loadArray<let N : int, let colMajor : bool>(
        SharedMemRef memptr, out float input[N]) {
        uint threadIdInWarp = threadInfo.thread_idx.x % g_warp_size;
        // Each thread in the warp will move N contiguous elements to their corresponding shared memory.
        if (!colMajor) {
            [ForceUnroll]
            for (int i = 0; i < N; i++)
                input[i] = inline_get_output_element(memptr + threadIdInWarp * N + i);
        } else {
            [ForceUnroll]
            for (int i = 0; i < N; i++)
                input[i] = inline_get_output_element(memptr + i * g_warp_size + threadIdInWarp);;
        }
    }

    SharedMemRef moveInputsToSharedMem<let N: int>(float input[N]) {
        // Pack in row-major format.
        SharedMemRef inPtr = inpBufferForCurrentWarp();
        return storeArray_input<N, false>(inPtr, input);
    }
    
    SharedMemRef moveDInputsToSharedMem<let N : int>(float input[N]) {
        // Pack in col-major format.
        SharedMemRef inPtr = inpBufferForCurrentWarp();
        return storeArray_weights<N, true>(inPtr, input);
    }

    SharedMemRef moveDOutputsToSharedMem<let N: int>(float d_output[N]) {
        // Pack in _transposed_ row-major.. which is just col-major.
        SharedMemRef outPtr = outBufferForCurrentWarp();
        return storeArray_input<N, true>(outPtr, d_output);
    }
    
    void moveOutputsToLocalArray<let N: int>(out float outputs[N]) {
        SharedMemRef outPtr = outBufferForCurrentWarp();
        loadArray<N, false>(outPtr, outputs);
        for (int i = 0; i < N; i++)
            outputs[i] = outputs[i] + bias_view.load_prim(i);
    }

    [BackwardDerivative(eval_bwd)]
    Output _eval(Input in_feature) {
        // Move the input and weights to shared memory.
        SharedMemRef inPtr = moveInputsToSharedMem<C>(in_feature.vals);
        SharedMemRef wtPtr = moveWeightsToSharedMem<false>();
        // Get the output buffer for the current warp.
        SharedMemRef outPtr = outBufferForCurrentWarp();
        // Do the matmul.
        __inline_matmul_impl_32_16_16(inPtr, wtPtr, outPtr);
        // Move the output to local memory.
        Output out_feature;
        moveOutputsToLocalArray<C>(out_feature.vals);
        // output the result.
        return out_feature;
    }

    void eval_bwd(inout DifferentialPair<Input> in_feature_pair, Feature<C>.Differential d_output) {
        uint warpOffset = calcOffset<C * C>();
        // Accumulate input derivatives. dodi
        // which is simply do*W^T
        {   SharedMemRef dOutPtr = moveInputsToSharedMem<C>(d_output.vals);
            SharedMemRef wtPtr = moveWeightsToSharedMem<true>();
            SharedMemRef dInPtr = outBufferForCurrentWarp();
            __inline_matmul_impl_32_16_16(dOutPtr, wtPtr, dInPtr);
            Input.Differential d_input_feature;
            loadArray<C, false>(dInPtr, d_input_feature.vals);
            in_feature_pair = DifferentialPair<Input>(in_feature_pair.p, d_input_feature);
        }
        // Accumulate weight derivatives.
        {   SharedMemRef inPtr = moveDInputsToSharedMem<C>(in_feature_pair.p.vals);
            SharedMemRef outPtr = moveDOutputsToSharedMem<C>(d_output.vals);

            SharedMemRef wtPtr = wtBufferForCurrentWarp();
            __inline_matmul_impl_16_32_16(outPtr, inPtr, wtPtr);
                        
            [ForceUnroll]
            for (uint i = 0; i < C; i += g_warp_size) {
                [ForceUnroll]
                for (uint j = 0; j < C; j++) {
                    var threadIdInWarp = threadInfo.thread_idx.x % g_warp_size;
                    if ((i + threadIdInWarp) >= C)
                        continue;
                    float oldVal;
                    weights_view.interlocked_add_grad(j, i + threadIdInWarp,
                        inline_get_output_element((i + threadIdInWarp) * C + j));
                }
            }
        }
        
        // Accumulate bias derivatives.
        {   [ForceUnroll]
            for (int i = 0; i < C; i ++) {
                float total_d_bias = WaveActiveSum(d_output.vals[i]);
                if (WaveIsFirstLane()) {
                    bias_view.interlocked_add_grad(i, total_d_bias);
                }
            }
        }
    }

};

[Differentiable]
static Linear<C>.Output eval<let C : int>(Linear<C> layer, Linear<C>.Input in_feature) {
    return layer._eval(in_feature);
}

// struct MLP<let C : int, let N : int> {
//     typedef Feature<C> Input;
//     typedef Feature<C> Output;

//     // Linear<C> layers[N];

//     // __init(Linear<C> layers[N], ThreadInfo threadInfo) {
//     //     this.layers = layers;
//     // }

//     // [Differentiable]
//     Output eval(Linear<C> layers[N], Input in_feature) {
//         Output out_feature = in_feature;
//         [ForceUnroll]
//         for (int i = 0; i < N; i++) {
//             out_feature = layers[i].eval(out_feature);
//             // ReLU
//             [ForceUnroll]
//             for (int j = 0; j < C; j++) {
//                 out_feature.vals[j] = max(0.0f, out_feature.vals[j]);
//             }
//         }
//         return out_feature;
//     }
// };