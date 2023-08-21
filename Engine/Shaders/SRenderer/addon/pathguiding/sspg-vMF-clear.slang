[[vk::push_constant]]
cbuffer PushConstants {
    int2 resolution;
    bool clearAll;
};

RWTexture2D<uint> u_epochCounter;

[shader("compute")]
[numthreads(16, 16, 1)] // 8x8x8 threads in a group
void ComputeMain(int3 dtid: SV_DispatchThreadID) {
    if (any(dtid.xy >= resolution)) return;
    
    if (clearAll) {
        u_epochCounter[dtid.xy] = 0;
    }
}