

RWStructuredBuffer<float> u_buffer;

[shader("compute")]
[numthreads(16, 16, 1)]
void ComputeMain(int3 dtid: SV_DispatchThreadID) {
    
    // u_buffer.Store<float>(dtid.x * 4, 3.14f);
}