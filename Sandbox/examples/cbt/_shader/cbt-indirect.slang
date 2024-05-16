#include "cbt/cbt.hlsli"
#include "cbt/cbt-sbt.hlsli"

RWStructuredBuffer<int> indirect_buffer;

[shader("compute")]
[numthreads(1, 1, 1)]
void ComputeMain(
    int3 dtid: SV_DispatchThreadID,
    int3 gtid: SV_GroupThreadID,
    int3 gid: SV_GroupID,
    int gi: SV_GroupIndex
) {
    int node_count = cbt_NodeCount();
    
    // indirect draw args
    indirect_buffer[0] = 360;
    indirect_buffer[1] = node_count;
    indirect_buffer[2] = 0;
    indirect_buffer[3] = 0;
    
    // indirect dispatch args
    indirect_buffer[4] = (node_count + 255) / 256;
    indirect_buffer[5] = 1;
    indirect_buffer[6] = 1;
}