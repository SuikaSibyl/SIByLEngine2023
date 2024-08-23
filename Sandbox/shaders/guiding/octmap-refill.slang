StructuredBuffer<float2> input_buffer;
RWStructuredBuffer<float> octmap_buffer;

[shader("compute")]
[numthreads(512, 1, 1)]
void ComputeMain(
    int3 dtid: SV_DispatchThreadID,
    int3 gtid: SV_GroupThreadID,
    int3 ctid: SV_GroupID,
    int gid: SV_GroupIndex
) {
    int globalID = dtid.x;
    float2 input_val = input_buffer[globalID];
    float value = input_val.x / input_val.y;
    if (input_val.y == 0) value = 0;
    // value = 1.f;
    octmap_buffer[(globalID / 64) * 84 + globalID % 64] = value;
}