RWTexture3D<float2> uRadOpaVox6D[6];

[shader("compute")]
[numthreads(8, 8, 8)] // 8x8x8 threads in a group
void ComputeMain(uint3 gid: SV_GroupID, uint3 gtid: SV_GroupThreadID, uint gi: SV_GroupIndex) {
    const uint3 voxelToFetch = gid * 8 + gtid;
    for (int i = 0; i < 6; ++i)
        uRadOpaVox6D[i][voxelToFetch] = float2(0.0f, 0.0f);
}