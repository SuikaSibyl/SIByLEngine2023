#ifndef SECOND_PASS
#define BUILD_MIP3 1
#endif

const RWTexture3D<float2> uRadOpaVox6D[6];      // 64x64x(64x6)  |  8x8x(8x6)
const RWTexture3D<float2> uRadOpaVox6D_MIP1[6]; // 32x32x(32x6)  |  4x4x(4x6)
const RWTexture3D<float2> uRadOpaVox6D_MIP2[6]; // 16x16x(16x6)  |  2x2x(2x6)
#ifdef BUILD_MIP3
const RWTexture3D<float2> uRadOpaVox6D_MIP3[6]; // 8x8x(8x6)
#endif // BUILD_MIP3

[[vk::push_constant]]
cbuffer PushConstantBuffer {
    bool SaveNumericalIssue;
};
groupshared float2 gs_buffer[8][8][8];

[shader("compute")]
[numthreads(8, 8, 8)] // 8x8x8 threads in a group
void ComputeMain(uint3 gid: SV_GroupID, uint3 gtid: SV_GroupThreadID, uint gi: SV_GroupIndex) {
#ifdef SECOND_PASS
    const uint faceIdx = gid.z; // face index
    gid.z = 0;                  // remove face index
#else
    const uint faceIdx = gid.z / 8; // face index
    gid.z %= 8;                     // remove face index
#endif
    uint3 igid = uint3(gid.xy, gid.z % 8);  // voxel index remove face index
    
    // each float2 pack consists of:
    // x: radiance luminance
    // y: opacity
    float2 rad_opc[8];
    // Voxel axis convention:
    //
    //                  /|\ 
    //                   | z-axis
    //                   |          
    //                //////////
    //               ---------//
    //     //////////| 4 | 6 |//
    //    ---------//---------//
    //    | 5 | 7 |//| 0 | 2 |/   -- y-axis -->
    //    ---------//---------
    //    | 1 | 3 |/    / x-axis
    //     ---------  \/_
    //
    const int3 offsets[8] = {
        int3(0, 0, 0), // 0
        int3(1, 0, 0), // 1
        int3(0, 1, 0), // 2
        int3(1, 1, 0), // 3
        int3(0, 0, 1), // 4
        int3(1, 0, 1), // 5
        int3(0, 1, 1), // 6
        int3(1, 1, 1)  // 7
    };
    const int4 faceIndexArr1st[6] = {
        int4(1, 3, 5, 7), // positive x
        int4(0, 2, 4, 6), // negative x
        int4(2, 3, 6, 7), // positive y
        int4(0, 1, 4, 5), // negative y
        int4(4, 5, 6, 7), // positive z
        int4(0, 1, 2, 3)  // negative z
    };
    const int4 faceIndexArr2nd[6] = {
        int4(0, 2, 4, 6), // positive x
        int4(1, 3, 5, 7), // negative x
        int4(0, 1, 4, 5), // positive y
        int4(2, 3, 6, 7), // negative y
        int4(0, 1, 2, 3), // positive z
        int4(4, 5, 6, 7)  // negative z
    };
    // Fetch face indeces
    const int4 faceIndex1st = faceIndexArr1st[faceIdx];
    const int4 faceIndex2nd = faceIndexArr2nd[faceIdx];

    // Load 0th mip level (64x64x64 voxels) to shared memory
    uint3 voxelToFetch = igid * 8 + gtid;
    const float2 opcLoad = uRadOpaVox6D[faceIdx][voxelToFetch];
    gs_buffer[gtid.x][gtid.y][gtid.z] = opcLoad;
    GroupMemoryBarrierWithGroupSync();

    // Create 1st mip level (4x4x4 chunks)
    if (gi >= 64) return; // release some warps
    uint3 chunkID = uint3(gi / 16, (gi / 4) % 4, gi % 4); // 4x4x4 chunks
    uint3 chunkOffset = chunkID * 2;
    for (int i = 0; i < 8; ++i) {
        const uint3 offset = chunkOffset + offsets[i];
        rad_opc[i] = gs_buffer[offset.x][offset.y][offset.z];
    }
    
    // synthesis 1st mip level
    float2 mip1 = (rad_opc[faceIndex1st[0]] + (1.0 - rad_opc[faceIndex1st[0]].y) * rad_opc[faceIndex2nd[0]] +
                   rad_opc[faceIndex1st[1]] + (1.0 - rad_opc[faceIndex1st[1]].y) * rad_opc[faceIndex2nd[1]] +
                   rad_opc[faceIndex1st[2]] + (1.0 - rad_opc[faceIndex1st[2]].y) * rad_opc[faceIndex2nd[2]] +
                   rad_opc[faceIndex1st[3]] + (1.0 - rad_opc[faceIndex1st[3]].y) * rad_opc[faceIndex2nd[3]]) * 0.25;
    if (SaveNumericalIssue)
        mip1.x *= 8;
    // save 1st mip level value
    uRadOpaVox6D_MIP1[faceIdx][igid * 4 + chunkID] = mip1;

    // also save 1st mip level value to the shared memory
    GroupMemoryBarrierWithGroupSync(); // sync before write
    gs_buffer[chunkID.x][chunkID.y][chunkID.z] = mip1;
    GroupMemoryBarrierWithGroupSync();

    // Create 2st mip level (2x2x2 chunks)
    if (gi >= 32) return; // release some warps
    float2 mip2;
    if (gi < 8) {
        chunkID = uint3(gi / 4, (gi / 2) % 2, gi % 2); // 2x2x2 chunks
        chunkOffset = chunkID * 2;
        for (int i = 0; i < 8; ++i) {
            const uint3 offset = chunkOffset + offsets[i];
            rad_opc[i] = gs_buffer[offset.x][offset.y][offset.z];
        }
        mip2 = (rad_opc[faceIndex1st[0]] + (1.0 - rad_opc[faceIndex1st[0]].y) * rad_opc[faceIndex2nd[0]] +
                rad_opc[faceIndex1st[1]] + (1.0 - rad_opc[faceIndex1st[1]].y) * rad_opc[faceIndex2nd[1]] +
                rad_opc[faceIndex1st[2]] + (1.0 - rad_opc[faceIndex1st[2]].y) * rad_opc[faceIndex2nd[2]] +
                rad_opc[faceIndex1st[3]] + (1.0 - rad_opc[faceIndex1st[3]].y) * rad_opc[faceIndex2nd[3]]) * 0.25;
        if (SaveNumericalIssue)
            mip2.x *= 8;
        // save 1st mip level value
        uRadOpaVox6D_MIP2[faceIdx][igid * 2 + chunkID] = mip2;
    }
#ifdef BUILD_MIP3
    // also save 1st mip level value to the shared memory
    GroupMemoryBarrierWithGroupSync(); // sync before write
    if (gi < 8) {
        gs_buffer[chunkID.x][chunkID.y][chunkID.z] = mip2;
    }
    GroupMemoryBarrierWithGroupSync();
    
    // Create 3st mip level (1x1x1 chunks)
    if (gi == 0) {
        for (int i = 0; i < 8; ++i) {
            const uint3 offset = offsets[i];
            rad_opc[i] = gs_buffer[offset.x][offset.y][offset.z];
        }
        float2 mip3 = (rad_opc[faceIndex1st[0]] + (1.0 - rad_opc[faceIndex1st[0]].y) * rad_opc[faceIndex2nd[0]] +
                       rad_opc[faceIndex1st[1]] + (1.0 - rad_opc[faceIndex1st[1]].y) * rad_opc[faceIndex2nd[1]] +
                       rad_opc[faceIndex1st[2]] + (1.0 - rad_opc[faceIndex1st[2]].y) * rad_opc[faceIndex2nd[2]] +
                       rad_opc[faceIndex1st[3]] + (1.0 - rad_opc[faceIndex1st[3]].y) * rad_opc[faceIndex2nd[3]]) * 0.25;
        if (SaveNumericalIssue)
            mip3.x *= 8;

        // save 1st mip level value
        uRadOpaVox6D_MIP3[faceIdx][igid] = mip3;
    }
#endif // BUILD_MIP3
}