
#define VOXEL_OPACITY_SCALA
#ifdef VOXEL_OPACITY_SCALA

const RWTexture3D<float> uOpacityVox;      // 64x64x64
const RWTexture3D<float> uOpacityVox_mip1; // 32x32x32
const RWTexture3D<float> uOpacityVox_mip2; // 16x16x16
const RWTexture3D<float> uOpacityVox_mip3; // 8x8x8

groupshared float gs_buffer[8][8][8];

void CreateOpacityMipmap(uint3 gid, uint3 gtid, uint gi) {
    const int3 offsets[8] = {
        int3(0, 0, 0),
        int3(1, 0, 0),
        int3(0, 1, 0),
        int3(1, 1, 0),
        int3(0, 0, 1),
        int3(1, 0, 1),
        int3(0, 1, 1),
        int3(1, 1, 1)
    };

    uint3 voxelToFetch = gid * 8 + gtid;
    const float opcLoad = uOpacityVox[voxelToFetch];
    gs_buffer[gtid.x][gtid.y][gtid.z] = opcLoad;
    GroupMemoryBarrierWithGroupSync();

    // For scalar opacity, maybe let's just average the scalar.
    
    float opcChunk[8];
    // Create 1st mip level (4x4x4 chunks)
    if (gi >= 64) return; // release some warps
    uint3 chunkID = uint3(gi / 16, (gi / 4) % 4, gi % 4); // 4x4x4 chunks
    uint3 chunkOffset = chunkID * 2;
    float avgOpcaity = 0.0;
    for (int i = 0; i < 8; ++i) {
        const uint3 offset = chunkOffset + offsets[i];
        opcChunk[i] = gs_buffer[offset.x][offset.y][offset.z];
        avgOpcaity += opcChunk[i];
    }
    avgOpcaity /= 8.0;
    // save 1st mip level value
    uOpacityVox_mip1[gid * 4 + chunkID] = avgOpcaity;
    // also save 1st mip level value to the shared memory
    GroupMemoryBarrierWithGroupSync();  // sync before write
    gs_buffer[chunkID.x][chunkID.y][chunkID.z] = avgOpcaity;
    GroupMemoryBarrierWithGroupSync();

    // Create 2st mip level (2x2x2 chunks)
    if (gi >= 32) return; // release some warps
    if (gi < 8) {
        chunkID = uint3(gi / 4, (gi / 2) % 2, gi % 2); // 2x2x2 chunks
        chunkOffset = chunkID * 2;
        avgOpcaity = 0.0;
        for (int i = 0; i < 8; ++i) {
            const uint3 offset = chunkOffset + offsets[i];
            opcChunk[i] = gs_buffer[offset.x][offset.y][offset.z];
            avgOpcaity += opcChunk[i];
        }
        avgOpcaity /= 8.0;
        // save 1st mip level value
        uOpacityVox_mip2[gid * 2 + chunkID] = avgOpcaity;
    }
    // also save 1st mip level value to the shared memory
    GroupMemoryBarrierWithGroupSync(); // sync before write
    gs_buffer[chunkID.x][chunkID.y][chunkID.z] = avgOpcaity;
    GroupMemoryBarrierWithGroupSync();

    // Create 3st mip level (1x1x1 chunks)
    if (gi == 0) {
        avgOpcaity = 0.0;
        for (int i = 0; i < 8; ++i) {
            const uint3 offset = offsets[i];
            opcChunk[i] = gs_buffer[offset.x][offset.y][offset.z];
            avgOpcaity += opcChunk[i];
        }
        avgOpcaity /= 8.0;
        // save 1st mip level value
        uOpacityVox_mip3[gid] = avgOpcaity;
    }
}

#endif

[shader("compute")]
[numthreads(8, 8, 8)] // 8x8x8 threads in a group
void ComputeMain(uint3 gid: SV_GroupID, uint3 gtid: SV_GroupThreadID, uint gi: SV_GroupIndex) {
    CreateOpacityMipmap(gid, gtid, gi);
}