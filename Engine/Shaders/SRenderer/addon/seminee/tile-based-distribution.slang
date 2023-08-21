#include "../../include/common/geometry.hlsli"
#include "seminee-config.hlsli"
#include "seminee_interface.hlsli"

RWStructuredBuffer<VPLData> u_VPLData;
RWStructuredBuffer<int4> u_SelectedVPL;
RWStructuredBuffer<float> u_DebugBuffer;

groupshared int gs_IDHasher[256];
groupshared float gs_Importance[256];

uint InsertOneBit(uint Value, uint OneBitMask) {
    uint Mask = OneBitMask - 1;
    return (Value & ~Mask) << 1 | (Value & Mask) | OneBitMask;
}

bool ShouldSwap(float A, float B) {
    return A < B;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void ComputeMain(int3 gid: SV_GroupID, int tid: SV_GroupIndex) {
    // Each group handles a tile
    int2 tileID = int2(gid.x, gid.y);
    int2 tileResolution = computeTileResolution(int2(1280, 720), TILE_SIZE);
    int tileIndex = tileID.y * tileResolution.x + tileID.x;
    int tileOffset = tileIndex * TILE_SIZE * TILE_SIZE;

    gs_IDHasher[tid] = -1;

    int vplID = tileOffset + tid;
    VPLData vpl = u_VPLData[vplID];
    int nodeID = -1;
    if (!vpl.isInvalid()) {
        while (vpl.idParent != -1) {
            VPLData parentVPL;
            parentVPL = u_VPLData[vpl.idParent];
            if (parentVPL.getEarlyStop() == 0.) {
                nodeID = vpl.id;
                break;
            }
            vpl = parentVPL;
            if (vpl.idParent == -1) {
                nodeID = vpl.id;
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (nodeID != -1)
        gs_IDHasher[nodeID % 256] = nodeID; // deduplicate by simple hashing
    GroupMemoryBarrierWithGroupSync();

    int dd_vplID = gs_IDHasher[tid];
    float weight = -1.f;
    if (dd_vplID != -1) {
        VPLData dd_vpl = u_VPLData[dd_vplID];
        weight = dd_vpl.luminance;
    }
    gs_Importance[tid] = weight;
    GroupMemoryBarrierWithGroupSync();
    
    // sort the importance
    // Sort on shared memory
    uint k;
    // This is better unrolled because it reduces ALU and because some
    // architectures can load/store two LDS items in a single instruction
    // as long as their separation is a compile-time constant.
    [unroll]
    for (k = 2; k <= 256; k <<= 1) {
        [unroll]
        for (uint j = k / 2; j > 0; j /= 2) {
            if (tid < 128) {
                uint Index2 = InsertOneBit(uint(tid), j);
                uint Index1 = Index2 ^ (k == 2 * j ? k - 1 : j);

                float A = gs_Importance[Index1];
                float B = gs_Importance[Index2];
                int a = gs_IDHasher[Index1];
                int b = gs_IDHasher[Index2];

                if (A < B) {
                    // Swap the keys
                    gs_Importance[Index1] = B;
                    gs_Importance[Index2] = A;
                    gs_IDHasher[Index1] = b;
                    gs_IDHasher[Index2] = a;
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }
    }

    int offsetDebug = tileIndex * 256 + tid;
    u_DebugBuffer[offsetDebug] = gs_Importance[tid];

    // select the top 4 ids
    if (tid < 4) {
        int id = gs_IDHasher[tid];
        u_SelectedVPL[tileIndex][tid] = id;
    }
}