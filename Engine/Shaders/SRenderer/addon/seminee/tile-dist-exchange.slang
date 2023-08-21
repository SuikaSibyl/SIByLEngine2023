#include "../../include/common/geometry.hlsli"
#include "seminee-config.hlsli"
#include "seminee_interface.hlsli"

RWStructuredBuffer<VPLData> u_VPLData;
RWStructuredBuffer<int4> u_SelectedVPL;
// RWStructuredBuffer<float> u_DebugBuffer;

groupshared int gs_IDHasher[128];
groupshared float gs_Importance[128];

uint InsertOneBit(uint Value, uint OneBitMask) {
    uint Mask = OneBitMask - 1;
    return (Value & ~Mask) << 1 | (Value & Mask) | OneBitMask;
}

bool ShouldSwap(float A, float B) {
    return A < B;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void ComputeMain(int3 gid: SV_GroupID, int thid: SV_GroupIndex) {
    const int2 superTileID = gid.xy;
    const int2 tileID = superTileID * 5 + int2(thid % 5, thid / 5);
    const int2 tileResolution = computeTileResolution(int2(1280, 720), TILE_SIZE);
    const int tileIndex = tileID.y * tileResolution.x + tileID.x;

    gs_IDHasher[thid] = -1;
    gs_Importance[thid] = -1;
    gs_IDHasher[thid + 64] = -1;
    gs_Importance[thid + 64] = -1;

    int4 selected = int4(-1);
    GroupMemoryBarrierWithGroupSync();
    if (thid < 25) {
        selected = u_SelectedVPL[tileIndex];
        for (int i = 0; i < 4; ++i) {
            if (selected[i] != -1) {
                gs_IDHasher[selected[i] % 128] = selected[i];
                gs_Importance[selected[i] % 128] = u_VPLData[selected[i]].luminance;
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();
    
    // sort all the importance
    uint k;
    [unroll]
    for (k = 2; k <= 128; k <<= 1) {
        [unroll]
        for (uint j = k / 2; j > 0; j /= 2) {
            uint Index2 = InsertOneBit(uint(thid), j);
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
            GroupMemoryBarrierWithGroupSync();
        }
    }

    if (thid < 25) {
        int offset = 0;
        for (int i = 0; i < 4; ++i) {
            if (selected[i] == -1) {
                int potential_id = gs_IDHasher[offset++];
                while (potential_id != -1) {
                    if (all(selected != int4(offset))) {
                        selected[i] = potential_id;
                        break;
                    }
                    potential_id = gs_IDHasher[offset++];
                }
            }
        }
        u_SelectedVPL[tileIndex] = selected;
    }
}