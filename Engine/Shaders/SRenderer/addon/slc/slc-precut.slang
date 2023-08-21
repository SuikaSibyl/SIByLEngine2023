

[numthreads(16, 16, 1)]
void main(uint3 DTid: SV_DispatchThreadID) {
    int2 anchor = DTid.xy * CutShareGroupSize;
    if (anchor.x >= scrWidth || anchor.y >= scrHeight) return;

    RandomSequence rng;

    int dispatchWidth = (scrWidth + CutShareGroupSize - 1) / CutShareGroupSize;
    RandomSequence_Initialize(rng, dispatchWidth * DTid.y + DTid.x, frameId);
    rng.Type = 0;

    int2 realCutSharedGroupSize = int2(min(scrWidth - anchor.x, CutShareGroupSize),
                                       min(scrHeight - anchor.y, CutShareGroupSize));

    int offset = min(realCutSharedGroupSize.x * realCutSharedGroupSize.y - 1,
                     int(realCutSharedGroupSize.x * realCutSharedGroupSize.y * RandomSequence_GenerateSample1D(rng)));
    // randomize pivot pixel position
    int offsetX = offset % realCutSharedGroupSize.x;
    int offsetY = offset / realCutSharedGroupSize.x;

    int2 samplePosition = CutShareGroupSize * DTid.xy + int2(offsetX, offsetY);
    float3 p = texPosition[samplePosition].xyz;
    float3 N = texNormal[samplePosition].xyz;
    float3 V = normalize(viewerPos - p);

    int startAddr = MAX_CUT_NODES * (DTid.y * ((scrWidth + CutShareGroupSize - 1) / CutShareGroupSize) + DTid.x);
    int numLights = 1;
    
    OneLevelLightHeapData heap[MAX_CUT_NODES + 1];
    heap[1].NodeID = 1;
    heap[1].error = 1e27;
    int maxId = 1;
    int lightcutNodes[MAX_CUT_NODES];
    lightcutNodes[0] = 1;
    while (numLights < MaxCutNodes)
    {
        int id = maxId;
        int NodeID = heap[id].NodeID;

#ifdef CPU_BUILDER
        int pChild = BLAS[NodeID].ID;
#else
        int pChild = NodeID << 1;
#endif
        int sChild = pChild + 1;
        
        lightcutNodes[id - 1] = pChild;
        heap[id].NodeID = pChild;
        heap[id].error = errorFunction(-1, pChild, p, N, V, TLAS, BLAS, g_BLASHeaders, TLASLeafStartIndex
			);

        // check bogus light
        if (BLAS[sChild].intensity > 0)
        {
            numLights++;
            lightcutNodes[numLights - 1] = sChild;
            heap[numLights].NodeID = sChild;
            heap[numLights].error = errorFunction(-1, sChild, p, N, V, TLAS, BLAS, g_BLASHeaders, TLASLeafStartIndex
				);
        }

        // find maxId
        float maxError = -1e10;
        for (int i = 1; i <= numLights; i++)
        {
            if (heap[i].error > maxError)
            {
                maxError = heap[i].error;
                maxId = i;
            }
        }
        if (maxError <= 0) break;
    }

    // write lightcut nodes
    int startAddr = MAX_CUT_NODES * (DTid.y * ((scrWidth + CutShareGroupSize - 1) / CutShareGroupSize) + DTid.x);
    for (int i = 0; i < MaxCutNodes; i++)
    {
        if (i < numLights) lightcutBuffer[startAddr + i] = lightcutNodes[i];
        else lightcutBuffer[startAddr + i] = -1;
    }
}