#include "include/slc_interface.hlsli"

[[vk::push_constant]]
cbuffer PushConstants {
	int srcLevel;
	int dstLevelStart;
	int dstLevelEnd;
	int numLevels;
	int numDstLevelsLights;
};

RWStructuredBuffer<SLCNode> u_nodes;

[shader("compute")]
[numthreads(512, 1, 1)]
void ComputeMain(uint3 DTid: SV_DispatchThreadID) {
	int nodeId = int(DTid.x);
    if (nodeId < numDstLevelsLights) {
		// Each thread processes a light tree node
        int dstNodeArr;	// the node array index of the destination node
        int dstLevel;

		// If only process one level
        if (dstLevelEnd == dstLevelStart + 1) {
            // the level of the destination node
			dstLevel = dstLevelStart;
			dstNodeArr = (1 << (numLevels - dstLevel - 1)) + nodeId;
		}
		// If process multiple levels
		else {
            dstNodeArr = (1 << (numLevels - dstLevelEnd)) + nodeId;
            dstLevel = numLevels - 1 - int(uintLog2(dstNodeArr));
		}

        // Get the start/end of subnodes to process
		const int startNodeId = dstNodeArr << (dstLevel - srcLevel);
		const int endNodeId = startNodeId + (1 << (dstLevel - srcLevel));

        SLCNode node = u_nodes[startNodeId];
        node.ID = dstLevel;

        for (int nodeId = startNodeId + 1; nodeId < endNodeId; nodeId++) {
            SLCNode srcNode = u_nodes[nodeId];
            // actual light
			if (srcNode.intensity > 0) {
				node.intensity += srcNode.intensity;
				node.boundMin = min(srcNode.boundMin, node.boundMin);
				node.boundMax = max(srcNode.boundMax, node.boundMax);
				node.cone = MergeCones(node.cone, srcNode.cone);
			}
		}

        u_nodes[dstNodeArr] = node;
	}
}