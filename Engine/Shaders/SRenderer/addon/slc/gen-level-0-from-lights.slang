#include "include/slc_interface.hlsli"

StructuredBuffer<float4> u_vplPositions;
StructuredBuffer<float4> u_vplNormals;
StructuredBuffer<float4> u_vplColors;
ByteAddressBuffer u_vplCounter;

ByteAddressBuffer u_keyIndexList;

RWStructuredBuffer<SLCNode> u_nodes;

[[vk::push_constant]]
cbuffer PushConstants {
    int numTreeLevels;
    int numTreeLights;
};

int CalculateTreeLevels(int numVPLs) {
    return int(ceil(log2(numVPLs))) + 1;
}

[shader("compute")]
[numthreads(512, 1, 1)]
void ComputeMain(uint3 DTid: SV_DispatchThreadID) {
    const int levelNodeId = int(DTid.x);
    const int numVPLs = int(u_vplCounter.Load(0));
	
    if (levelNodeId < numTreeLights) {
        SLCNode node;
        const int nodeArr = (1 << (numTreeLevels - 1)) + levelNodeId;
		if (levelNodeId < numVPLs) {
			uint2 KeyIndexPair = u_keyIndexList.Load2(8 * levelNodeId);
			int index = int(KeyIndexPair.x);
			float3 lightPos = u_vplPositions[index].xyz;
			float3 lightN = u_vplNormals[index].xyz;
			float3 lightColor = u_vplColors[index].xyz;
			node.ID = index;
			// For 16 bit version this doesn't matter (if we really need to construct the similar invalid bound it 
			// needs to be corner + (1+eps)dimension)
			float3 boundMin = 1e10;
			float3 boundMax = -1e10;
			node.intensity = GetColorIntensity(lightColor.xyz);
 			//real light
            if (node.intensity > 0) {
                boundMin = lightPos.xyz;
                node.cone.xyz = lightN.xyz;
                node.cone.w = 0;
				node.boundMax = node.boundMin = boundMin;
			}
			else {
				node.boundMin = 1e10;
				node.boundMax = -1e10;
			}
		}
		else {
			node.intensity = 0;
			node.boundMin = 1e10;
            node.boundMax = -1e10;
            node.ID = -1;
		}
		
		u_nodes[nodeArr] = node;
	}
}