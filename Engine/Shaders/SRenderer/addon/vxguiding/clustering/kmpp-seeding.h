shared float warp_prob[32];
shared vec3 current_center;
shared int selected_cluster;

layout(binding = 0, set = 0) buffer ClusterSeedBuffer { int seeds[]; };
layout(binding = 1, set = 0) buffer CompactIndexBuffer { uint compactIndex[]; };
layout(binding = 2, set = 0) buffer SVXInfoBuffer { vec4 vxNormal[]; };
layout(binding = 3, set = 0) buffer CounterBuffer { int counter[]; };

layout(binding = 4, set = 0) buffer DebugBuffer { float debug[]; };
