#extension GL_EXT_shader_16bit_storage : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int8: enable

struct TreeNode {
    // 16
    uvec2 aabbMin;
    uvec2 aabbMax;
    // 16
    float intensity;
    uint flag;           // got node already processed?
    uint16_t parent_idx; // parent node
    uint16_t left_idx;   // index of left  child node
    uint16_t right_idx;  // index of right child node
    uint16_t vx_idx;     // == 0xFFFF if internal node.
};

layout(binding = 0, set = 0) buffer TopLevelTreeBuffer { float tltree[]; };
layout(binding = 1, set = 0) buffer ClusterNodesBuffer { int cluster[]; };
layout(binding = 2, set = 0) buffer TreeNodesBuffer { TreeNode nodes[]; };
layout(binding = 3, set = 0, r32ui) readonly uniform uimage2D visibilityIMG;
layout(binding = 4, set = 0) buffer AvgVisibilityBuffer { float avg_visibility[]; };

// layout(binding = 4, set = 0) buffer TopLevelProbBuffer { float tlprob[]; };
