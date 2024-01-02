[[vk::push_constant]]
cbuffer PushConstants {
    uint max_vertex;
    uint max_joint;
};

// input all skeleton position data
[[vk::binding(0,0)]] RWByteAddressBuffer skeleton_pos;
[[vk::binding(1,0)]] RWByteAddressBuffer skeleton_normal;
[[vk::binding(2,0)]] StructuredBuffer<uint32_t> skeleton_offset;
[[vk::binding(3,0)]] StructuredBuffer<int4> skeleton_joints;
[[vk::binding(4,0)]] StructuredBuffer<float4> skeleton_weights;
[[vk::binding(5,0)]] StructuredBuffer<float4x4> skeleton_transforms;

// output to both position and vertex buffer
[[vk::binding(6, 0)]] RWByteAddressBuffer position_buffer;
[[vk::binding(7, 0)]] RWByteAddressBuffer vertex_buffer;

groupshared float4x4 gs_jointmat[512];

float4x4 get_jointmat(int i) {
    if (i < max_joint) return gs_jointmat[i];
    else return skeleton_transforms[i];
}

[shader("compute")]
[numthreads(512, 1, 1)]
void ComputeMain(uint3 DTid: SV_DispatchThreadID,
                 uint Gid: SV_GroupIndex,
) {
    const uint tid = DTid.x;
    if (tid >= max_vertex) return;
    // load the jointmat
    if (Gid < max_joint)
        gs_jointmat[Gid] = skeleton_transforms[Gid];
    GroupMemoryBarrierWithGroupSync();
    
    const uint4 a_index = skeleton_joints[tid];
    const float4 a_weight = skeleton_weights[tid];
    
    float4x4 jointMatrix[4];
    for (int i = 0; i < 4; ++i) {
        jointMatrix[i] = get_jointmat(a_index[i]);
    }

    const float4x4 skinMat =
        a_weight.x * jointMatrix[0] +
        a_weight.y * jointMatrix[1] +
        a_weight.z * jointMatrix[2] +
        a_weight.w * jointMatrix[3];
    
    const float3 a_position = skeleton_pos.Load<float3>(tid * 12);
    const float3 a_normal = skeleton_normal.Load<float3>(tid * 12);
    const uint32_t offset = skeleton_offset[tid];

    const float3 worldPosition = mul(skinMat, float4(a_position, 1.0)).xyz;
    const float3 worldNormal = normalize(mul(skinMat, float4(a_normal, 0.0)).xyz);
    
    position_buffer.Store<float3>((offset + tid) * 12 + 0, worldPosition);
    vertex_buffer.Store<float3>((offset + tid) * 44 + 0, worldPosition);
    vertex_buffer.Store<float3>((offset + tid) * 44 + 12, worldNormal);
}