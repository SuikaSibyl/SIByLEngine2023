#include "../../../include/common/cylindrical.hlsli"
#include "compressed-directional-quadtree.hlsli"

RWTexture2D<float> u_PdfNormalizing;
const StructuredBuffer<uint64_t> u_cdq;

float NodeInvArea(
    in_ref(uint64_t) cdq,
    in_ref(float2) uv,
) {
    uint child_id = find_child_id(uv);
    int nodeID = child_id;
    float area = 1.0;
    while (nodeID < 64 && is_inner(cdq, nodeID)) {
        // is inner node
        uv = zoom_in_coord(uv);
        child_id = find_child_id(uv);
        nodeID = child(cdq, nodeID, child_id);
        area *= 4;
    }
    return area;
}

[shader("compute")]
[numthreads(16, 16, 1)] // 8x8x8 threads in a group
void ComputeMain(int3 dtid: SV_DispatchThreadID) {
    // load the first hit from the vbuffer
    const int2 pixel = int2(dtid.xy);
    // const VBufferData vhit = UnpackVBufferData(u_vBuffer[debugPixel]);
    // if (IsInvalid(vhit)) {
    //     u_output[pixel] = float4(1, 0, 1, 1); // error color
    //     return;
    // }
    
    const float2 uv = (float2(pixel) + 0.5f) / float2(512);
    const uint64_t cdq = u_cdq[0];
    const float importance = NodeInvArea(cdq, uv);
    u_PdfNormalizing[pixel] = importance;
}