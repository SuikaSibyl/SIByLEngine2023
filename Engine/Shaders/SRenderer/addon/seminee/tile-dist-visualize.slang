#include "../../include/common/geometry.hlsli"
#include "../../include/common/camera.hlsli"
#include "seminee_interface.hlsli"
#include "seminee-config.hlsli"

cbuffer CameraBuffer {
    CameraData gCamera;
};

[[vk::push_constant]]
cbuffer PushConstantBuffer {
   int2 chosenPixel;
  float gVPLRenderScale;
  float gVPLColorScale;
}

const RWStructuredBuffer<VPLData> u_VPLData;
RWStructuredBuffer<int4> u_SelectedVPL;

struct AssembledVertex {
    int vertexId : SV_VertexId;
    int instanceId : SV_InstanceID;
};
struct CoarseVertex {
    int vplId;
    int isBackFace;
};
struct VertexStageOutput {
    CoarseVertex coarseVertex : CoarseVertex;
    float4 sv_position : SV_Position;
};

[shader("vertex")]
VertexStageOutput vertexMain(
    AssembledVertex assembledVertex)
{
    const float3 rect[6] = {
        // triangle 0
        float3(-0.5, -0.5, 0),
        float3(+0.5, -0.5, 0),
        float3(-0.5, +0.5, 0),
        // triangle 1
        float3(+0.5, -0.5, 0),
        float3(-0.5, +0.5, 0),
        float3(+0.5, +0.5, 0),
    };

    int2 tileID = computeTileID(chosenPixel, TILE_SIZE);
    int2 tileResolution = computeTileResolution(int2(1280, 720), TILE_SIZE);
    int tileIndex = tileID.y * tileResolution.x + tileID.x;

    const int vplId = u_SelectedVPL[tileIndex][assembledVertex.instanceId];
    if (vplId == -1) {
        VertexStageOutput output;
        output.sv_position = float4(-10.f, -10.f, -10.f, -10.f);
    }

    const VPLData vpl = u_VPLData[vplId];
    const float3 center = (vpl.getAABBMin() + vpl.getAABBMax()) * 0.5f;
    const float3 variance = vpl.getVariance();
    const float3 normal = vpl.getNormW();
    const float3 position = vpl.getPosW();

    //   const float2 xy = variance.xy * rect[assembledVertex.vertexId].xy;
    const float3x3 R = getRotationMatrixFromAToB(normal, float3(0.f, 0.f, 1.f));
    const float ex = vpl.ex;
    const float ey = vpl.ey;
        
    const float2 xy = gVPLRenderScale * float2(ex, ey) * rect[assembledVertex.vertexId].xy;
    const float3 positionWS = position + R[0] * xy.x + R[1] * xy.y;
    const float4 positionCS = mul(float4(positionWS, 1.0f), gCamera.viewProjMat);

    const float3 viewDirection = positionWS - gCamera.posW;
    const bool isBackFace = dot(viewDirection, vpl.getNormW()) > 0.0f;

    VertexStageOutput output;
    output.coarseVertex.vplId = int(vplId);
    output.coarseVertex.isBackFace = isBackFace ? 1 : 0;
    output.sv_position = positionCS;
    return output;
}

[shader("fragment")]
float4 fragmentMain(VertexStageOutput vsOut) : SV_TARGET {
  const int vplId = vsOut.coarseVertex.vplId;
  const VPLData vpl = u_VPLData[vplId];
  const float3 color = (vsOut.coarseVertex.isBackFace == 1) 
    ? float3(0.5f, 0.0f, 0.5f) : float3(vpl.getColor() * gVPLColorScale);
  return float4(color, 1.0f);
}
