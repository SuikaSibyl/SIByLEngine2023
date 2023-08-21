#include "../../include/common/geometry.hlsli"
#include "../../include/common/camera.hlsli"
#include "seminee_interface.hlsli"

cbuffer CameraBuffer {
    CameraData gCamera;
};

[[vk::push_constant]]
cbuffer PushConstantBuffer {
    int vplIdOffset;
   uint packedSetting;
  float gVPLRenderScale;
  float gVPLColorScale;
}

const RWStructuredBuffer<TreeNode> u_Nodes;
const RWStructuredBuffer<VPLData> u_VPLData;
// const RWStructuredBuffer<VPLMerge> u_Merge;
const RWStructuredBuffer<TreeConstrIndirectArgs> u_ConstrIndirectArgs;

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

    const int nodeId = vplIdOffset + assembledVertex.instanceId;

  TreeNode node = u_Nodes[nodeId];
//   if (node.parent_idx != 0xFFFFFFFF) {
//       const TreeNode parent_node = u_Nodes[node.parent_idx];
//       const VPLData parent_vpl = u_VPLData[parent_node.vpl_idx];
//       if (parent_vpl.getEarlyStop() != 0.f) {
//           VertexStageOutput output;
//           output.sv_position = float4(-10.f, -10.f, -10.f, -10.f);
//           return output;
//       }
//   }

  const VPLData vpl = u_VPLData[node.vpl_idx];
  if (vpl.isInvalid() || vpl.getEarlyStop() == 0.f) {
      VertexStageOutput output;
      output.sv_position = float4(-10.f, -10.f, -10.f, -10.f);
      return output;
  }

  if(node.left_idx == 0xFFFFFFFF && node.right_idx == 0xFFFFFFFF) {
      VertexStageOutput output;
      output.sv_position = float4(-10.f, -10.f, -10.f, -10.f);
      return output;
  }

  if(node.parent_idx != 0xFFFFFFFF) {
      const TreeNode parent_node = u_Nodes[node.parent_idx];
      const VPLData parent_vpl = u_VPLData[parent_node.vpl_idx];
      if(parent_vpl.getEarlyStop() != 0.f) {
          VertexStageOutput output;
          output.sv_position = float4(-10.f, -10.f, -10.f, -10.f);
          return output;
      }
      //   if(parent_node.left_idx == nodeId || parent_node.right_idx == nodeId) {
    //       VertexStageOutput output;
    //       output.sv_position = float4(-10.f, -10.f, -10.f, -10.f);
    //       return output;
    //   }
  }



  const float3 position = vpl.getPosW();
  const float3 variance = vpl.getVariance();
  const float3 normal = vpl.getNormW();

  //   const float2 xy = variance.xy * rect[assembledVertex.vertexId].xy;
  const float2 xy = gVPLRenderScale * variance.xy * rect[assembledVertex.vertexId].xy;
  const float3x3 R = getRotationMatrixFromAToB(normal, float3(0.f, 0.f, 1.f));
  const float3 positionWS = position + R[0] * xy.x + R[1] * xy.y;
  const float4 positionCS = mul(float4(positionWS, 1.0f), gCamera.viewProjMat);

  const float3 viewDirection = positionWS - gCamera.posW;
  const bool isBackFace = dot(viewDirection, vpl.getNormW()) > 0.0f;

  VertexStageOutput output;
  output.coarseVertex.vplId = int(node.vpl_idx);
  output.coarseVertex.isBackFace = isBackFace ? 1 : 0;
  output.sv_position = positionCS;
  return output;
}

[shader("fragment")]
float4 fragmentMain(VertexStageOutput vsOut) : SV_TARGET {
  const int vplId = vsOut.coarseVertex.vplId;
  const VPLData vpl = u_VPLData[vplId];
//   const float3 color = float3(vpl.getColor() * gVPLColorScale);
  const float3 color = (vsOut.coarseVertex.isBackFace == 1) 
    ? float3(0.5f, 0.0f, 0.5f) : float3(vpl.getColor() * gVPLColorScale);
  return float4(color, 1.0f);
}
