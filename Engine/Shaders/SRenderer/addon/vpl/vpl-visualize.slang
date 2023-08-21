#include "../../include/common/geometry.hlsli"
#include "../../include/common/camera.hlsli"

cbuffer CameraBuffer {
    CameraData gCamera;
};

[[vk::push_constant]]
cbuffer PushConstantBuffer {
  float gVPLRenderScale;
  float gVPLColorScale;
}

ByteAddressBuffer u_vplCounter;
StructuredBuffer<float4> u_vplPositions;
StructuredBuffer<float4> u_vplNormals;
StructuredBuffer<float4> u_vplColors;

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
  const float3 sphere[30] = {
      // triangle 0
      float3(0, -0.5, 0),
      float3(0, -0.5, 0),
      float3(0, 0.5, 0),
      // triangle 1
      float3(0, -0.5, 0),
      float3(0, 0.5, 0),
      float3(0, 0.5, 0),
      // triangle 2
      float3(0, 0, 0),
      float3(3.061616997868383e-17, 0.5, 0),
      float3(-0.35355339059327373, 0.3535533905932738, 0),
      // triangle 3
      float3(0, 0, 0),
      float3(-0.35355339059327373, 0.3535533905932738, 0),
      float3(-0.5, 6.123233995736766e-17, 0),
      // triangle 4
      float3(0, 0, 0),
      float3(-0.5, 6.123233995736766e-17, 0),
      float3(-0.35355339059327384, -0.35355339059327373, 0),
      // triangle 5
      float3(0, 0, 0),
      float3(-0.35355339059327384, -0.35355339059327373, 0),
      float3(-9.184850993605148e-17, -0.5, 0),
      // triangle 6
      float3(0, 0, 0),
      float3(-9.184850993605148e-17, -0.5, 0),
      float3(0.3535533905932737, -0.35355339059327384, 0),
      // triangle 7
      float3(0, 0, 0),
      float3(0.3535533905932737, -0.35355339059327384, 0),
      float3(0.5, -1.2246467991473532e-16, 0),
      // triangle 8
      float3(0, 0, 0),
      float3(0.5, -1.2246467991473532e-16, 0),
      float3(0.35355339059327384, 0.3535533905932737, 0),
      // triangle 9
      float3(0, 0, 0),
      float3(0.35355339059327384, 0.3535533905932737, 0),
      float3(1.5308084989341916e-16, 0.5, 0)
  };

  const int vplId = assembledVertex.instanceId;
  const int maxID = int(u_vplCounter.Load(0));

  // Degenerate vertex if vplId is out of range
  if (vplId >= maxID) {
      VertexStageOutput degenerate_output;
      degenerate_output.sv_position = float4(0.0f, 0.0f, 0.0f, 0.0f);
      return degenerate_output;
  }

  const float3 normWS = u_vplNormals[vplId].xyz;
  const float3 posWS = u_vplPositions[vplId].xyz;

  const float3x3 obn = createONB(normWS);
  const float3 positionOS = to_world(obn, sphere[assembledVertex.vertexId] * gVPLRenderScale);
  const float3 positionWS = positionOS + posWS;
  const float4 positionCS = mul(float4(positionWS, 1.0f), gCamera.viewProjMat);

  const float3 viewDirection = positionWS - gCamera.posW;
  const bool isBackFace = dot(viewDirection, normWS) > 0.0f;
  
  VertexStageOutput output;
  output.coarseVertex.vplId = vplId;
  output.coarseVertex.isBackFace = isBackFace ? 1 : 0;
  output.sv_position = positionCS;
  return output;
}

[shader("fragment")]
float4 fragmentMain(VertexStageOutput vsOut) : SV_TARGET {
  const int vplId = vsOut.coarseVertex.vplId;
  const float3 vplColor = u_vplColors[vplId].xyz * gVPLColorScale;
  const float3 color = (vsOut.coarseVertex.isBackFace == 1)
                           ? float3(0.5f, 0.0f, 0.5f) : vplColor;
  return float4(color, 1.0f);
}
