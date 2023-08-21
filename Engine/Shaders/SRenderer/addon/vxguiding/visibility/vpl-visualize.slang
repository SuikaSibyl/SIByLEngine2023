#include "../../../include/common/camera.hlsli"
#include "../../../include/common/geometry.hlsli"
#include "../../../include/common/octahedral.hlsli"
#include "../../../include/common/packing.hlsli"
#include "../../../include/common/random.hlsli"

cbuffer CameraBuffer {
    CameraData gCamera;
};

[[vk::push_constant]]
cbuffer PushConstantBuffer {
  float gVPLRenderScale;
}

const StructuredBuffer<float4> u_cluster_gathered; // 32 * (32 * 32)
const StructuredBuffer<uint> u_cluster_counter;    // 32

struct AssembledVertex {
    int vertexId : SV_VertexId;
    int instanceId : SV_InstanceID;
};
struct CoarseVertex {
    int clusterId;
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
  const int clusterVplId = vplId % 1024;
  const int clusterVplCount = u_cluster_counter[vplId / 1024];

  // Degenerate vertex if vplId is out of range
  if (clusterVplId >= clusterVplCount) {
      VertexStageOutput degenerate_output;
      degenerate_output.sv_position = float4(0.0f, 0.0f, 0.0f, 0.0f);
      return degenerate_output;
  }

  const float4 packedPosNorm = u_cluster_gathered[vplId];
  const float3 normWS = Unorm32OctahedronToUnitVector(asuint(packedPosNorm.w));
  const float3 posWS = packedPosNorm.xyz;

  const float3x3 obn = createONB(normWS);
  const float3 positionOS = to_world(obn, sphere[assembledVertex.vertexId] * gVPLRenderScale);
  const float3 positionWS = positionOS + posWS;
  const float4 positionCS = mul(float4(positionWS, 1.0f), gCamera.viewProjMat);

  const float3 viewDirection = positionWS - gCamera.posW;
  const bool isBackFace = dot(viewDirection, normWS) > 0.0f;

  VertexStageOutput output;
  output.coarseVertex.clusterId = vplId / 1024;
  output.coarseVertex.isBackFace = isBackFace ? 1 : 0;
  output.sv_position = positionCS;
  return output;
}

[shader("fragment")]
float4 fragmentMain(VertexStageOutput vsOut) : SV_TARGET {
  const int clusterId = vsOut.coarseVertex.clusterId;
  uint colorPacked = Hash_murmur(clusterId);
  const float3 vplColor = UNPACK_FLOAT3_UINT(colorPacked);

  const float3 color = (vsOut.coarseVertex.isBackFace == 1)
                           ? float3(0.5f, 0.0f, 0.5f) : vplColor;
  return float4(color, 1.0f);
}
