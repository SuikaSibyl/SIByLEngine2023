#ifndef _SRENDERER_COMMON_DESCRIPOTR_SET_HEADER_
#define _SRENDERER_COMMON_DESCRIPOTR_SET_HEADER_

#include "common/camera.hlsli"
#include "common/light.hlsli"
#include "common/shading.hlsli"

// Global uniforms carrying camera information.
struct GlobalUniforms {
    CameraData cameraData;
};

// interleaved vertex layout
struct InterleavedVertex {
    float3 position;
    float3 normal;
    float3 tangent;
    float2 texCoords;
};

// geometry info
struct GeometryInfo {
    uint vertexOffset;
    uint indexOffset;
    uint materialID;
    uint indexSize;
    float surfaceArea;
    uint lightID;
    uint primitiveType;
    float oddNegativeScaling;
    float4 transform[3];
    float4 transformInverse[3];
};

struct MaterialInfo {
    float3 baseOrDiffuseColor;
    int flags;
    float3 specularColor;
    int bsdfID;
    float3 emissiveColor;
    int domain;

    float opacity;
    float roughness;
    float metalness;
    float normalTextureScale;

    float occlusionStrength;
    float alphaCutoff;
    float transmissionFactor;
    int baseOrDiffuseTextureIndex;

    int metalRoughOrSpecularTextureIndex;
    int emissiveTextureIndex;
    int normalTextureIndex;
    int occlusionTextureIndex;

    int transmissionTextureIndex;
    int padding1;
    int padding2;
    int padding3;
};

struct SceneInfoUniforms {
    uint light_num;
    uint light_offset_pmf;
    uint light_offset_cdf;
    uint env_map;
};

[[vk::binding(0, 0)]]
cbuffer _GlobalUniforms {
    GlobalUniforms globalUniform;
};
[[vk::binding(1, 0)]] ByteAddressBuffer vertices;
[[vk::binding(2, 0)]] StructuredBuffer<uint32_t> indices;
[[vk::binding(3, 0)]] StructuredBuffer<GeometryInfo> geometries;
[[vk::binding(4, 0)]] StructuredBuffer<MaterialInfo> materials;
[[vk::binding(5, 0)]] StructuredBuffer<PolymorphicLightInfo> lights;
[[vk::binding(6, 0)]] StructuredBuffer<float> sampleDistDatas;

[[vk::binding(7, 0)]]
cbuffer _SceneInfoBuffer {
    SceneInfoUniforms sceneInfoUniform;
};

[[vk::binding(8, 0)]] Sampler2D textures[];

// previous frame binding
cbuffer PrevGlobalUniforms {
    GlobalUniforms prevGlobalUniform;
};
StructuredBuffer<GeometryInfo> prevGeometries;

static const int kVertexSize = 44;
static const int kVertexPositionOffset = 0;
static const int kVertexNormalOffset   = 12;
static const int kVertexTangentOffset  = 24;
static const int kVertexTexCoordOffset = 36;

// layout(binding = 8, set = 0) uniform sampler2D textures[];
// layout(binding = 8, set = 0) uniform samplerCube textures_cube[];

float3 fetchVertexPosition(int vertexIndex) {
    return vertices.Load<float3>(
        vertexIndex * kVertexSize
        + kVertexPositionOffset);
}
float3 fetchVertexNormal(int vertexIndex) {
    return vertices.Load<float3>(
        vertexIndex * kVertexSize
        + kVertexNormalOffset);
}
float3 fetchVertexTangent(int vertexIndex) {
    return vertices.Load<float3>(
        vertexIndex * kVertexSize
        + kVertexTangentOffset);
}
float2 fetchVertexTexCoord(int vertexIndex) {
    return vertices.Load<float2>(
        vertexIndex * kVertexSize
        + kVertexTexCoordOffset);
}
uint3 fetchTriangleIndices(in_ref(GeometryInfo) geoInfo, int triangleIndex) {
    return uint3(indices[geoInfo.indexOffset + triangleIndex * 3 + 0],
                 indices[geoInfo.indexOffset + triangleIndex * 3 + 1],
                 indices[geoInfo.indexOffset + triangleIndex * 3 + 2]);
}

float4x4 ObjectToWorld(in GeometryInfo geometry) {
    return transpose(float4x4(geometry.transform[0], geometry.transform[1], geometry.transform[2], float4(0, 0, 0, 1)));
}

float4x4 WorldToObject(in GeometryInfo geometry) {
    return transpose(float4x4(geometry.transformInverse[0], geometry.transformInverse[1], geometry.transformInverse[2], float4(0, 0, 0, 1)));
}

float4x4 ObjectToWorldNormal(in GeometryInfo geometry) {
    return float4x4(geometry.transformInverse[0], geometry.transformInverse[1], geometry.transformInverse[2], float4(0, 0, 0, 1));
}

ShadingSurface fetchShadingSurfaceTrimesh(
    in_ref(GeometryInfo) geometry,
    in_ref(float3) bary,
    int primitiveID)
{
    const int3 index = int3(fetchTriangleIndices(geometry, primitiveID));
    const MaterialInfo material = materials[geometry.materialID];

    float3 vertexPositions[3];
    vertexPositions[0] = fetchVertexPosition(index[0] + int(geometry.vertexOffset));
    vertexPositions[1] = fetchVertexPosition(index[1] + int(geometry.vertexOffset));
    vertexPositions[2] = fetchVertexPosition(index[2] + int(geometry.vertexOffset));

    const float4x4 o2w = ObjectToWorld(geometry);
    const float3 positionOS = interpolate(vertexPositions, bary);
    const float3 positionWS = mul(float4(positionOS, 1.0), o2w).xyz;

    float2 vertexUVs[3];
    vertexUVs[0] = fetchVertexTexCoord(index[0] + int(geometry.vertexOffset));
    vertexUVs[1] = fetchVertexTexCoord(index[1] + int(geometry.vertexOffset));
    vertexUVs[2] = fetchVertexTexCoord(index[2] + int(geometry.vertexOffset));
    float2 uv = interpolate(vertexUVs, bary);

    float3 objectSpaceFlatNormal = normalize(cross(
        vertexPositions[1] - vertexPositions[0],
        vertexPositions[2] - vertexPositions[0]));

    float4x4 o2wn = ObjectToWorldNormal(geometry);
    const float3 flatNormal = normalize(mul(float4(objectSpaceFlatNormal, 0.0), o2wn).xyz);

    float3 normals[3];
    normals[0] = fetchVertexNormal(index[0] + int(geometry.vertexOffset));
    normals[1] = fetchVertexNormal(index[1] + int(geometry.vertexOffset));
    normals[2] = fetchVertexNormal(index[2] + int(geometry.vertexOffset));
    float3 geometryNormalOS = interpolate(normals, bary);
    float3 geometryNormalWS = normalize(mul(float4(geometryNormalOS, 0.0), o2wn).xyz);

    float3 tangents[3];
    tangents[0] = fetchVertexTangent(index[0] + int(geometry.vertexOffset));
    tangents[1] = fetchVertexTangent(index[1] + int(geometry.vertexOffset));
    tangents[2] = fetchVertexTangent(index[2] + int(geometry.vertexOffset));
    float3 tangentOS = interpolate(tangents, bary);
    float4 tangentWS = float4(normalize(mul(float4(tangentOS, 0), o2w).xyz), geometry.oddNegativeScaling);

    float3 shadingNormal = geometryNormalWS;
    float3 bitangent = cross(geometryNormalWS, tangentWS.xyz) * tangentWS.w;
    float3x3 TBN = float3x3(tangentWS.xyz, bitangent, geometryNormalWS);
    float3 normalMap = float3(0.5, 0.5, 1.0);
    if (material.normalTextureIndex != -1) {
        normalMap = textures[material.normalTextureIndex].Sample(uv).rgb;
        shadingNormal = normalize(mul((normalMap * 2 - 1), TBN));
    }

    // Grab material infomation
    float3 diffuseColor = textures[material.baseOrDiffuseTextureIndex].Sample(uv).rgb;

    ShadingSurface surface;
    surface.worldPos = positionWS;
    surface.roughness;
    surface.viewDir;
    surface.viewDepth;
    surface.normal = shadingNormal;
    surface.diffuseProbability;
    surface.geoNormal = flatNormal;
    surface.diffuseAlbedo = diffuseColor;
    surface.specularF0;
    return surface;
}

#endif // !_SRENDERER_COMMON_DESCRIPOTR_SET_HEADER_