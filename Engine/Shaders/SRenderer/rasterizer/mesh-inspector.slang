#include "../include/common/geometry.hlsli"
#include "../include/common/random.hlsli"
#include "../include/common/math.hlsli"
#include "../include/common/packing.hlsli"
#include "../include/scene_descriptor_set.hlsli"

[[vk::push_constant]]
cbuffer PushConstants {
    struct PushConstants {
        uint geometry_idx;
    } pConst;
};

enum ShowEnum {
    BaseColor,
    Metalness,
    Anisotropy,
    Roughness,
    FlatNormal,
    GeometryNormal,
    ShadingNormal,
    NormalMap,
    Opacity,
    Emission,
    SpecularF0,
    VertexColor,
    Matcap,
    MatcapSurface,
    VertexNormal,
    UVChecker,
}

struct GeoVisUniform {
    // show setting
    ShowEnum showEnum;
    int uv_checker_resource;
    int matcap_resource;
    int padding;
    // wireframe settings
    float3 wireframe_color;
    uint use_wireframe;
    float wireframe_smoothing;
    float wireframe_thickness;
};

[[vk::binding(0, 1)]]
cbuffer _GeoVisUniform {
    GeoVisUniform uni;
}

[[vk::binding(1, 1)]]
Sampler2D<float4> matcapSampler;

float3 GetAlbedoWithWireframe(in const float3 color,
                              float3 barys) {
    // float3 albedo = GetAlbedo(i);
    const float3 deltas = fwidth(barys);
    const float3 smoothing = deltas * uni.wireframe_smoothing;
    const float3 thickness = deltas * uni.wireframe_thickness;
    barys = smoothstep(thickness, thickness + smoothing, barys);
    float minBary = min(barys.x, min(barys.y, barys.z));
    return lerp(uni.wireframe_color, color, minBary);
}

float2 getMatcapUV(float3 posVS, float3 normVS, bool isFrontFace) {
    const float3 eyeVector = -normalize(posVS);
    const float3 frontNormal = normalize(isFrontFace ? normVS : -normVS);
    const float3 materialNormal = frontNormal;
    const float3 nm_x = float3(-eyeVector.z, 0.0, eyeVector.x);
    const float3 nm_y = cross(nm_x, eyeVector);
    const float2 uvMatcap = float2(dot(materialNormal.xz, -nm_x.xz), dot(materialNormal, nm_y)) * 0.5 + 0.5;
    return uvMatcap;
}

struct AssembledVertex { int vertexId : SV_VertexId; };
struct CoarseVertex {
    uint geoID;
};
struct VertexStageOutput {
    CoarseVertex coarseVertex : CoarseVertex;
    float4 sv_position : SV_Position;
};

[shader("vertex")]
VertexStageOutput vertexMain(
    AssembledVertex assembledVertex)
{
    const GeometryInfo geometry = geometries[pConst.geometry_idx];
    const float4x4 o2w = ObjectToWorld(geometry);

    const float3 positionOS = fetchVertexPosition(assembledVertex.vertexId);
    const float3 positionWS = mul(float4(positionOS, 1.0), o2w).xyz;
    const float4 positionCS = mul(float4(positionWS, 1.0f), globalUniform.cameraData.viewProjMat);

    VertexStageOutput output;
    output.coarseVertex.geoID = pConst.geometry_idx;
    output.sv_position = positionCS;
    return output;
}

[shader("fragment")]
float4 fragmentMain(
    in CoarseVertex coarseVertex: CoarseVertex,
    nointerpolation in int primitiveID: SV_PrimitiveID,
    in bool isFrontFace: SV_IsFrontFace,
    in float3 bary: SV_Barycentrics) : SV_Target
{
    const GeometryInfo geometry = geometries[coarseVertex.geoID];
    const int3 index = int3(fetchTriangleIndices(geometry, primitiveID));

    MaterialInfo material = materials[geometry.materialID];

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
    float3 flatNormal = normalize(mul(float4(objectSpaceFlatNormal, 0.0), o2wn).xyz);
    const float3 viewDirection = positionWS - globalUniform.cameraData.posW;
    bool faceforward = dot(flatNormal, viewDirection) < 0;
    flatNormal = faceforward ? flatNormal : -flatNormal;
    
    float3 normals[3];
    normals[0] = fetchVertexNormal(index[0] + int(geometry.vertexOffset));
    normals[1] = fetchVertexNormal(index[1] + int(geometry.vertexOffset));
    normals[2] = fetchVertexNormal(index[2] + int(geometry.vertexOffset));
    float3 geometryNormalOS = interpolate(normals, bary);
    float3 geometryNormalWS = normalize(mul(float4(geometryNormalOS, 0.0), o2wn).xyz);
    geometryNormalWS = faceforward ? geometryNormalWS : -geometryNormalWS;

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
        shadingNormal = normalize(mul((normalMap*2 - 1), TBN));
    }
    
    // float3 worldSpacePosition = mul(gs.instance.transform, float4(gs.objectSpacePosition, 1.0)).xyz;
    // gs.flatNormal = normalize(cross(ddy(worldSpacePosition), ddx(worldSpacePosition)));

    float3 color = float3(0);
    if (uni.showEnum == ShowEnum::BaseColor) {
        color = textures[material.baseOrDiffuseTextureIndex].Sample(uv).rgb;
    }
    else if (uni.showEnum == ShowEnum::FlatNormal) {
        color = (flatNormal.rgb + 1.0) * 0.5;
    }
    else if (uni.showEnum == ShowEnum::GeometryNormal) {
        color = (geometryNormalWS + 1.0) * 0.5;
    }
    else if (uni.showEnum == ShowEnum::ShadingNormal) {
        color = (shadingNormal + 1.0) * 0.5;
    }
    else if (uni.showEnum == ShowEnum::NormalMap) {
        color = normalMap;
    }
    else if (uni.showEnum == ShowEnum::Matcap) {
        const float3 geometryNormalVS = mul((globalUniform.cameraData.invViewMat), float4(geometryNormalWS, 0.0)).xyz;

        const float2 matcapUV = getMatcapUV(viewDirection, geometryNormalVS, isFrontFace);
        color = matcapSampler.Sample(matcapUV).rgb;
        // const float3 diffuse = sRGBToLinear(texture(Texture0, uvMatcap).rgb) * getCurvature(materialNormal);
        // const float3 frag = diffuse;
    }
    else {
        uint colorPacked = Hash_murmur(primitiveID);
        color = UNPACK_FLOAT3_UINT(colorPacked);
    }
    
    // vec3 localVertex = Vertex.xyz;
    // localVertex += Vertex.w;
    // localVertex = localVertex * uQVS * vec3(5.693428104390918e+0, 1.1196796730479215e+0, 6.345989923538313e+0).zxy + uQVT;
    // vec3 localNormal = Normal.xyy;
    // localNormal.z = dot(localNormal.xy, localNormal.xy);
    // localNormal.xyz = vec3(2.0 * localNormal.xy * sqrt(1.0 - localNormal.z), 1.0 - 2.0 * localNormal.z); ;
    // vViewNormal = uModelViewNormalMatrix * localNormal;
    // vViewVertex = uModelViewMatrix * vec4(localVertex, 1.0);
    // mat4 jitteredProjection = uProjectionMatrix;
    // jitteredProjection[2].xy += (1.0 - uDisplay2D) * (uHalton.xy * uGlobalTexRatio.xy / uGlobalTexSize.xy);
    // gl_Position = jitteredProjection * vViewVertex;
    // gl_PointSize = min(64.0, max(1.0, -uPointSize / vViewVertex.z));

    // GeometrySample gs = getGeometryFromHit(g_Instance.instance, g_Instance.geometryIndex, primitiveID, i_bary.yz,
    //                                        GeomAttr_All, t_InstanceData, t_GeometryData, t_MaterialConstants);
    // float3x3 TBN = buildTangentToWorld(coarseVertex.tangentWS, coarseVertex.normalWS);
    // if (!isFrontFace) TBN[2] = -TBN[2];


    if(uni.use_wireframe != 0) {
        color = GetAlbedoWithWireframe(color, bary);
    }

    return float4(color, 1.0);
}
