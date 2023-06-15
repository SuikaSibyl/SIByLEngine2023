struct GeometryInfo
{
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

struct InterleavedVertex
{
    float3 position;
    float3 normal;
    float3 tangent;
    float2 texCoords;
};

struct PushConstants
{
    uint geometry_idx;
};

struct CameraData
{
    row_major float4x4 viewMat;
    row_major float4x4 prevViewMat;
    row_major float4x4 projMat;
    row_major float4x4 viewProjMat;
    row_major float4x4 invViewProj;
    row_major float4x4 viewProjMatNoJitter;
    row_major float4x4 prevViewProjMatNoJitter;
    row_major float4x4 projMatNoJitter;
    float3 posW;
    float focalLength;
    float3 prevPosW;
    float _padding0;
    float3 up;
    float aspectRatio;
    float3 target;
    float nearZ;
    float3 cameraU;
    float farZ;
    float3 cameraV;
    float jitterX;
    float3 cameraW;
    float jitterY;
    float frameHeight;
    float frameWidth;
    float focalDistance;
    float apertureRadius;
    float shutterSpeed;
    float ISOSpeed;
    float _padding1;
    float _padding2;
};

struct GlobalUniforms
{
    CameraData cameraData;
};

struct MaterialData
{
    float4 data_pack_0;
    float4 data_pack_1;
    float4 data_pack_2;
    float3 data_pack_3;
    uint bsdf_type;
};

struct LightData
{
    uint lightType;
    float3 intensity;
    uint index;
    uint sample_dist_size_0;
    uint sample_dist_offset_pmf_0;
    uint sample_dist_offset_cdf_0;
    float pmf;
    uint sample_dist_size_1;
    uint sample_dist_offset_pmf_1;
    uint sample_dist_offset_cdf_1;
};

struct SceneInfoUniforms
{
    uint light_num;
    uint light_offset_pmf;
    uint light_offset_cdf;
    uint env_map;
};

RWByteAddressBuffer _100 : register(u1, space0);
RWByteAddressBuffer _122 : register(u3, space0);
cbuffer _GlobalUniforms : register(b0, space0)
{
    GlobalUniforms _158_globalUniform : packoffset(c0);
};

RWByteAddressBuffer _213 : register(u2, space0);
RWByteAddressBuffer _218 : register(u4, space0);
RWByteAddressBuffer _223 : register(u5, space0);
RWByteAddressBuffer _227 : register(u6, space0);
cbuffer _SceneInfoBuffer : register(b7, space0)
{
    SceneInfoUniforms _231_sceneInfoUniform : packoffset(c0);
};

cbuffer PushConsts
{
    PushConstants _126_pushConstants : packoffset(c0);
};

Texture2D<float4> textures[1] : register(t8, space0);
SamplerState _textures_sampler[1] : register(s8, space0);
TextureCube<float4> textures_cube[1] : register(t8, space0);
SamplerState _textures_cube_sampler[1] : register(s8, space0);

static float4 gl_Position;
static int gl_VertexIndex;
static float2 uv;
static uint matID;
static float3 normalWS;
static float4 tangentWS;

struct SPIRV_Cross_Input
{
    uint gl_VertexIndex : SV_VertexID;
};

struct SPIRV_Cross_Output
{
    float2 uv : TEXCOORD0;
    uint matID : TEXCOORD1;
    float3 normalWS : TEXCOORD2;
    float4 tangentWS : TEXCOORD3;
    float4 gl_Position : SV_Position;
};

InterleavedVertex fetchVertex()
{
    InterleavedVertex _106;
    _106.position = asfloat(_100.Load3(gl_VertexIndex * 44 + 0));
    _106.normal = asfloat(_100.Load3(gl_VertexIndex * 44 + 12));
    _106.tangent = asfloat(_100.Load3(gl_VertexIndex * 44 + 24));
    _106.texCoords = asfloat(_100.Load2(gl_VertexIndex * 44 + 36));
    InterleavedVertex _109;
    _109.position = _106.position;
    _109.normal = _106.normal;
    _109.tangent = _106.tangent;
    _109.texCoords = _106.texCoords;
    InterleavedVertex _108 = _109;
    return _108;
}

float4x4 ObjectToWorld(GeometryInfo geometry)
{
    return transpose(float4x4(float4(geometry.transform[0]), float4(geometry.transform[1]), float4(geometry.transform[2]), float4(0.0f, 0.0f, 0.0f, 1.0f)));
}

float4x4 ObjectToWorldNormal(GeometryInfo geometry)
{
    return float4x4(float4(geometry.transformInverse[0]), float4(geometry.transformInverse[1]), float4(geometry.transformInverse[2]), float4(0.0f, 0.0f, 0.0f, 1.0f));
}

void vert_main()
{
    InterleavedVertex vertex = fetchVertex();
    GeometryInfo _132;
    _132.vertexOffset = _122.Load(_126_pushConstants.geometry_idx * 128 + 0);
    _132.indexOffset = _122.Load(_126_pushConstants.geometry_idx * 128 + 4);
    _132.materialID = _122.Load(_126_pushConstants.geometry_idx * 128 + 8);
    _132.indexSize = _122.Load(_126_pushConstants.geometry_idx * 128 + 12);
    _132.surfaceArea = asfloat(_122.Load(_126_pushConstants.geometry_idx * 128 + 16));
    _132.lightID = _122.Load(_126_pushConstants.geometry_idx * 128 + 20);
    _132.primitiveType = _122.Load(_126_pushConstants.geometry_idx * 128 + 24);
    _132.oddNegativeScaling = asfloat(_122.Load(_126_pushConstants.geometry_idx * 128 + 28));
    [unroll]
    for (int _0ident = 0; _0ident < 3; _0ident++)
    {
        _132.transform[_0ident] = asfloat(_122.Load4(_0ident * 16 + _126_pushConstants.geometry_idx * 128 + 32));
    }
    [unroll]
    for (int _1ident = 0; _1ident < 3; _1ident++)
    {
        _132.transformInverse[_1ident] = asfloat(_122.Load4(_1ident * 16 + _126_pushConstants.geometry_idx * 128 + 80));
    }
    GeometryInfo _133;
    _133.vertexOffset = _132.vertexOffset;
    _133.indexOffset = _132.indexOffset;
    _133.materialID = _132.materialID;
    _133.indexSize = _132.indexSize;
    _133.surfaceArea = _132.surfaceArea;
    _133.lightID = _132.lightID;
    _133.primitiveType = _132.primitiveType;
    _133.oddNegativeScaling = _132.oddNegativeScaling;
    _133.transform[0] = _132.transform[0];
    _133.transform[1] = _132.transform[1];
    _133.transform[2] = _132.transform[2];
    _133.transformInverse[0] = _132.transformInverse[0];
    _133.transformInverse[1] = _132.transformInverse[1];
    _133.transformInverse[2] = _132.transformInverse[2];
    GeometryInfo geometry = _133;
    GeometryInfo param = geometry;
    float4x4 o2w = ObjectToWorld(param);
    float4 positionWorld = mul(float4(vertex.position, 1.0f), o2w);
    gl_Position = mul(positionWorld, _158_globalUniform.cameraData.viewProjMat);
    uv = vertex.texCoords;
    matID = geometry.materialID;
    GeometryInfo param_1 = geometry;
    normalWS = normalize(mul(float4(vertex.normal, 0.0f), ObjectToWorldNormal(param_1)).xyz);
    tangentWS = float4(normalize(mul(float4(vertex.tangent, 0.0f), o2w).xyz), geometry.oddNegativeScaling);
}

SPIRV_Cross_Output main(SPIRV_Cross_Input stage_input)
{
    gl_VertexIndex = int(stage_input.gl_VertexIndex);
    vert_main();
    SPIRV_Cross_Output stage_output;
    stage_output.gl_Position = gl_Position;
    stage_output.uv = uv;
    stage_output.matID = matID;
    stage_output.normalWS = normalWS;
    stage_output.tangentWS = tangentWS;
    return stage_output;
}
