#version 330
#ifdef GL_ARB_shading_language_420pack
#extension GL_ARB_shading_language_420pack : require
#endif

struct SLANG_ParameterGroup_PushConstants_PushConstants_0
{
    vec2 center_0;
    vec2 diag_0;
};

struct SLANG_ParameterGroup_PushConstants_0
{
    SLANG_ParameterGroup_PushConstants_PushConstants_0 pConst_0;
};

struct CameraData_0
{
    mat4 viewMat_0;
    mat4 prevViewMat_0;
    mat4 projMat_0;
    mat4 viewProjMat_0;
    mat4 invViewProj_0;
    mat4 viewProjMatNoJitter_0;
    mat4 prevViewProjMatNoJitter_0;
    mat4 projMatNoJitter_0;
    vec3 posW_0;
    float focalLength_0;
    vec3 prevPosW_0;
    float rectArea_0;
    vec3 up_0;
    float aspectRatio_0;
    vec3 target_0;
    float nearZ_0;
    vec3 cameraU_0;
    float farZ_0;
    vec3 cameraV_0;
    float jitterX_0;
    vec3 cameraW_0;
    float jitterY_0;
    float frameHeight_0;
    float frameWidth_0;
    float focalDistance_0;
    float apertureRadius_0;
    float shutterSpeed_0;
    float ISOSpeed_0;
    float _padding1_0;
    float _padding2_0;
};

struct SLANG_ParameterGroup_GlobalUniforms_0
{
    CameraData_0 gCamera_0;
};

layout(binding = 0, std140) uniform _S3
{
    SLANG_ParameterGroup_GlobalUniforms_0 _data;
} GlobalUniforms_0;

struct _S2
{
    SLANG_ParameterGroup_PushConstants_0 _data;
};

uniform _S2 PushConstants_0;

out vec2 _S4;

void main()
{
    uint _S5 = uint(gl_VertexID);
    bool _S1;
    vec2 uvs_0[6];
    if (_S1)
    {
    }
    else
    {
        uvs_0[0] = vec2(0.0);
        uvs_0[1] = vec2(1.0, 0.0);
        uvs_0[2] = vec2(0.0, 1.0);
        uvs_0[3] = vec2(1.0, 0.0);
        uvs_0[4] = vec2(0.0, 1.0);
        uvs_0[5] = vec2(1.0);
        _S1 = true;
    }
    _S4 = uvs_0[_S5];
    gl_Position = vec4(vec3(((uvs_0[_S5] * 2.0) - vec2(1.0)) * PushConstants_0._data.pConst_0.diag_0 + PushConstants_0._data.pConst_0.center_0, -1.0), 1.0) * GlobalUniforms_0._data.gCamera_0.viewProjMat_0;
}

