#version 450
layout(column_major) uniform;
layout(column_major) buffer;

            #define DRAWID_MACRO gl_DrawID
        

#line 4 0
struct DrawIndexedIndirectEX_0
{
    uint indexCount_0;
    uint instanceCount_0;
    uint firstIndex_0;
    int vertexOffset_0;
    uint firstInstance_0;
    uint geometryID_0;
    uint padding0_0;
    uint padding1_0;
};


#line 4 1
layout(std430, binding = 10) readonly buffer StructuredBuffer_DrawIndexedIndirectEX_t_0 {
    DrawIndexedIndirectEX_0 _data[];
} indirect_draws_0;

#line 22 2
struct GeometryInfo_0
{
    uint vertexOffset_1;
    uint indexOffset_0;
    uint materialID_0;
    uint indexSize_0;
    float surfaceArea_0;
    uint lightID_0;
    uint primitiveType_0;
    float oddNegativeScaling_0;
    vec4  transform_0[3];
    vec4  transformInverse_0[3];
};


#line 74
layout(std430, binding = 3) readonly buffer StructuredBuffer_GeometryInfo_t_0 {
    GeometryInfo_0 _data[];
} geometries_0;

#line 74
layout(std430, binding = 1) readonly buffer StructuredBuffer_float_t_0 {
    float _data[];
} vertices_0;

#line 6 3
struct CameraData_0
{
    mat4x4 viewMat_0;
    mat4x4 invViewMat_0;
    mat4x4 projMat_0;
    mat4x4 invProjMat_0;
    mat4x4 viewProjMat_0;
    mat4x4 invViewProj_0;
    mat4x4 viewProjMatNoJitter_0;
    mat4x4 projMatNoJitter_0;
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
    vec2 clipToWindowScale_0;
    vec2 clipToWindowBias_0;
};


#line 9 2
struct GlobalUniforms_0
{
    CameraData_0 cameraData_0;
};


#line 71
struct SLANG_ParameterGroup_GlobalUniforms_0
{
    GlobalUniforms_0 globalUniform_0;
};


#line 71
layout(binding = 0)
layout(std140) uniform _S1
{
    GlobalUniforms_0 globalUniform_0;
}_GlobalUniforms_0;

#line 123
mat4x4 ObjectToWorld_0(GeometryInfo_0 geometry_0)
{

#line 124
    return transpose(mat4x4(geometry_0.transform_0[0], geometry_0.transform_0[1], geometry_0.transform_0[2], vec4(0.0, 0.0, 0.0, 1.0)));
}


#line 1692 4
vec3 ByteAddressBuffer_Load_0(int _S2)
{

#line 2496
    return vec3(vertices_0._data[_S2 / 4], vertices_0._data[(_S2 + 4) / 4], vertices_0._data[(_S2 + 8) / 4]);
}


#line 97 2
vec3 fetchVertexPosition_0(int vertexIndex_0)
{

#line 98
    return ByteAddressBuffer_Load_0(vertexIndex_0 * 44);
}


#line 7 1
struct VertexStageOutput_0
{
    vec4 sv_position_0;
};


#line 10
void main()
{

#line 11
    uint draw_idx_0 = DRAWID_MACRO();

#line 21
    VertexStageOutput_0 output_0;
    output_0.sv_position_0 = (((_GlobalUniforms_0.globalUniform_0.cameraData_0.viewProjMat_0) * (vec4((((ObjectToWorld_0(geometries_0._data[indirect_draws_0._data[draw_idx_0].geometryID_0])) * (vec4(fetchVertexPosition_0(gl_VertexIndex), 1.0)))).xyz, 1.0))));

#line 22
    gl_Position = output_0.sv_position_0;

#line 22
    return;
}

