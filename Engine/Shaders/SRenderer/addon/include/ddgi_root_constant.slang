

// RTXGI_PUSH_CONSTS_TYPE may be passed in as a define at shader compilation time.
// This define specifies how the shader will reference the push constants data block.
// If not using DDGI push constants, this define can be ignored.

#define RTXGI_PUSH_CONSTS_TYPE_SDK 1
#define RTXGI_PUSH_CONSTS_TYPE_APPLICATION 2

#if RTXGI_PUSH_CONSTS_TYPE == RTXGI_PUSH_CONSTS_TYPE_APPLICATION

// Note: Vulkan only allows a single block of memory for push constants. When using an
// application's pipeline layout in RTXGI shaders, the RTXGI shaders must understand
// the organization of the application's push constants data block!

// RTXGI_PUSH_CONSTS_VARIABLE_NAME must be passed in as a define at shader compilation time.
// This define specifies the variable name of the push constants block.
#ifndef RTXGI_PUSH_CONSTS_VARIABLE_NAME
#error Required define RTXGI_PUSH_CONSTS_VARIABLE_NAME is not defined!
#endif

// RTXGI_PUSH_CONSTS_FIELD_DDGI_VOLUME_INDEX_NAME must be passed in as a define at shader compilation time.
// This define specifies the name of the volume index field in the push constants struct.
#ifndef RTXGI_PUSH_CONSTS_FIELD_DDGI_VOLUME_INDEX_NAME
#error Required define RTXGI_PUSH_CONSTS_FIELD_DDGI_VOLUME_INDEX_NAME is not defined!
#endif

#if RTXGI_DECLARE_PUSH_CONSTS

// RTXGI_PUSH_CONSTS_STRUCT_NAME must be passed in as a define at shader compilation time.
// This define specifies the name of the push constants type struct.
#ifndef RTXGI_PUSH_CONSTS_STRUCT_NAME
#error Required define RTXGI_PUSH_CONSTS_STRUCT_NAME is not defined!
#endif

struct RTXGI_PUSH_CONSTS_STRUCT_NAME {
    // IMPORTANT: insert padding to match the layout of your push constants!
    // The padding below matches the size of the Test Harness' "GlobalConstants" struct
    // with 48 float values before the DDGIRootConstants (see test-harness/include/graphics/Types.h)
    float4x4 padding0;
    float4x4 padding1;
    float4x4 padding2;
    uint RTXGI_PUSH_CONSTS_FIELD_DDGI_VOLUME_INDEX_NAME;
    uint2 ddgi_pad0;
    uint RTXGI_PUSH_CONSTS_FIELD_DDGI_REDUCTION_INPUT_SIZE_X_NAME;
    uint RTXGI_PUSH_CONSTS_FIELD_DDGI_REDUCTION_INPUT_SIZE_Y_NAME;
    uint RTXGI_PUSH_CONSTS_FIELD_DDGI_REDUCTION_INPUT_SIZE_Z_NAME;
    uint2 ddgi_pad1;
};
[[vk::push_constant]] RTXGI_PUSH_CONSTS_STRUCT_NAME RTXGI_PUSH_CONSTS_VARIABLE_NAME;
#endif

uint GetDDGIVolumeIndex() { return RTXGI_PUSH_CONSTS_VARIABLE_NAME.RTXGI_PUSH_CONSTS_FIELD_DDGI_VOLUME_INDEX_NAME; }
uint3 GetReductionInputSize()
{
    return uint3(RTXGI_PUSH_CONSTS_VARIABLE_NAME.RTXGI_PUSH_CONSTS_FIELD_DDGI_REDUCTION_INPUT_SIZE_X_NAME,
                 RTXGI_PUSH_CONSTS_VARIABLE_NAME.RTXGI_PUSH_CONSTS_FIELD_DDGI_REDUCTION_INPUT_SIZE_Y_NAME,
                 RTXGI_PUSH_CONSTS_VARIABLE_NAME.RTXGI_PUSH_CONSTS_FIELD_DDGI_REDUCTION_INPUT_SIZE_Z_NAME);
}

#elif RTXGI_PUSH_CONSTS_TYPE == RTXGI_PUSH_CONSTS_TYPE_SDK

[[vk::push_constant]] ConstantBuffer<DDGIRootConstants> DDGI;
uint GetDDGIVolumeIndex() { return DDGI.volumeIndex; }
uint3 GetReductionInputSize() { return uint3(DDGI.reductionInputSizeX, DDGI.reductionInputSizeY, DDGI.reductionInputSizeZ); }

#endif // RTXGI_PUSH_CONSTS_TYPE

// These functions are not relevant in Vulkan since descriptor heap style bindless is not available
uint GetDDGIVolumeConstantsIndex() { return 0; }
uint GetDDGIVolumeResourceIndicesIndex() { return 0; }
