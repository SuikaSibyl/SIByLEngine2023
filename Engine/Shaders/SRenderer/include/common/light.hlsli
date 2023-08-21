#ifndef _SRENDERER_LIGHT_CPP_
#define _SRENDERER_LIGHT_CPP_

#include "cpp_compatible.hlsli"
#include "packing.hlsli"

/**
* The type of the light. Enumeration up to 15,
* because the type is stored in the upper 4 bits
* of the colorTypeAndFlags field.
*/
enum_macro PolymorphicLightType{
    // singular lights
    kDirectional    = 0,
    kPoint          = 1,
    kSpot           = 2,
    // area lights
    kTriangle       = 3,
    kRectangle      = 4,
    kMeshPrimitive  = 5,
    kEnvironment    = 6,
    // virtual lights
    kVpl            = 7,
};

/**
* Stores shared light information (type) and specific light information
* 
*/
struct PolymorphicLightInfo {
    float3 center;
    // rgb8 (24 bits) color, 4 bits for type, 4 bits for flags
    uint colorTypeAndFlags;

    uint databyte0;   // geometry id / direction1(oct-encoded)
    uint databyte1;   // index id / direction2(oct-encoded)
    uint databyte2;   // scalar / 2x float16
    uint logRadiance; // uint16

    uint iesProfileIndex;
    uint primaryAxis;             // oct-encoded
    uint cosConeAngleAndSoftness; // 2x float16
    uint shadowMapIndex;          // shadow map index
};

struct PolymorphicLightSample {
    float3 position;
    float3 normal;
    float3 radiance;
    float solidAnglePdf;
};

static const uint kPolymorphicLightTypeShift = 24;
static const uint kPolymorphicLightTypeMask = 0xf;
static const uint kPolymorphicLightShapingEnableBit = 1 << 28;
static const uint kPolymorphicLightIesProfileEnableBit = 1 << 29;
static const float kPolymorphicLightMinLog2Radiance = -8.f;
static const float kPolymorphicLightMaxLog2Radiance = 40.f;

PolymorphicLightType getLightType(in const PolymorphicLightInfo lightInfo) {
    uint typeCode = (lightInfo.colorTypeAndFlags >> kPolymorphicLightTypeShift)
        & kPolymorphicLightTypeMask;
    return (PolymorphicLightType)typeCode;
}

float unpackLightRadiance(uint logRadiance) {
    return (logRadiance == 0) ? 0 : exp2((float(logRadiance - 1) / 65534.0) * 
        (kPolymorphicLightMaxLog2Radiance - kPolymorphicLightMinLog2Radiance) + kPolymorphicLightMinLog2Radiance);
}

float3 unpackLightColor(in_ref(PolymorphicLightInfo) lightInfo) {
    float3 color = Unpack_R8G8B8_UFLOAT(lightInfo.colorTypeAndFlags & 0xffffffu);
    float radiance = unpackLightRadiance(lightInfo.logRadiance & 0xffff);
    return color * float3(radiance);
}

void packLightColor(float3 radiance, inout_ref(PolymorphicLightInfo) lightInfo) {
    float intensity = max(radiance.r, max(radiance.g, radiance.b));
    if (intensity > 0.0) {
        float logRadiance = saturate((log2(intensity) - kPolymorphicLightMinLog2Radiance) 
            / (kPolymorphicLightMaxLog2Radiance - kPolymorphicLightMinLog2Radiance));
        uint packedRadiance = min(uint32_t(ceil(logRadiance * 65534.0)) + 1, 0xffffu);
        float unpackedRadiance = unpackLightRadiance(packedRadiance);

        float3 normalizedRadiance =
            saturate(radiance / float3(unpackedRadiance));

        lightInfo.logRadiance = lightInfo.logRadiance & ~0xffffu;
        lightInfo.logRadiance |= packedRadiance;
        lightInfo.colorTypeAndFlags = lightInfo.colorTypeAndFlags & ~0xffffffu;
        lightInfo.colorTypeAndFlags |= Pack_R8G8B8_UFLOAT(normalizedRadiance);
    }
}

#endif // _SRENDERER_LIGHT_CPP_