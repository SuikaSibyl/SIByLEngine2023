#ifndef _SRENDERER_LIGHT_
#define _SRENDERER_LIGHT_

#include "../common/common_packing.h"

const uint enum_polymorphic_light_type_sphere       = 0;
const uint enum_polymorphic_light_type_triangle     = 1;
const uint enum_polymorphic_light_type_directional  = 2;
const uint enum_polymorphic_light_type_envmap       = 3;

const float kPolymorphicLightMinLog2Radiance = -8.f;
const float kPolymorphicLightMaxLog2Radiance = 40.f;

// Stores shared light information (type) and specific light information
struct PolymorphicLightInfo {
    // uint4[0]
    vec3 center;
    uint colorTypeAndFlags; // RGB8 + uint8
    // uint4[1]
    uint direction1;
    uint direction2;
    uint scalars;
    uint logRadiance;               // uint16
    // uint4[2] -- optional, contains only shape data
    uint iesProfileIndex;
    uint primaryAxis;               // oct-encoded
    uint cosConeAngleAndSoftness;   // 2x float16
    uint padding;
};

float unpackLightRadiance(uint logRadiance) {
    return (logRadiance == 0) 
        ? 0 
        : exp2((float(logRadiance - 1) / 65534.0) 
            * (kPolymorphicLightMaxLog2Radiance - kPolymorphicLightMinLog2Radiance)
            + kPolymorphicLightMinLog2Radiance);
}

vec3 unpackLightColor(PolymorphicLightInfo lightInfo) {
    const vec3 color = Unpack_R8G8B8_UFLOAT(lightInfo.colorTypeAndFlags);
    const float radiance = unpackLightRadiance(lightInfo.logRadiance & 0xffff);
    return color * radiance;
}

void packLightColor(vec3 radiance, inout PolymorphicLightInfo lightInfo) {   
    const float intensity = max(radiance.r, max(radiance.g, radiance.b));
    if (intensity > 0.0) {
        const float logRadiance = saturate((log2(intensity) - kPolymorphicLightMinLog2Radiance) 
            / (kPolymorphicLightMaxLog2Radiance - kPolymorphicLightMinLog2Radiance));
        const uint packedRadiance = min(uint32_t(ceil(logRadiance * 65534.0)) + 1, 0xffffu);
        const float unpackedRadiance = unpackLightRadiance(packedRadiance);
        const vec3 normalizedRadiance = saturate(radiance.rgb / unpackedRadiance.xxx);
        lightInfo.logRadiance |= packedRadiance;
        lightInfo.colorTypeAndFlags |= Pack_R8G8B8_UFLOAT(normalizedRadiance);
    }
}

#endif