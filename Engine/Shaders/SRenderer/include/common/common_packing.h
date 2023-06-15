#ifndef _SRENDERER_COMMON_PACKING_INCLUDED_
#define _SRENDERER_COMMON_PACKING_INCLUDED_

// Packs a normalized float value into an 8-bit unsigned integer representation
uint Pack_R8_UFLOAT(float value) {
    // Clamp the value to the valid range [0, 1]
    value = clamp(value, 0.0, 1.0);
    // Convert the float value to an 8-bit integer value
    return uint(value * 255.0f + 0.5f);
}

// Unpacks an 8-bit value into a normalized float value
float Unpack_R8_UFLOAT(uint packedValue) {
    // Convert the packed value to a normalized float
    return clamp(packedValue, 0, 255) / 255.0f;
}

uint Pack_R8G8B8_UFLOAT(float3 rgb) {
    uint r = Pack_R8_UFLOAT(rgb.r);
    uint g = Pack_R8_UFLOAT(rgb.g) << 8;
    uint b = Pack_R8_UFLOAT(rgb.b) << 16;
    return r | g | b;
}

float3 Unpack_R8G8B8_UFLOAT(uint rgb) {
    float r = Unpack_R8_UFLOAT(rgb);
    float g = Unpack_R8_UFLOAT(rgb >> 8);
    float b = Unpack_R8_UFLOAT(rgb >> 16);
    return float3(r, g, b);
}

uint Pack_R8G8B8A8_Gamma_UFLOAT(float4 rgba, float gamma) {
    rgba = pow(clamp(rgba, 0.0, 1.0), float4(1.0 / gamma));
    uint r = Pack_R8_UFLOAT(rgba.r);
    uint g = Pack_R8_UFLOAT(rgba.g) << 8;
    uint b = Pack_R8_UFLOAT(rgba.b) << 16;
    uint a = Pack_R8_UFLOAT(rgba.a) << 24;
    return r | g | b | a;
}

float4 Unpack_R8G8B8A8_Gamma_UFLOAT(uint rgba, float gamma) {
    float r = Unpack_R8_UFLOAT(rgba);
    float g = Unpack_R8_UFLOAT(rgba >> 8);
    float b = Unpack_R8_UFLOAT(rgba >> 16);
    float a = Unpack_R8_UFLOAT(rgba >> 24);
    float4 v = float4(r, g, b, a);
    v = pow(clamp(v, 0.0, 1.0), float4(gamma));
    return v;
}

#endif