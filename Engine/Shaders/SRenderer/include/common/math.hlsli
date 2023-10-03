#ifndef _SRENDERER_COMMMON_MATH_HEADER_
#define _SRENDERER_COMMMON_MATH_HEADER_

static const float k_pi         = 3.1415926535897932f;
static const float k_2pi        = 6.2831853071795864f;
static const float k_pi_over_2  = k_pi / 2;
static const float k_pi_over_4  = k_pi / 4;
static const float k_inv_pi     = 1. / k_pi;
static const float k_inv_2_pi   = 1. / (2 * k_pi);

static const float k_inf        = 1.0f / 0.0f;

static const float k_numeric_limits_float_min = 1.0f / exp2(126);

/** Returns the smallest component of the vector. */
float minComponent(in const float3 v) {
    return min(v.x, min(v.y, v.z)); }
/** Returns the largest component of the vector. */
float maxComponent(in const float3 v) {
    return max(v.x, max(v.y, v.z)); }

/** Returns the index of the smallest component of the vector. */
int minDimension(in const float3 v) {
    return (v.x < v.y) ? ((v.x < v.z) ? 0 : 2) : 
           ((v.y < v.z) ? 1 : 2); }
/** Returns the index of the largest component of the vector. */
int maxDimension(in const float3 v) {
    return (v.x > v.y) ? ((v.x > v.z) ? 0 : 2) : 
           ((v.y > v.z) ? 1 : 2); }

// permute a float3 vector
float3 permute(in const float3 v, int x, int y, int z) {
    return float3(v[x], v[y], v[z]); }

/** Returns either -1 or 1 based on the sign of the input value.
* If the input is zero, 1 is returned. */
float signNotZero(float x) {
    return (x >= 0.f) ? 1.f : -1.f; }
/** 2-component version of signNotZero. */
float2 signNotZero(float2 v) {
    return float2(signNotZero(v.x), signNotZero(v.y)); }
/** 3-component version of signNotZero. */
float3 signNotZero(float3 v) {
    return float3(signNotZero(v.x), signNotZero(v.y), signNotZero(v.z)); }
/** 4-component version of signNotZero. */
float4 signNotZero(float4 v) {
    return float4(signNotZero(v.x), signNotZero(v.y), signNotZero(v.z), signNotZero(v.w)); }

/** Returns the luminance of the input rgb. */
float luminance(in const float3 rgb) {
    return dot(rgb, float3(0.212671, 0.715160, 0.072169));
}

float2 interpolate(float2 vertices[3], float3 bary) {
    return vertices[0] * bary[0] + vertices[1] * bary[1] + vertices[2] * bary[2];
}
float3 interpolate(float3 vertices[3], float3 bary) {
    return vertices[0] * bary[0] + vertices[1] * bary[1] + vertices[2] * bary[2];
}
float4 interpolate(float4 vertices[3], float3 bary) {
    return vertices[0] * bary[0] + vertices[1] * bary[1] + vertices[2] * bary[2];
}

float length_square(in float3 v) {
    return dot(v, v);
}

float distance_squared(in float3 v0, in float3 v1) {
    return dot(v0 - v1, v0 - v1);
}

float square(in float v) {
    return v * v;
}

#endif // !_SRENDERER_COMMMON_MATH_HEADER_