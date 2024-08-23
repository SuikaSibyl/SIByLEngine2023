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
static const float k_numeric_limits_float_max = 3.402823466e+38;

static const float k_one_minus_epsilon = 0.999999940395355225f;
static const float k_float_epsilon = 1.192092896e-07f;

static const float k_sqrt2 = 1.41421356237309504880f;

/** Returns the smallest component of the vector. */
float minComponent(in const float3 v) {
    return min(v.x, min(v.y, v.z)); }
/** Returns the largest component of the vector. */
float maxComponent(in const float3 v) {
    return max(v.x, max(v.y, v.z)); }
/** Returns the largest component of the vector. */
float maxComponent(in const float4 v) {
    return max(max(v.x, v.y), max(v.z, v.w)); }

/** Returns the index of the smallest component of the vector. */
int minDimension(in const float3 v) {
    return (v.x < v.y) ? ((v.x < v.z) ? 0 : 2) : 
           ((v.y < v.z) ? 1 : 2); }
/** Returns the index of the largest component of the vector. */
int maxDimension(in const float2 v) {
    return (v.x >= v.y) ? 0 : 1; }
/** Returns the index of the largest component of the vector. */
int maxDimension(in const float3 v) {
    return (v.x > v.y) ? ((v.x > v.z) ? 0 : 2) : 
           ((v.y > v.z) ? 1 : 2); }
/** Returns the index of the largest component of the vector. */
int maxDimension(in const float4 v) {
    int4 isMax = select(v == maxComponent(v), int4(1), int4(0));
    return dot(isMax, int4(0, 1, 2, 3)); }

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

[Differentiable] float length_squared(in float2 v) { return dot(v, v); }
[Differentiable] float length_squared(in float3 v) { return dot(v, v); }

float distance_squared(in float3 v0, in float3 v1) {
    return dot(v0 - v1, v0 - v1);
}

float square(in float v) {
    return v * v;
}

[Differentiable] float sqr(float v) { return v * v; }
[Differentiable] float safe_sqrt(float v) { return sqrt(max(v, 0.f)); }

float elevation(float3 d) {
    return 2.f * asin(.5f * sqrt(sqr(d.x) + sqr(d.y) + sqr(d.z - 1.f)));
}

uint hprod<let Dim : int>(vector<uint, Dim> v) {
    uint result = v[0];
    for (int i = 1; i < Dim; ++i)
        result = result * v[i];
    return result;
}

float hprod<let Dim : int>(vector<float, Dim> v) {
    float result = v[0];
    for (int i = 1; i < Dim; ++i)
        result = result * v[i];
    return result;
}

void swap(inout float a, inout float b) {
    float temp = a; a = b; b = temp; }
void swap(inout float2 a, inout float2 b) {
    float2 temp = a; a = b; b = temp; }
void swap(inout float3 a, inout float3 b) {
    float3 temp = a; a = b; b = temp; }
void swap(inout float4 a, inout float4 b) {
    float4 temp = a; a = b; b = temp; }

float hypot(float2 v) { return length(v); }
float hypot(float3 v) { return length(v); }
float hypot(float4 v) { return length(v); }
float hypot(float x, float y) { return length(float2(x, y)); }
float hypot(float x, float y, float z) { return length(float3(x, y, z)); }
float hypot(float x, float y, float z, float w) { return length(float4(x, y, z, w)); }

float copysign(float mag, float sign) { return (sign >= 0) ? abs(mag) : -abs(mag); }

/**
 * floating-point complex structure based on pbrt-v4.
 */
struct complex {
    float re;
    float im;

    __init(float re) { this.re = re; this.im = 0; }
    __init(float re, float im) { this.re = re; this.im = im; }
};
[Differentiable] complex operator+(complex a, complex b) { return complex(a.re + b.re, a.im + b.im); }
[Differentiable] complex operator+(float value, complex z) { return complex(value) + z; }
[Differentiable] complex operator-(complex a, complex b) { return complex(a.re - b.re, a.im - b.im); }
[Differentiable] complex operator-(float value, complex z) { return complex(value) - z; }
[Differentiable] complex operator*(complex a, complex b) { 
    return complex(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re); }
[Differentiable] complex operator*(float value, complex z) { return complex(value) * z; }
[Differentiable] complex operator*(complex z, float value) { return complex(value) * z; }
[Differentiable] complex operator/(complex a, complex b) {
    float scale = 1 / (b.re * b.re + b.im * b.im);
    return complex(scale * (a.re * b.re + a.im * b.im),
                   scale * (a.im * b.re - a.re * b.im)); }
complex operator /(float value, complex z) { return complex(value) / z; }
[Differentiable] float real(complex z) { return z.re; }
[Differentiable] float imag(complex z) { return z.im; }
[Differentiable] float norm(complex z) { return z.re * z.re + z.im * z.im; }
[Differentiable] float abs(complex z) { return sqrt(norm(z)); }
[Differentiable] complex sqr(complex x) { return x * x; }
[Differentiable] complex sqrt(complex z) {
    const float n = abs(z);
    const float t1 = sqrt(.5 * (n + abs(z.re)));
    const float t2 = .5 * z.im / t1;

    if (n == 0) return complex(0);
    if (z.re >= 0) return { t1, t2 };
    else return complex(abs(t2), copysign(t1, z.im));
}


#endif // !_SRENDERER_COMMMON_MATH_HEADER_