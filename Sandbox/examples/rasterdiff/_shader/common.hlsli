#ifndef _RASTERDIFF_COMMON_
#define _RASTERDIFF_COMMON_

float float2sign(float x) { return (x > 0.5f) ? 1.0f : -1.0f; }
float3 float2sign(float3 x) { return float3(
        float2sign(x[0]), float2sign(x[1]), float2sign(x[2])); }

#endif // _RASTERDIFF_COMMON_