#ifndef _SRENDERER_PATH_GUIDING_COMMON_HEADER_
#define _SRENDERER_PATH_GUIDING_COMMON_HEADER_

/**
 * Exponential smoothing of a value
 * @param oldVal The old value
 * @param newVal The new value
 * @param alpha The smoothing factor
 */
float ExponentialSmooth(float oldVal, float newVal, float alpha) {
    return lerp(oldVal, newVal, alpha); }
float2 ExponentialSmooth(float2 oldVal, float2 newVal, float2 alpha) {
    return lerp(oldVal, newVal, alpha); }
float3 ExponentialSmooth(float3 oldVal, float3 newVal, float3 alpha) {
    return lerp(oldVal, newVal, alpha); }
float4 ExponentialSmooth(float4 oldVal, float4 newVal, float4 alpha) {
    return lerp(oldVal, newVal, alpha); }

#endif // !_SRENDERER_PATH_GUIDING_COMMON_HEADER_