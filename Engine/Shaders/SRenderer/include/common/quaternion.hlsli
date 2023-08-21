#ifndef _SRENDERER_COMMMON_QUATERNION_HEADER_
#define _SRENDERER_COMMMON_QUATERNION_HEADER_

/** Rotate vector v with quaternion q. */
float3 quaternionRotate(float3 v, float4 q) {
    const float3 b = q.xyz;
    const float b2 = dot(b, b);
    return (v * (q.w * q.w - b2) + b * (dot(v, b) * 2.f) + cross(b, v) * (q.w * 2.f));
}

/**
 * Quaternion conjugate.
 * For unit quaternions, conjugate equals inverse.
 * Use this to create a quaternion that rotates in the opposite direction.
 */
float4 quaternionConjugate(float4 q) {
    return float4(-q.xyz, q.w);
}

#endif // !_SRENDERER_COMMMON_QUATERNION_HEADER_