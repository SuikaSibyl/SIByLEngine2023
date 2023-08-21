#ifndef _SRENDERER_RAYTRACER_PLUGIN_CONE_PRIMITIVE_
#define _SRENDERER_RAYTRACER_PLUGIN_CONE_PRIMITIVE_

#include "../../include/common/cpp_compatible.hlsli"
#include "../../include/common/math.hlsli"
#include "../../include/common/raycast.hlsli"

struct Cone {
    float3 position; // tip position
    float cosa;      // half cone angle
    float3 axis;     // normalized axis
    float height;    // height
};

/**
 * Ray-Cone intersection test.
 * @param s Cone
 * @param r Ray
 * @return Hit information: (t, n) if hit, (-1, 0, 0, 0) if no hit.
*/
float4 RayConeIntersection(in_ref(Cone) s, in_ref(Ray) r) {
    const float3 co = r.origin - s.position;
    const float a = dot(r.direction, s.axis) * dot(r.direction, s.axis) - s.cosa * s.cosa;
    const float b = 2. * (dot(r.direction, s.axis) * dot(co, s.axis) - dot(r.direction, co) * s.cosa * s.cosa);
    const float c = dot(co, s.axis) * dot(co, s.axis) - dot(co, co) * s.cosa * s.cosa;

    float det = b * b - 4. * a * c;
    if (det < 0.) return float4(-1, 0, 0, 0);

    det = sqrt(det);
    float t1 = (-b - det) / (2. * a);
    float t2 = (-b + det) / (2. * a);
    float reverseNormal = 1.f;

    // This is a bit messy; there ought to be a more elegant solution.
    float t = t1; float tback = t2;
    if (t < 0. || t2 > 0. && t2 < t) {
        t = t2; tback = t1;
    }
    if (t < 0.) return float4(-1, 0, 0, 0);

    float3 cp = r.origin + t * r.direction - s.position;
    float3 cpback = r.origin + tback * r.direction - s.position;
    float h = dot(cp, s.axis);
    float hback = dot(cpback, s.axis);

    // t sol is in the shadow cone
    if (h < 0 || h > s.height) {
        if (hback < 0. || hback > s.height) {
            return float4(-1, 0, 0, 0);
        }
        else if (tback > 0) {
            if (h > s.height) reverseNormal = -1.f;
            cp = cpback;
            h = hback;
            t = tback;
        }
    }
    else if ((tback > 0. && hback < 0.) || (tback < 0. && hback > 0.)) {
        reverseNormal = -1.f;
    }
    const float3 n = normalize(cp * dot(s.axis, cp) / dot(cp, cp) - s.axis);
    return float4(t, n * reverseNormal);
}

#endif // _SRENDERER_RAYTRACER_PLUGIN_CONE_PRIMITIVE_