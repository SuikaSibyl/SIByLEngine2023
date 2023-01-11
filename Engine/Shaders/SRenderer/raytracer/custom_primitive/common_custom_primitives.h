#ifndef _SRENDERER_COMMON_CUSTOM_PRIMITIVES_HEADER_
#define _SRENDERER_COMMON_CUSTOM_PRIMITIVES_HEADER_

#include "../../../Utility/math.h"
#include "../include/common_rt_config.h"

/** 
* Compute intersection of a ray and a sphere 
* @ref: http://viclw17.github.io/2018/07/16/raytracing-ray-sphere-intersection
* @ref: <ray tracing gems> https://link.springer.com/content/pdf/10.1007/978-1-4842-4427-2_7.pdf
*/

float hitSphere(
    in vec3 center, in float radius, 
    in vec3 origin, in vec3 direction)
{
    const vec3  oc           = origin - center;
    const float a            = dot(direction, direction);
    const float b            = 2.0 * dot(oc, direction);
    const float c            = dot(oc, oc) - radius * radius;
    // Degenerated case
    if (a == 0) {
        if (b == 0)
            return -1.0; // if no solve
        return -c / b;
    }
    vec3 term = oc - (dot(oc,direction)*direction);
    float discriminant = 4* a * (radius*radius - dot(term,term));
    if (discriminant < 0)
        return -1.0; // if no solve
    float root_discriminant = sqrt(discriminant);
    float t0, t1;
    if (b >= 0) {
        t0 = (- b - root_discriminant) / (2 * a);
        t1 = 2 * c / (- b - root_discriminant);
    } else {
        t0 = 2 * c / (- b + root_discriminant);
        t1 = (- b + root_discriminant) / (2 * a);
    }
    float t = -1;
    if (t0 >= 0 && t0 < T_MAX)
        t = t0;
    if (t1 >= 0 && t1 < T_MAX && t < 0)
        t = t1;
    return t;
}

#endif