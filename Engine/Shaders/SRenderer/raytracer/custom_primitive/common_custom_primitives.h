#ifndef _SRENDERER_COMMON_CUSTOM_PRIMITIVES_HEADER_
#define _SRENDERER_COMMON_CUSTOM_PRIMITIVES_HEADER_

#include "../../../Utility/math.h"
#include "../include/common_rt_config.h"

/** 
* Compute intersection of a ray and a sphere 
* @ref: http://viclw17.github.io/2018/07/16/raytracing-ray-sphere-intersection 
*/
float hitSphere(
    in vec3 center, in float radius, 
    in vec3 origin, in vec3 direction)
{
    const vec3  oc           = origin - center;
    const float a            = dot(direction, direction);
    const float b            = 2.0 * dot(oc, direction);
    const float c            = dot(oc, oc) - radius * radius;
    float t0, t1;
    bool itersected = quadratic(a, b, c, t0, t1);
    if(!itersected) return -1.0; // if no solve
    float t = -1;
    if (t0 >= 0 && t0 < T_MAX)
        t = t0;
    if (t1 >= 0 && t1 < T_MAX && t < 0)
        t = t1;
    return t;
}

#endif