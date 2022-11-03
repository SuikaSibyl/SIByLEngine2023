#ifndef _AAF_COMMON_HEADER_
#define _AAF_COMMON_HEADER_

#include "../../../../Utility/math.h"

/** output image resolution */
const uvec2 resolution = uvec2(800, 600);
/* half the fov */
const float half_fov = 11.0;
/** pixel radius */
const ivec2 pixel_radius = ivec2(10, 10);
/** the light is a Gaussian of standard deviation σ meters.
* we assume that the effective width of the light is 2σ. */
const vec3 lightPos0 = vec3(-0.24, 1.979, -0.22);
const vec3 lightPos1 = vec3(-0.24 + 0.47, 1.979, -0.22);
const vec3 lightPos2 = vec3(-0.24, 1.979, -0.22 + 0.38);
const vec3 lightVec1 = lightPos1 - lightPos0;
const vec3 lightVec2 = lightPos2 - lightPos0;
const vec3 lightCenter = lightPos0 + 0.5 * (lightVec1 + lightVec2);
const float lightSigma = sqrt(length(cross(lightVec1,lightVec2))/4.0f);
const vec3  light_normal = vec3(0,-1,0);
/* various threshold */
const float dist_scale_threshold = 10.0;
const float dist_threshold = 10.0;
const float angle_threshold = 20.0 * k_pi /180.0;

/**
* @param s2 is the flatter slope of double wedge
*/
float compute_omega_x_f(in float s2, in float omega_pix_max) {
    const float omega_x_bound_by_light = 2.0 / (lightSigma * s2);
    const float omega_x_bound_by_pixel = 1.0 / (omega_pix_max * (1+s2));
    return min(omega_x_bound_by_light, omega_x_bound_by_pixel);
}

float gaussian_filter(float distsq, float omegaxf) {
    // k = 3
    // beta = 1/k * max[sigma (d1/d2max - 1), 1/omega_x_max]
    // exponent = distsq / (2*beta^2)
    const float exponent = distsq * omegaxf * omegaxf;
    if(exponent > 0.9999) { return 0.0; }
    return exp(-3*exponent);
}

float computeSPP(in float s1, in float s2, in float proj_dist, in float omegaxf) {
    const float spp_t1 = (1/(1+s2) + proj_dist*omegaxf);
    const float spp_t2 = (1 + lightSigma * min(s1*omegaxf,1/proj_dist * s1/(1+s1)));
    const float spp = 4*spp_t1*spp_t1*spp_t2*spp_t2;
    return spp;
}

#endif