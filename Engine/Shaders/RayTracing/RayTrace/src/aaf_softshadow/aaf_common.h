#ifndef _AAF_COMMON_HEADER_
#define _AAF_COMMON_HEADER_

#include "../../../../Utility/math.h"
#include "../../../../Utility/random.h"

/** output image resolution */
const uvec2 resolution = uvec2(800, 600);
/* half the fov */
const float half_fov = 30.0;
/** pixel radius */
const ivec2 pixel_radius = ivec2(10, 10);
/** the light is a Gaussian of standard deviation σ meters.
* we assume that the effective width of the light is 2σ. */
const vec3 lightPos0 = vec3(-4.5, 16, 8);
const vec3 lightPos1 = vec3(1.5, 16, 8);
const vec3 lightPos2 = vec3(-4.5, 21.8284, 3.8284);
const vec3 lightVec1 = lightPos1 - lightPos0;
const vec3 lightVec2 = lightPos2 - lightPos0;
const vec3 lightCenter = lightPos0 + 0.5 * (lightVec1 + lightVec2);
const float lightSigma = sqrt(length(cross(lightVec1,lightVec2))/4.0f);
const vec3  light_normal = vec3(0,-1,0);
/* various threshold */
const float dist_scale_threshold = 10.0;
const float dist_threshold = 10.0;
const float angle_threshold = 20.0 * k_pi /180.0;

const vec3 Kd = vec3(0.87402f, 0.87402f, 0.87402f);
const float alpha = 1.f;
const float mu = 2.f;
const float k = 3;
/**
* @param s2 is the flatter slope of double wedge
*/
float compute_omega_x_f(in float s2, in float proj_dist) {
    const float omega_x_bound_by_light = mu / (lightSigma * s2);
    const float omega_x_bound_by_pixel = alpha / (proj_dist * (1+s2));
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
    const float Ap = proj_dist * proj_dist;
    const float Al = 4.f * lightSigma * lightSigma;
    const float spp_t1 = 1 + mu * s1/s2;
    const float spp_t2 = mu * 2/s2 * sqrt(Ap/Al) + alpha / (1+s2);
    return 4*spp_t1*spp_t1*spp_t2*spp_t2;
}

float computeBeta(in float s2, in float proj_dist) {
    const float omega_x_max = alpha / (proj_dist * (1+s2));
    return max(lightSigma * s2, 1./omega_x_max) / (k*mu);
}

float gaussian(float distsq, float beta) {
    const float sqrt_2_pi = sqrt(2*k_pi);
    const float exponent = - distsq / (2 * beta * beta);
    return exp(exponent) / (sqrt_2_pi * beta);
}

vec3 sampleAreaLight(inout uint rngState) {
    vec2 lsample = vec2(stepAndOutputRNGFloat(rngState), stepAndOutputRNGFloat(rngState));
    return lightPos0 + lsample.x * lightVec1 + lsample.y * lightVec2;
}

vec3 sampleAreaLight(in vec2 lsample) {
    return lightPos0 + lsample.x * lightVec1 + lsample.y * lightVec2;
}

#endif