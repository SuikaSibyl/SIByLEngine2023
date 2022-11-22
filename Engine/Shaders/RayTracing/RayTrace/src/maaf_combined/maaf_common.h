#ifndef _MAAF_COMMON_HEADER_
#define _MAAF_COMMON_HEADER_

#include "../../../../Utility/random.h"
#include "../../../../Utility/sampling.h"

// Output Setting
const uvec2 resolution = uvec2(800, 600);
const vec2  default_slope = vec2(k_inf, -k_inf);
// Camera Setting
const float half_fov = 11.0;
// Apeture Setting
const float focus_dist = 5.5f;
const float length_radius = 0.5;
const float focal_plane_half_width = focus_dist * tan(half_fov*k_pi/180)*float(resolution.x)/float(resolution.y);
const float sigma_lens = 0.33f;
const float inv_sigma_lens = 1.f/sigma_lens;
// GI Setting
const float zMIN = 0.02 * 2.f;
const float zMINNONE = 0.02 * 20.f;
// Filtering Setting
int prefilter_radius = 2;
int filter_radius = 10;
// Direct Setting
const float dist_scale_threshold = 10.0;
const float dist_threshold = 0.1;
const float angle_threshold = 20.0 * k_pi /180.0;
const float cos_angle_threshold = abs(cos(20.0 * k_pi /180.0));
const float defocus_threashold = 100;

float computeDefocusBlurSlope(in float z) {
    return length_radius * (resolution.x/2) / focal_plane_half_width * (focus_dist/z - 1);
}

struct PrimaryRayPayload {
    vec3 worldLocation;
    uint rngState;
    vec3 worldNormal;
    float visibility;
    vec3 brdf;
    float y0;
    vec3 indirect;
    float y1;
    vec2 reflectorDistMinMax;
    vec2 direcetSlopeMinMax;
    float v0;
    float v1;
    bool rayHitSky;
};

struct ShadowRayPayload {
    float distanceMin;
    float distanceMax;
    bool  hitOccluder;
};

struct IndirectRayPayload {
    vec3    L;
    float   distanceToReflector;
    vec3    worldPosition;
    vec3    worldNormal;
    vec3    albedo;
    bool    hit;
};

// 25 accumulated components.
// size: 25 * vec4
struct MAAFIntermediate {
    vec4 data[25];
};

// will reuse the MAAFIntermediate buffer
// 24 * vec4(pairs);
struct InitialSampleRecord {
    vec4 illumination[16];  // 16 vec4
    vec2 y[16];            // 8  vec4
    vec4 padding;
};

struct IntermediateStruct {
    vec4 data[25];
};

void clearIntermediateStruct(inout IntermediateStruct interm) {
    for(int i=0; i<25; ++i)
        interm.data[i] = vec4(0);
}

// cornell box light setting
const vec3 lightPos0 = vec3(-0.240, 1.975, -0.240);
const vec3 lightPos1 = vec3(-0.240, 1.975, +0.240);
const vec3 lightPos2 = vec3(+0.240, 1.975, -0.240);
const vec3 lightVec1 = lightPos1 - lightPos0;
const vec3 lightVec2 = lightPos2 - lightPos0;
const vec3 lightCenter = lightPos0 + 0.5 * (lightVec1 + lightVec2);
const float lightSigma = sqrt(length(cross(lightVec1,lightVec2))/4.0f);
const float invLightSigma = 1.f / lightSigma;
const float omega_light_max = invLightSigma;
const vec3  light_normal = vec3(0,-1,0);
vec3 sampleAreaLight(inout uint rngState) {
    vec2 lsample = vec2(stepAndOutputRNGFloat(rngState), stepAndOutputRNGFloat(rngState));
    return lightPos0 + lsample.x * lightVec1 + lsample.y * lightVec2;
}
vec3 sampleAreaLight(in vec2 lsample) {
    return lightPos0 + lsample.x * lightVec1 + lsample.y * lightVec2;
}

// cornell box scene setting
const vec3 Kd = vec3(1.);
vec3 getAlbedo(vec3 worldNormal) {
    vec3 color = vec3(0.8f);
    const float dotX = dot(worldNormal, vec3(1.0, 0.0, 0.0));
    if(dotX > 0.99)
        color = vec3(0.8, 0.0, 0.0);
    else if(dotX < -0.99)
        color = vec3(0.0, 0.8, 0.0);
    return color;
}

float gaussian(float distsq, float beta) {
    const float sqrt_2_pi = sqrt(2*k_pi);
    const float exponent = - distsq / (2 * beta * beta);
    return exp(exponent) / (sqrt_2_pi * beta);
}

float unnormalized_gaussian(in float dist, in float invsigma) {
    return exp(-0.5 * dist * dist * invsigma * invsigma);
}

float unnormalized_gaussian_sq(in float distsq, in float invsigma) {
    return exp(-0.5 * distsq * invsigma * invsigma);
}

// 18+2 float
struct MAAFParameters {
    float cxp[3];
    float cyp[3];
    float cup[3];
    // fourier domain Gaussian sigma
    // also, inverse of primal domain sigma
    float sigmaxp[3];
    float sigmayp[3];
    float sigmaup[3];
    float scaling[2];
};

// 40 floats
struct MAAFParametersGroup {
    MAAFParameters directParams;
    MAAFParameters indirectParams;
};

// struct MAAFParameters {
//     float cx1i, cx2i, cx1d, cx2d;
//     float cy1i, cy2i, cy1d, cy2d;
//     float cu1a, cu2a, sigmayai, sigmayad;
//     float sigmax0i, sigmax1i, sigmax2i, sigmauaa;
//     float sigmax0d, sigmax1d, sigmax2d, padding;
// };

// void computeMAAFParameters(
//     in float omega_y_direct_max, in float omega_y_indirect_max, in float omega_u_max,
//     in vec2 slope_y_direct, in vec2 slope_y_indirect, in vec2 slope_u,
//     out vec2 cxp_direct, out vec2 cxp_indirect, out vec2 cyp_direct, out vec2 cyp_indirect, out vec2 cup, 
//     out vec3 sigmax_direct, out vec3 sigmax_indirect, out float sigmayp_direct, out float sigmayp_indirect, out float sigmaup) 
// {
//     const int N = 3;
//     const int tNm1 = 2*N - 1;
//     // compute cy direct
//     const float cy1_direct = omega_y_direct_max * 2*1 / tNm1;
//     const float cy2_direct = omega_y_direct_max * 2*2 / tNm1;
//     cyp_direct = vec2(cy1_direct, cy2_direct);
//     sigmayp_direct = omega_y_direct_max / tNm1;
//     // compute cy indirect
//     const float cy1_indirect = omega_y_direct_max * 2*1 / tNm1;
//     const float cy2_indirect = omega_y_direct_max * 2*2 / tNm1;
//     cyp_indirect = vec2(cy1_indirect, cy2_indirect);
//     sigmayp_indirect = omega_y_indirect_max / tNm1;
//     // compute cy
//     const float cu1 = omega_u_max * 2*1 / tNm1;
//     const float cu2 = omega_u_max * 2*2 / tNm1;
//     cup = vec2(cu1, cu2);
//     sigmaup = omega_u_max / tNm1;
//     // compute xy direct
//     const float cxy1_direct = 0.2/slope_y_direct.y + 0.6/slope_y_direct.x;
//     const float cxy2_direct = 0.6/slope_y_direct.y + 1.0/slope_y_direct.x;
//     const float sigmaxy0_direct = 0.5 * omega_y_direct_max * (0.2/slope_y_direct.x - (-0.2)/slope_y_direct.y);
//     const float sigmaxy1_direct = 0.5 * omega_y_direct_max * (0.6/slope_y_direct.x - 0.2/slope_y_direct.y);
//     const float sigmaxy2_direct = 0.5 * omega_y_direct_max * (1.0/slope_y_direct.x - 0.6/slope_y_direct.y);
//     // compute xu direct
//     const float cxu1 = 0.2/slope_u.y + 0.6/slope_u.x;
//     const float cxu2 = 0.6/slope_u.y + 1.0/slope_u.x;
//     const float sigmaxu0 = 0.5 * omega_u_max * (0.2/slope_u.x - (-0.2)/slope_u.y);
//     const float sigmaxu1 = 0.5 * omega_u_max * (0.6/slope_u.x - 0.2/slope_u.y);
//     const float sigmaxu2 = 0.5 * omega_u_max * (1.0/slope_u.x - 0.6/slope_u.y);
//     // compute direct x
//     const float sigmax0_direct = max(sigmaxy0_direct, sigmaxu0);
//     const float cx1_direct = (max(cxy1_direct+sigmaxy1_direct, cxu1+sigmaxu1) + min(cxy1_direct-sigmaxy1_direct, cxu1-sigmaxu1)) / 2;
//     const float sigmax1_direct = (max(cxy1_direct+sigmaxy1_direct, cxu1+sigmaxu1) - min(cxy1_direct-sigmaxy1_direct, cxu1-sigmaxu1)) / 2;
//     const float cx2_direct = (max(cxy2_direct+sigmaxy2_direct, cxu2+sigmaxu2) + min(cxy2_direct-sigmaxy2_direct, cxu2-sigmaxu2)) / 2;
//     const float sigmax2_direct = (max(cxy2_direct+sigmaxy2_direct, cxu2+sigmaxu2) - min(cxy2_direct-sigmaxy2_direct, cxu2-sigmaxu2)) / 2;
//     // compute xy indirect
//     const float cxy1_indirect = 0.2/slope_y_indirect.y + 0.6/slope_y_indirect.x;
//     const float cxy2_indirect = 0.6/slope_y_indirect.y + 1.0/slope_y_indirect.x;
//     const float sigmaxy0_indirect = 0.5 * omega_y_indirect_max * (0.2/slope_y_indirect.x - (-0.2)/slope_y_indirect.y);
//     const float sigmaxy1_indirect = 0.5 * omega_y_indirect_max * (0.6/slope_y_indirect.x - 0.2/slope_y_indirect.y);
//     const float sigmaxy2_indirect = 0.5 * omega_y_indirect_max * (1.0/slope_y_indirect.x - 0.6/slope_y_indirect.y);
//     // compute indirect x
//     const float sigmax0_indirect = max(sigmaxy0_indirect, sigmaxu0);
//     const float cx1_indirect = (max(cxy1_indirect+sigmaxy1_indirect, cxu1+sigmaxu1) + min(cxy1_indirect-sigmaxy1_indirect, cxu1-sigmaxu1)) / 2;
//     const float sigmax1_indirect = (max(cxy1_indirect+sigmaxy1_indirect, cxu1+sigmaxu1) - min(cxy1_indirect-sigmaxy1_indirect, cxu1-sigmaxu1)) / 2;
//     const float cx2_indirect = (max(cxy2_indirect+sigmaxy2_indirect, cxu2+sigmaxu2) + min(cxy2_indirect-sigmaxy2_indirect, cxu2-sigmaxu2)) / 2;
//     const float sigmax2_indirect = (max(cxy2_indirect+sigmaxy2_indirect, cxu2+sigmaxu2) - min(cxy2_indirect-sigmaxy2_indirect, cxu2-sigmaxu2)) / 2;
//     cxp_direct = vec2(cx1_direct, cx2_direct);
//     cxp_indirect = vec2(cx1_indirect, cx2_indirect);
//     sigmax_direct = vec3(sigmax0_direct, sigmax1_direct, sigmax2_direct);
//     sigmax_indirect = vec3(sigmax0_indirect, sigmax1_indirect, sigmax2_indirect);
// }


void computeMAAFParameters(
    in float omgea_y_max, in float omgea_u_max,
    in float s_min, in float s_max,
    in float su_min, in float su_max,
    inout MAAFParameters params) 
{
    const int N = 3;
    const float invtNm1 = 1./(2*N - 1);
    // if(su_min*su_max < 0) {
    //     su_min = 0.001;
    //     su_max = abs(su_max);
    // }
    // else {
    //     const float tmp = su_min;
    //     su_min = abs(su_max);
    //     su_max = abs(tmp);
    // }
    // compute cyp
    params.cxp[0] = 0;
    params.cyp[0] = 0;
    params.cup[0] = 0;
    if(s_min == k_inf) {
        // params.sigmaxp[0] = omgea_u_max * invtNm1 / su_min;
        // params.sigmayp[0] = omgea_u_max * s_max * invtNm1 / su_min;
        // params.sigmaup[0] = omgea_u_max * invtNm1;
        // for(int i=1; i<3; ++i) {
        //     params.cxp[i] = 0.5 * omgea_u_max * (((2*i-1)*invtNm1/su_max) + ((2*i+1)*invtNm1/su_min));
        //     params.cyp[i] = 0.5 * omgea_u_max * (((2*i-1)*s_min*invtNm1/su_max) + ((2*i+1)*s_max*invtNm1/su_min));
        //     params.cup[i] = omgea_u_max * 2*float(i) * invtNm1;
        //     params.sigmaxp[i] = 0.5 * omgea_u_max * (((2*i+1)*invtNm1/su_min) - ((2*i-1)*invtNm1/su_max));
        //     // params.sigmayp[i] = 0.5 * omgea_u_max * (((2*i+1)*s_max*invtNm1/su_min) - ((2*i-1)*s_min*invtNm1/su_max));
        //     params.sigmayp[i] = 0;
        //     params.sigmaup[i] = omgea_u_max * invtNm1;
        // }
        // for(int i=0; i<3; ++i) {
        //     params.sigmaxp[i] = 1.f / params.sigmaxp[i];
        //     params.sigmayp[i] = 0.f;
        //     params.sigmaup[i] = 1.f / params.sigmaup[i];
        // }
    }
    else {
        params.sigmaxp[0] = omgea_y_max * invtNm1 / s_min;
        params.sigmayp[0] = omgea_y_max * invtNm1;
        params.sigmaup[0] = omgea_y_max * su_max * invtNm1 / s_min;
        for(int i=1; i<3; ++i) {
            params.cxp[i] = 0.5 * omgea_y_max * (((2*i-1)*invtNm1/s_max) + ((2*i+1)*invtNm1/s_min));
            params.cyp[i] = omgea_y_max * 2*float(i) * invtNm1;
            params.cup[i] = 0.5 * omgea_y_max * (((2*i-1)*su_min*invtNm1/s_max) + ((2*i+1)*su_max*invtNm1/s_min));
            params.sigmaxp[i] = 0.5 * omgea_y_max * (((2*i+1)*invtNm1/s_min) - ((2*i-1)*invtNm1/s_max));
            params.sigmayp[i] = omgea_y_max * invtNm1;
            params.sigmaup[i] = 0.5 * omgea_y_max * (((2*i+1)*su_max*invtNm1/s_min) - ((2*i-1)*su_min*invtNm1/s_max));
        }
        // for(int i=0; i<3; ++i) {
        //     params.sigmaxp[i] = 1.f / params.sigmaxp[i];
        //     params.sigmayp[i] = 1.f / params.sigmayp[i];
        //     params.sigmaup[i] = 1.f / params.sigmaup[i];
        // }
    }
}

void computeMAAFParameters(
    in float omega_x_max, in float omega_y_max, in float omega_u_max,
    in float s_min, in float s_max,
    in float su_min, in float su_max,
    in float proj_dist,
    inout MAAFParameters params) 
{    
    const int N = 3;
    const float invtNm1 = 1./(2*N - 1);
	const float bar_width_u = 2.f * omega_u_max * invtNm1;
    // if su are all minus
    float sign_u = 1.f;
    if(su_min<0 && su_max<0) {
        const float tmp = abs(su_min);
        su_min = abs(su_max);
        su_max = tmp;
        sign_u = -1.f;
    }
    // if su are all minus / positive
    if(su_min>0 && su_max>0) {
        // central components
        float min_u = -0.5f * bar_width_u;
        float max_u = min_u + bar_width_u;
        float max_x = min(omega_x_max, max_u / su_min);
        float min_x = -max_x;
        float max_y = (s_min == k_inf)
            ? omega_y_max
            : min(omega_y_max, max_x * s_max / proj_dist);
        float min_y = -max_y;
        params.cxp[0]       = 0.5 * (min_x + max_x);
        params.cyp[0]       = 0.5 * (min_y + max_y);
        params.cup[0]       = 0.5 * (min_u + max_u);
        params.sigmaxp[0]   = 0.5 * (max_x - min_x);
        params.sigmayp[0]   = 0.5 * (max_y - min_y);
        params.sigmaup[0]   = 0.5 * (max_u - min_u);
        // outer components
        for(int i=1; i<N; ++i) {
            min_u = (i-0.5f) * bar_width_u;
            max_u = min_u + bar_width_u;
            max_x = min(omega_x_max, max_u / su_min);
            min_x = min(omega_x_max, max(min_u / su_max, params.cxp[i-1]+params.sigmaxp[i-1]));
            float max_y = (s_min == k_inf)
                ? omega_y_max
                : min(omega_y_max, max_x * s_max / proj_dist);
            float min_y = (s_min == k_inf)
                ? -max_y
                : min(omega_y_max, min_x * s_min / proj_dist);
            params.cxp[i]       = 0.5 * (min_x + max_x) * sign_u;
            params.cyp[i]       = 0.5 * (min_y + max_y) * sign_u;
            params.cup[i]       = 0.5 * (min_u + max_u);
            params.sigmaxp[i]   = 0.5 * (max_x - min_x);
            params.sigmayp[i]   = 0.5 * (max_y - min_y);
            params.sigmaup[i]   = 0.5 * (max_u - min_u);
            params.scaling[i-1] = 
                unnormalized_gaussian(params.cxp[i], omega_x_max * 0.5) *
				unnormalized_gaussian(params.cyp[i], omega_y_max * 0.5) *
				unnormalized_gaussian(params.cup[i], omega_u_max * 0.5);
            // if(params.sigmaxp[i] < 1.f/20)
            //     params.scaling[i-1] = 0;
            // if(params.sigmayp[i] < 0.00001)
            //     params.scaling[i-1] = 0;
            if(abs(params.cxp[i])*10 < 3.14)
            // || abs(params.cyp[i])*lightSigma < 3.14
            // || abs(params.cup[i])*lightSigma < 3.14)
                params.scaling[i-1] = 0;
        }
    }
    // if su_min is minus, su_max is positive
    // will degenerate into box filter
    else {
        // central components
        float min_u = -0.5f * bar_width_u;
        float max_u = min_u + bar_width_u;
        float max_x = omega_x_max;
        float min_x = -omega_x_max;
        float max_y = (s_min == k_inf)
            ? omega_y_max
            : min(omega_y_max, max_x * s_max / proj_dist);
        float min_y = -max_y;
        params.cxp[0]       = 0.5 * (min_x + max_x);
        params.cyp[0]       = 0.5 * (min_y + max_y);
        params.cup[0]       = 0.5 * (min_u + max_u);
        params.sigmaxp[0]   = 0.5 * (max_x - min_x);
        params.sigmayp[0]   = 0.5 * (max_y - min_y);
        params.sigmaup[0]   = 0.5 * (max_u - min_u);
        // outer components
        for(int i=1; i<N; ++i) {
            params.scaling[i-1] = -1;
        }
    }

}

void computeMAAFParameters_indir(
    in float omgea_x_max, in float omgea_y_max, in float omgea_u_max,
    in float sy_min, in float sy_max,
    in float su_min, in float su_max,
    inout MAAFParameters params) {
    //
    const float N = 3;
    const float invtNm1 = 1./(2*N - 1);
    // center components have 0 offsets
    params.cxp[0] = 0;
    params.cyp[0] = 0;
    params.cup[0] = 0;

    params.sigmaxp[0] = omgea_y_max * invtNm1 / sy_min;
    params.sigmayp[0] = omgea_y_max * invtNm1;
    params.sigmaup[0] = omgea_y_max * su_max * invtNm1 / sy_min;

    for(int i=1; i<3; ++i) {
        params.cxp[i] = 0.5 * omgea_y_max * (((2*i-1)*invtNm1/sy_max) + ((2*i+1)*invtNm1/sy_min));
        params.cyp[i] = omgea_y_max * 2*float(i) * invtNm1;
        params.cup[i] = 0.5 * omgea_y_max * (((2*i-1)*su_min*invtNm1/sy_max) + ((2*i+1)*su_max*invtNm1/sy_min));
        params.sigmaxp[i] = 0.5 * omgea_y_max * (((2*i+1)*invtNm1/sy_min) - ((2*i-1)*invtNm1/sy_max));
        params.sigmayp[i] = omgea_y_max * invtNm1;
        params.sigmaup[i] = 0.5 * omgea_y_max * (((2*i+1)*su_max*invtNm1/sy_min) - ((2*i-1)*su_min*invtNm1/sy_max));
    }
}

/*
* Compute omega_x_max for direct lllumination.
* @param proj_dist: projected distance of hit point
* @param indirect_slope_min: minimum indirect slope
* @param apeture_slope_abs_min: minimum absolute of apeture slope
*/
float compute_omega_x_max_dir(
    in float proj_dist,
    in float direct_slope_min,
    in float apeture_slope_abs_min) 
{
    const float omega_u_max = 1.;
    const float omega_y_max = invLightSigma;
    float omega_x_max = 0.5; // maximum allowed pixel bandlimit
    // limited by apeture slope
    if(apeture_slope_abs_min != k_inf && apeture_slope_abs_min != 0)
        omega_x_max = min(omega_x_max, omega_u_max / apeture_slope_abs_min);
     // we mul proj_dist cause our x is in pixel space
    if(direct_slope_min != k_inf) // if current pixel has indirect slope
        omega_x_max = min(omega_x_max, omega_y_max * proj_dist / direct_slope_min);
    return omega_x_max;
}

/*
* Compute omega_x_max for indirect lllumination.
* @param proj_dist: projected distance of hit point
* @param indirect_slope_min: minimum indirect slope
* @param apeture_slope_abs_min: minimum absolute of apeture slope
*/
float compute_omega_x_max_ind(
    in float proj_dist,
    in float indirect_slope_min,
    in float apeture_slope_abs_min) 
{
    const float omega_u_max = 1.;
    const float omega_y_max = 2.2;
    float omega_x_max = 0.5; // maximum allowed pixel bandlimit
    // limited by apeture slope
    if(apeture_slope_abs_min != k_inf && apeture_slope_abs_min != 0)
        omega_x_max = min(omega_x_max, omega_u_max / apeture_slope_abs_min);
     // we mul proj_dist cause our x is in pixel space
    if(indirect_slope_min != k_inf) // if current pixel has indirect slope
        omega_x_max = min(omega_x_max, omega_y_max * proj_dist / indirect_slope_min);
    return omega_x_max;
}

/**
* Preintegral MAAF inner integral using Monte Carlo method.
*/
void preIntegralMAAF_dir(
    in MAAFParameters params, 
    in float y0, in float y1,
    in float u0, in float u1,
    in vec3 f, 
    inout MAAFIntermediate intermediate)
{
    float axis0comp[5];
    float axis0weights[5];
    float axis1comp[5];
    float axis1weights[5];

    const float gaussian_00 = unnormalized_gaussian(y0, params.sigmayp[0]) * unnormalized_gaussian(u0, params.sigmaup[0]);
    const float gaussian_01 = unnormalized_gaussian(y0, params.sigmayp[1]) * unnormalized_gaussian(u0, params.sigmaup[1]);
    const float gaussian_02 = unnormalized_gaussian(y0, params.sigmayp[2]) * unnormalized_gaussian(u0, params.sigmaup[2]);
    axis0weights[0] = gaussian_00;
    axis0weights[1] = gaussian_01;
    axis0weights[2] = gaussian_02;
    axis0weights[3] = axis0weights[1];
    axis0weights[4] = axis0weights[2];
    axis0comp[0] = axis0weights[0];
    axis0comp[1] = cos((params.cyp[1]*y0 + params.cup[1]*u0))* axis0weights[1];
    axis0comp[2] = cos((params.cyp[2]*y0 + params.cup[2]*u0))* axis0weights[2];
    axis0comp[3] = sin((params.cyp[1]*y0 + params.cup[1]*u0))* axis0weights[3];
    axis0comp[4] = sin((params.cyp[2]*y0 + params.cup[2]*u0))* axis0weights[4];

    const float gaussian_10 = unnormalized_gaussian(y1, params.sigmayp[0]) * unnormalized_gaussian(u1, params.sigmaup[0]);
    const float gaussian_11 = unnormalized_gaussian(y1, params.sigmayp[1]) * unnormalized_gaussian(u1, params.sigmaup[1]);
    const float gaussian_12 = unnormalized_gaussian(y1, params.sigmayp[2]) * unnormalized_gaussian(u1, params.sigmaup[2]);
    axis1weights[0] = gaussian_10;
    axis1weights[1] = gaussian_11;
    axis1weights[2] = gaussian_12;
    axis1weights[3] = axis1weights[1];
    axis1weights[4] = axis1weights[2];
    axis1comp[0] = axis1weights[0];
    axis1comp[1] = cos((params.cyp[1]*y1 + params.cup[1]*u1))* axis1weights[1];
    axis1comp[2] = cos((params.cyp[2]*y1 + params.cup[2]*u1))* axis1weights[2];
    axis1comp[3] = sin((params.cyp[1]*y1 + params.cup[1]*u1))* axis1weights[3];
    axis1comp[4] = sin((params.cyp[2]*y1 + params.cup[2]*u1))* axis1weights[4];

    for(int i=0; i<5; ++i)
        for(int j=0; j<5; ++j) {
            const float weight = axis0weights[i] * axis1weights[j];
            const float comp = axis0comp[i] * axis1comp[j];
            intermediate.data[i*5+j] += vec4(comp*f, weight);
        }
}

/**
* Preintegral MAAF inner integral using Monte Carlo method.
*/
void preIntegralMAAF_ind(
    in MAAFParameters params, 
    in float y0, in float y1,
    in float u0, in float u1,
    in vec3 f, 
    inout MAAFIntermediate intermediate)
{
    float axis0comp[5];
    float axis0weights[5];
    float axis1comp[5];
    float axis1weights[5];

    const float gaussian_y00 = unnormalized_gaussian(u0, params.sigmaup[0]);
    const float gaussian_y01 = unnormalized_gaussian(u0, params.sigmaup[1]);
    const float gaussian_y02 = unnormalized_gaussian(u0, params.sigmaup[2]);
    axis0weights[0] = gaussian_y00;
    axis0weights[1] = gaussian_y01;
    axis0weights[2] = gaussian_y02;
    axis0weights[3] = axis0weights[1];
    axis0weights[4] = axis0weights[2];
    axis0comp[0] = axis0weights[0];
    axis0comp[1] = cos((params.cyp[1]*y0 + params.cup[1]*u0))* axis0weights[1];
    axis0comp[2] = cos((params.cyp[2]*y0 + params.cup[2]*u0))* axis0weights[2];
    axis0comp[3] = sin((params.cyp[1]*y0 + params.cup[1]*u0))* axis0weights[3];
    axis0comp[4] = sin((params.cyp[2]*y0 + params.cup[2]*u0))* axis0weights[4];

    const float gaussian_y10 = unnormalized_gaussian(u1, params.sigmaup[0]);
    const float gaussian_y11 = unnormalized_gaussian(u1, params.sigmaup[1]);
    const float gaussian_y12 = unnormalized_gaussian(u1, params.sigmaup[2]);
    axis1weights[0] = gaussian_y10;
    axis1weights[1] = gaussian_y11;
    axis1weights[2] = gaussian_y12;
    axis1weights[3] = axis1weights[1];
    axis1weights[4] = axis1weights[2];
    axis1comp[0] = axis1weights[0];
    axis1comp[1] = cos((params.cyp[1]*y1 + params.cup[1]*u1))* axis1weights[1];
    axis1comp[2] = cos((params.cyp[2]*y1 + params.cup[2]*u1))* axis1weights[2];
    axis1comp[3] = sin((params.cyp[1]*y1 + params.cup[1]*u1))* axis1weights[3];
    axis1comp[4] = sin((params.cyp[2]*y1 + params.cup[2]*u1))* axis1weights[4];

    for(int i=0; i<5; ++i)
        for(int j=0; j<5; ++j) {
            const float weight = axis0weights[i] * axis1weights[j];
            const float comp = axis0comp[i] * axis1comp[j];
            intermediate.data[i*5+j] += vec4(comp*f, weight);
        }
}

/*
* Compute SPP for multi-effects continue sampling.
* 
*/

vec3 computeSPP(
    in float dir_slope_max,
    in float ind_slope_max,
    in float dof_slope_max,
    in float proj_dist,
    in float omega_x_max_dir,
    in float omega_x_max_ind) 
{
    const int initial_spp = 16;
    const float omega_pix_max = 0.5f;
    const float np_sqrt = (omega_pix_max + omega_x_max_dir) * (1 + dof_slope_max * omega_x_max_dir);
    const float ndir_sqrt = 1 + lightSigma * dir_slope_max * omega_x_max_dir / proj_dist;
    const float nind_sqrt = omega_x_max_ind + ind_slope_max * omega_x_max_ind / proj_dist;
    return vec3(np_sqrt, ndir_sqrt, nind_sqrt);
}

#endif