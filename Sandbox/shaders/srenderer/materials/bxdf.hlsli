#ifndef _SRENDERER_BSDF_HEADER_
#define _SRENDERER_BSDF_HEADER_

#include "common/math.hlsli"
#include "common/geometry.hlsli"

namespace ibsdf {
struct eval_in {
    float3 wi;
    float3 wo;
    float3 geometric_normal;
    Frame shading_frame;
};

struct sample_in {
    float2 u;
    float3 wi;
    float3 geometric_normal;
    Frame shading_frame;
};

struct sample_out {
    float3 bsdf;
    float3 wo;
    float pdf;
};

struct pdf_in {
    float3 wi;
    float3 wo;
    float3 geometric_normal;
    Frame shading_frame;
};

float u2theta(float u) { return 
    sqr(u) * (k_pi / 2.f); }
float2 u2theta(float2 u) { return 
    float2(u2theta(u.x), u2theta(u.y)); }
float3 u2theta(float3 u) { return 
    float3(u2theta(u.xy), u2theta(u.z)); }
float4 u2theta(float4 u) { return 
    float4(u2theta(u.xy), u2theta(u.zw)); }

float u2phi(float u) { return 
    (2.f * u - 1.f) * k_pi; }
float2 u2phi(float2 u) { return 
    float2(u2phi(u.x), u2phi(u.y)); }
float3 u2phi(float3 u) { return 
    float3(u2phi(u.xy), u2phi(u.z)); }
float4 u2phi(float4 u) { return 
    float4(u2phi(u.xy), u2phi(u.zw)); }

float theta2u(float theta) { return 
    sqrt(theta * (2.f / k_pi)); }
float2 theta2u(float2 theta) { return 
    float2(theta2u(theta.x), theta2u(theta.y)); }
float3 theta2u(float3 theta) { return 
    float3(theta2u(theta.xy), theta2u(theta.z)); }
float4 theta2u(float4 theta) { return 
    float4(theta2u(theta.xy), theta2u(theta.zw)); }

float phi2u(float phi) { return 
    (phi + k_pi) / (2.f * k_pi); }
float2 phi2u(float2 phi) { return 
    float2(phi2u(phi.x), phi2u(phi.y)); }
float3 phi2u(float3 phi) { return 
    float3(phi2u(phi.xy), phi2u(phi.z)); }
float4 phi2u(float4 phi) { return 
    float4(phi2u(phi.xy), phi2u(phi.zw)); }
}

interface IBxDF {
    // Evaluate the BSDF
    float3 eval(ibsdf::eval_in i);
    // importance sample the BSDF
    ibsdf::sample_out sample(ibsdf::sample_in i);
    // Evaluate the PDF of the BSDF sampling
    float pdf(ibsdf::pdf_in i);
}

#endif // _SRENDERER_BSDF_HEADER_