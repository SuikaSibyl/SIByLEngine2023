#ifndef _SRENDERER_ORENNAYAR_MATERIAL_
#define _SRENDERER_ORENNAYAR_MATERIAL_

#include "bxdf.hlsli"
#include "common/math.hlsli"
#include "common/sampling.hlsli"

///////////////////////////////////////////////////////////////////////////////////////////
// Oren-Nayar Diffuse Material
// ----------------------------------------------------------------------------------------
// Rough surfaces generally appear brighter when the illumination direction
// approaches the viewing direction. Oren-Nayar follows this observation by
// modeling the rough surfaces by V-shape microfacets described by a spherical
// Gaussian distribution with a single parameter sigma.
///////////////////////////////////////////////////////////////////////////////////////////

struct OrenNayarMaterial : IBxDFParameter{
    float3 R;
    float sigma; // Gaussian deviation parameter in radians.
    float A;
    float B;

    __init(float3 _R, float _sigma) {
        this.R = _R;
        this.sigma = _sigma;
        float sigma = radians(this.sigma);
        float sigma2 = sigma * sigma;
        this.A = 1.f - (sigma2 / (2.f * (sigma2 + 0.33f)));
        this.B = 0.45f * sigma2 / (sigma2 + 0.09f);
    }
};

struct OrenNayarBRDF : IBxDF {
    typedef OrenNayarMaterial TParam;

    // Evaluate the BSDF
    static float3 eval(ibsdf::eval_in i, OrenNayarMaterial material) {
        const float3 wi = i.shading_frame.to_local(i.wi);
        const float3 wo = i.shading_frame.to_local(i.wo);
        if (dot(i.geometric_normal, i.wi) < 0 ||
            dot(i.geometric_normal, i.wo) < 0) {
            // No light below the surface
            return float3(0);
        }
        
        float sinThetaI = theta_phi_coord::SinTheta(wi);
        float sinThetaO = theta_phi_coord::SinTheta(wo);
        // Compute cosine term of Oren–Nayar model
        float maxCos = 0;
        if (sinThetaI > 1e-4 && sinThetaO > 1e-4) {
            float sinPhiI = theta_phi_coord::SinPhi(wi);
            float cosPhiI = theta_phi_coord::CosPhi(wi);
            float sinPhiO = theta_phi_coord::SinPhi(wo);
            float cosPhiO = theta_phi_coord::CosPhi(wo);
            float dCos = cosPhiI * cosPhiO + sinPhiI * sinPhiO;
            maxCos = max(0.0, dCos);
        }
        // Compute sine and tangent terms of Oren–Nayar model
        float sinAlpha;
        float tanBeta;
        if (theta_phi_coord::AbsCosTheta(wi) > 
            theta_phi_coord::AbsCosTheta(wo)) {
            sinAlpha = sinThetaO;
            tanBeta = sinThetaI / theta_phi_coord::AbsCosTheta(wi);
        } else {
            sinAlpha = sinThetaI;
            tanBeta = sinThetaO / theta_phi_coord::AbsCosTheta(wo);
        }
        return material.R * k_inv_pi * (material.A + material.B * maxCos * sinAlpha * tanBeta)
                 * max(dot(i.shading_frame.n, i.wo), 0.f);
    }
    // importance sample the BSDF
    static ibsdf::sample_out sample(ibsdf::sample_in i, OrenNayarMaterial material) {
        ibsdf::sample_out o;
        // For Lambertian, we importance sample the cosine hemisphere domain.
        if (dot(i.geometric_normal, i.wi) < 0) {
            // Incoming direction is below the surface.
            o.bsdf = float3(0);
            o.wo = float3(0);
            o.pdf = 0;
            return o;
        }

        // Flip the shading frame if it is
        // inconsistent with the geometry normal.
        Frame frame = i.shading_frame;
        o.wo = frame.to_world(sample_cos_hemisphere(i.u.xy));
        o.pdf = max(dot(frame.n, o.wo), 0.f) * k_inv_pi;

        // evaluate the BSDF
        ibsdf::eval_in eval_in;
        eval_in.wi = i.wi;
        eval_in.wo = o.wo;
        eval_in.geometric_normal = i.geometric_normal;
        eval_in.shading_frame = i.shading_frame;
        o.bsdf = eval(eval_in, material) / o.pdf;
        return o;
    }
    // Evaluate the PDF of the BSDF sampling
    float pdf(ibsdf::pdf_in i) {
        if (dot(i.geometric_normal, i.wi) < 0 ||
            dot(i.geometric_normal, i.wo) < 0) {
            // No light below the surface
            return float(0);
        }
        // Flip the shading frame if it is
        // inconsistent with the geometry normal.
        Frame frame = i.shading_frame;
        return max(dot(frame.n, i.wo), float(0)) * k_inv_pi;
    }
};

#endif // _SRENDERER_ORENNAYAR_MATERIAL_