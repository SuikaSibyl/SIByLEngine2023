#ifndef _SRENDERER_LAMBERTIAN_MATERIAL_
#define _SRENDERER_LAMBERTIAN_MATERIAL_

#include "../spt.hlsli"
#include "../../common/math.hlsli"

struct LambertMaterial : IDifferentiable {
    float3 albedo;
};

[Differentiable]
float3 EvalLambertian(
    LambertMaterial material,
    no_diff BSDFEvalGeometry evalGeom
) {
    if (dot(evalGeom.geometric_normal, evalGeom.dir_in) < 0 ||
        dot(evalGeom.geometric_normal, evalGeom.dir_out) < 0) {
        // No light below the surface
        return float3(0);
    }
    // Making sure the shading frame is consistent with the view direction.
    float3x3 frame = evalGeom.frame;
    const float3 demodulate = saturate(dot(frame[2], evalGeom.dir_out)) / k_pi;
    return material.albedo * demodulate;
}

#include "bxdf.hlsli"
#include "common/sampling.hlsli"

struct LambertianBRDF : IBxDF {
    /// Reflectance of the material
    float3 reflectance;

    __init(float3 reflectance) {
        this.reflectance = reflectance;
    }
    
    // Evaluate the BSDF
    float3 eval(ibsdf::eval_in i) {
        if (dot(i.geometric_normal, i.wi) < 0 ||
            dot(i.geometric_normal, i.wo) < 0) {
            // No light below the surface
            return float3(0);
        }
        Frame frame = i.shading_frame;
        // Lambertian BRDF
        return max(dot(frame.n, i.wo), 0.f) *
               reflectance * k_inv_pi;
    }
    // importance sample the BSDF
    ibsdf::sample_out sample(ibsdf::sample_in i) {
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
        o.wo = frame.to_world(sample_cos_hemisphere(i.u));
        o.pdf = max(dot(frame.n, o.wo), 0.f) * k_inv_pi;
        o.bsdf = reflectance;
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
}

#endif // !_SRENDERER_LAMBERTIAN_MATERIAL_