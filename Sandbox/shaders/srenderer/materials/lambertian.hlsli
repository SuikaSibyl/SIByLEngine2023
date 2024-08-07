#ifndef _SRENDERER_LAMBERTIAN_MATERIAL_
#define _SRENDERER_LAMBERTIAN_MATERIAL_

#include "bxdf.hlsli"
#include "common/sampling.hlsli"

///////////////////////////////////////////////////////////////////////////////////////////
// Lambertian Diffuse Material
// ----------------------------------------------------------------------------------------
// A very simple diffuse material that follows the Lambertian reflectance model.
///////////////////////////////////////////////////////////////////////////////////////////

struct LambertMaterial : IBxDFParameter {
    float3 R; // reflectance
};

struct LambertianBRDF : IBxDF {
    typedef LambertMaterial TParam;
    
    // Evaluate the BSDF
    static float3 eval(ibsdf::eval_in i, LambertMaterial material) {
        if (dot(i.geometric_normal, i.wi) < 0 ||
            dot(i.geometric_normal, i.wo) < 0) {
            // No light below the surface
            return float3(0);
        }
        Frame frame = i.shading_frame;
        // Lambertian BRDF
        return max(dot(frame.n, i.wo), 0.f) *
               material.R * k_inv_pi;
    }
    // importance sample the BSDF
    static ibsdf::sample_out sample(ibsdf::sample_in i, LambertMaterial material) {
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
}

#endif // !_SRENDERER_LAMBERTIAN_MATERIAL_