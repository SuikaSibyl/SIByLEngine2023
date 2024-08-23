#ifndef _SRENDERER_DIELECTRIC_BRDF_
#define _SRENDERER_DIELECTRIC_BRDF_

#include "bxdf.hlsli"

struct DielectricMaterial : IBxDFParameter {
    float eta; // IoR
    float alpha;
};

struct DielectricBRDF : IBxDF {
    typedef DielectricMaterial TParam;

    // Evaluate the BSDF
    static float3 eval(ibsdf::eval_in i, DielectricMaterial material) {
        if (dot(i.geometric_normal, i.wi) < 0 ||
            dot(i.geometric_normal, i.wo) < 0) {
            // No light below the surface
            return float3(0);
        }
        Frame frame = i.shading_frame;
        // Lambertian BRDF
        return 0.f;
    }
    // importance sample the BSDF
    static ibsdf::sample_out sample(ibsdf::sample_in i, DielectricMaterial material) {
        ibsdf::sample_out o;
        o.bsdf = float3(0);
        o.wo = float3(0);
        o.pdf = 0;

        if (IsotropicTrowbridgeReitzDistribution::effectively_smooth(material.alpha)) {
            const Frame frame = i.shading_frame;
            const float3 wi = i.shading_frame.to_local(i.wi);
            // Sample perfect specular dielectric BSDF
            const float R = FresnelDielectric(theta_phi_coord::CosTheta(wi), material.eta);
            const float T = 1 - R;
            // Compute probabilities pr and pt for sampling reflection and transmission
            const float pr = R; const float pt = T;
            if (pr == 0 && pt == 0) return o;

            if (i.u.z < pr / (pr + pt)) {
                // Sample perfect specular dielectric BRDF
                o.wo = frame.to_world(float3(-wi.x, -wi.y, wi.z));
                const float3 fr = R;
                o.pdf = pr / (pr + pt);
                o.bsdf = fr / o.pdf;
                return o;
            } else {
                // Sample perfect specular dielectric BTDF
                // Compute ray direction for specular transmission
                float3 wo = safe_refract(wi, float3(0, 0, 1), material.eta);
                if (all(wo == 0)) return o;

                const float3 ft = T;
                // Account for non-symmetry with transmission to different medium
                o.wo = frame.to_world(wo);
                o.pdf = pt / (pr + pt);
                o.bsdf = ft / o.pdf;
                return o;
            }
        } else {
            // Sample rough dielectric BSDF
            IsotropicTrowbridgeReitzParameter params;
            params.alpha = material.alpha;
            ibsdf::sample_out o = microfacet_reflection::sample_vnormal<
                IsotropicTrowbridgeReitzDistribution>(i, params);
            
            const float3 wi = i.shading_frame.to_local(i.wi);
            const float R = FresnelDielectric(dot(i.wi, o.wh), material.eta);
            const float T = 1 - R;
            const float pr = R; const float pt = T;
            if (i.u.z < pr / (pr + pt)) {
                // Sample reflection at rough dielectric interface
                float3 wh = i.shading_frame.to_local(o.wh);
                float3 wo = i.shading_frame.to_local(o.wo);
                if(wo.z < 0) {
                    o.bsdf = float3(0);
                    o.wo = float3(0);
                    o.pdf = 0;
                    return o;
                }
                float3 f = IsotropicTrowbridgeReitzDistribution::D(wh, params)
                     * IsotropicTrowbridgeReitzDistribution::G(wo, wi, params)
                     * R / (4 * theta_phi_coord::AbsCosTheta(wi));
                o.pdf = o.pdf_wh / (4 * abs(dot(wi, wh))) * pr / (pr + pt);
                o.bsdf = f / o.pdf;
                return o;
            } else {
                // Sample transmission at rough dielectric interface
                float eta = material.eta;
                float3 wh = i.shading_frame.to_local(o.wh);
                float3 wo = safe_refract(wi, wh, eta);
                if (all(wo == 0)) {
                    o.bsdf = float3(0);
                    o.wo = float3(0);
                    o.pdf = 0;
                    return o;
                }
                o.wo = i.shading_frame.to_world(wo);
                // Compute PDF of rough dielectric transmission
                float denom = sqr(dot(wo, wh) + dot(wi, wh) / eta);
                float dwm_dwi = abs(dot(wo, wh)) / denom;
                o.pdf = o.pdf_wh * dwm_dwi * pt / (pr + pt);
                // Evaluate BRDF and return BSDFSample for rough transmission
                float3 ft = IsotropicTrowbridgeReitzDistribution::D(wh, params)
                     * IsotropicTrowbridgeReitzDistribution::G(wo, wi, params)
                     * T * abs(dot(wi, wh) * dot(wo, wh) / (theta_phi_coord::AbsCosTheta(wi) * denom));
                // Account for non-symmetry with transmission to different medium
                ft /= sqr(eta);
                o.bsdf = ft / o.pdf;
            }
            return o;
        }
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

        return 0.f;
    }
}

#endif // !_SRENDERER_DIELECTRIC_BRDF_