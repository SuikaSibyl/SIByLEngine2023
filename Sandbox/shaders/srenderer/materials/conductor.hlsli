#ifndef _SRENDERER_CONDUCTOR_BRDF_
#define _SRENDERER_CONDUCTOR_BRDF_

#include "bxdf.hlsli"

struct ConductorMaterial : IBxDFParameter {
    float eta;  // real component of IoR
    float k;    // imaginary component of IoR
    float alpha;
};

struct ConductorBRDF : IBxDF {
    typedef ConductorMaterial TParam;

    // Evaluate the BSDF
    static float3 eval(ibsdf::eval_in i, ConductorMaterial material) {
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
    static ibsdf::sample_out sample(ibsdf::sample_in i, ConductorMaterial material) {
        ibsdf::sample_out o;
        const Frame frame = i.shading_frame;
        const float3 wi = i.shading_frame.to_local(i.wi);
        if (IsotropicTrowbridgeReitzDistribution::effectively_smooth(material.alpha)) {
            float f = FresnelComplex(theta_phi_coord::AbsCosTheta(wi),
                                     complex(material.eta, material.k));
            o.wo = frame.to_world(float3(-wi.x, -wi.y, wi.z));
            o.pdf = 1.f;
            o.bsdf = float3(f) / o.pdf;
            return o;
        }
        // Sample rough conductor BRDF
        // Sample microfacet normal wm and reflected direction wi
        IsotropicTrowbridgeReitzParameter params;
        params.alpha = material.alpha;
        o = microfacet_reflection::sample_vnormal<
            IsotropicTrowbridgeReitzDistribution>(i, params);
        // Evaluate Fresnel factor F for conductor BRDF
        float F = FresnelComplex(abs(dot(i.wi, o.wh)), complex(material.eta, material.k));
        float3 wh = i.shading_frame.to_local(o.wh);
        float3 wo = i.shading_frame.to_local(o.wo);
        float3 f = IsotropicTrowbridgeReitzDistribution::D(wh, params)
                    * IsotropicTrowbridgeReitzDistribution::G(wo, wi, params)
                    * F / (4 * theta_phi_coord::AbsCosTheta(wi));
        o.bsdf = f / o.pdf;
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

// struct ConductorBRDFDerivative : IBxDFDerivative {
//     typedef ConductorMaterial TParam;

//     /** Backward derivative of bxdf evaluation */
//     static void bwd_eval(const ibsdf::eval_in i, inout DifferentialPair<ConductorMaterial> param) {
//         IsotropicTrowbridgeReitzParameter ggx_param;
//         ggx_param.alpha = param.p.alpha;

//         // Evaluate Fresnel factor F for conductor BRDF
//         float3 wi = i.shading_frame.to_local(i.wi);
//         float3 wo = i.shading_frame.to_local(i.wo);
//         float3 wh = normalize(wi + wo);
//         float F = FresnelComplex(abs(dot(wi, wh)), complex(param.p.eta, param.p.k));

//         float D = IsotropicTrowbridgeReitzDistribution::D(wh, ggx_param);
//         float G = IsotropicTrowbridgeReitzDistribution::G(wo, wi, ggx_param);
//         float tmp = F / (4 * theta_phi_coord::AbsCosTheta(wi));

//         var ggx_param_pair = diffPair(ggx_param);
//         bwd_diff(IsotropicTrowbridgeReitzDistribution::D)(wh, ggx_param_pair, float(1));

//         // accumulate the derivatives
//         ConductorMaterial.Differential dparam = param.d;
//         dparam.alpha += ggx_param_pair.d.alpha;
//         param = diffPair(param.p, dparam);
//     }

//     /** sample and compute brdfd with primal importance sample sampling */
//     static ibsdf::dsample_out<PlasticMaterial> sample_primal(
//         const ibsdf::sample_in i, PlasticMaterial material) {
//         // sample the primal BRDF
//         ibsdf::sample_out sample_o = PlasticBRDF::sample(i, material);
//         ibsdf::dsample_out<PlasticMaterial> o;
//         o.wo = sample_o.wo;
//         o.pdf = sample_o.pdf;
//         // evaluate the BRDF derivative
//         ibsdf::eval_in eval_in;
//         eval_in.wi = i.wi;
//         eval_in.wo = o.wo;
//         eval_in.geometric_normal = i.geometric_normal;
//         eval_in.shading_frame = i.shading_frame;
//         DifferentialPair<PlasticMaterial> material_pair = diffPair(material, PlasticMaterial::dzero());
//         PlasticBRDFDerivative::bwd_eval(eval_in, material_pair);
//         o.dparam = material_pair.d;
//         o.dparam.alpha /= o.pdf; // divide by the pdf
//         if (o.pdf == 0) o.dparam.alpha = 0.f;
//         if (isnan(o.dparam.alpha)) o.dparam.alpha = 0.f;
//         // reject samples below the surface
//         const float3 wo = i.shading_frame.to_local(o.wo);
//         if (wo.z < 0.f || o.pdf == 0.f) {
//             o.dparam.alpha = 0.f;
//         }
//         return o;
//     }

//     /** sample but not compute brdfd with primal importance sample sampling */
//     static ibsdf::dsample_noeval_out sample_noeval_primal(
//         const ibsdf::sample_in i, PlasticMaterial material) {
//         // sample the primal BRDF
//         IsotropicTrowbridgeReitzParameter params;
//         params.alpha = material.alpha;
//         ibsdf::sample_out sample_o = microfacet_reflection::sample_normal<
//             IsotropicTrowbridgeReitzDistribution>(i, params);
//         ibsdf::dsample_noeval_out o;
//         o.wo = sample_o.wo;
//         o.pdf = sample_o.pdf;
//         o.wh = sample_o.wh;
//         o.pdf_wh = sample_o.pdf_wh;
//         return o;
//     }

//     /** The pdf of postivized derivative sampling, positive part */
//     static ibsdf::dsample_out<PlasticMaterial> sample_pos_derivative(
//         const ibsdf::sample_in i, PlasticMaterial material) {
//         IsotropicTrowbridgeReitzParameter ggx_param;
//         ggx_param.alpha = material.alpha;

//         float4 sample_pdf = microfacet_reflection::sample_pos<
//             IsotropicTrowbridgeReitzDerivative>(i, ggx_param);
//         ibsdf::dsample_out<PlasticMaterial> o;
//         o.wo = sample_pdf.xyz;
//         o.pdf = sample_pdf.w;

//         ibsdf::eval_in eval_in;
//         eval_in.wi = i.wi;
//         eval_in.wo = o.wo;
//         eval_in.geometric_normal = i.geometric_normal;
//         eval_in.shading_frame = i.shading_frame;

//         DifferentialPair<PlasticMaterial> material_pair = diffPair(material, PlasticMaterial::dzero());
//         PlasticBRDFDerivative::bwd_eval(eval_in, material_pair);

//         o.dparam = material_pair.d;
//         o.dparam.alpha /= o.pdf;
//         if (o.pdf == 0) o.dparam.alpha = 0.f;
//         if (isnan(o.dparam.alpha)) o.dparam.alpha = 0.f;

//         // reject samples below the surface
//         const float3 wo = i.shading_frame.to_local(o.wo);
//         if (wo.z < 0.f || o.pdf == 0.f) {
//             o.dparam.alpha = 0.f;
//         }

//         return o;
//     }

//     /** sample but not compute brdfd with postivized derivative sampling, positive */
//     static ibsdf::dsample_noeval_out sample_noeval_pos_derivative(
//         const ibsdf::sample_in i, PlasticMaterial material) {
//         IsotropicTrowbridgeReitzParameter ggx_param;
//         ggx_param.alpha = material.alpha;

//         float4 sample_pdf = microfacet_reflection::sample_pos<
//             IsotropicTrowbridgeReitzDerivative>(i, ggx_param);
//         ibsdf::dsample_noeval_out o;

//         const float3 wi = i.shading_frame.to_local(i.wi);
//         const float3 wh = sample_pdf.xyz;
//         o.wo = i.shading_frame.to_world(reflect(-wi, wh));
//         o.pdf = sample_pdf.w / (4 * abs(dot(wi, wh)));
//         o.wh = i.shading_frame.to_world(wh);
//         o.pdf_wh = sample_pdf.w;
//         return o;
//     }

//     /** The pdf of postivized derivative sampling, negative part */
//     static ibsdf::dsample_out<PlasticMaterial> sample_neg_derivative(
//         const ibsdf::sample_in i, PlasticMaterial material) {
//         IsotropicTrowbridgeReitzParameter ggx_param;
//         ggx_param.alpha = material.alpha;

//         float4 sample_pdf = microfacet_reflection::sample_neg<
//             IsotropicTrowbridgeReitzDerivative>(i, ggx_param);
//         ibsdf::dsample_out<PlasticMaterial> o;
//         o.wo = sample_pdf.xyz;
//         o.pdf = sample_pdf.w;

//         ibsdf::eval_in eval_in;
//         eval_in.wi = i.wi;
//         eval_in.wo = o.wo;
//         eval_in.geometric_normal = i.geometric_normal;
//         eval_in.shading_frame = i.shading_frame;

//         DifferentialPair<PlasticMaterial> material_pair = diffPair(material, PlasticMaterial::dzero());
//         PlasticBRDFDerivative::bwd_eval(eval_in, material_pair);

//         o.dparam = material_pair.d;
//         o.dparam.alpha /= o.pdf;
//         if (o.pdf == 0) o.dparam.alpha = 0.f;
//         if (isnan(o.dparam.alpha)) o.dparam.alpha = 0.f;

//         // reject samples below the surface
//         const float3 wo = i.shading_frame.to_local(o.wo);
//         if (wo.z < 0.f || o.pdf == 0.f) {
//             o.dparam.alpha = 0.f;
//         }

//         return o;
//     }

//     /** sample but not compute brdfd with postivized derivative sampling, negative */
//     static ibsdf::dsample_noeval_out sample_noeval_neg_derivative(
//         const ibsdf::sample_in i, PlasticMaterial material) {
//         IsotropicTrowbridgeReitzParameter ggx_param;
//         ggx_param.alpha = material.alpha;

//         float4 sample_pdf = microfacet_reflection::sample_neg<
//             IsotropicTrowbridgeReitzDerivative>(i, ggx_param);
//         ibsdf::dsample_noeval_out o;
//         o.wo = sample_pdf.xyz;
//         o.pdf = sample_pdf.w;
//         return o;
//     }

//     /** The pdf of postivized derivative sampling, positive part */
//     static float pdf_pos_derivative(const ibsdf::pdf_in i, PlasticMaterial material) {
//         IsotropicTrowbridgeReitzParameter ggx_param;
//         ggx_param.alpha = material.alpha;
//         return microfacet_reflection::pdf_pos<
//             IsotropicTrowbridgeReitzDerivative>(i, ggx_param);
//     }
//     /** The pdf of postivized derivative sampling, negative part */
//     static float pdf_neg_derivative(const ibsdf::pdf_in i, PlasticMaterial material) {
//         IsotropicTrowbridgeReitzParameter ggx_param;
//         ggx_param.alpha = material.alpha;
//         return microfacet_reflection::pdf_neg<
//             IsotropicTrowbridgeReitzDerivative>(i, ggx_param);
//     }
// };

struct AnisoConductorMaterial : IBxDFParameter {
    float eta; // real component of IoR
    float k;   // imaginary component of IoR
    float alpha_x;
    float alpha_y;
};

struct AnisoConductorBRDF : IBxDF {
    typedef AnisoConductorMaterial TParam;

    // Evaluate the BSDF
    static float3 eval(ibsdf::eval_in i, AnisoConductorMaterial material) {
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
    static ibsdf::sample_out sample(ibsdf::sample_in i, AnisoConductorMaterial material) {
        ibsdf::sample_out o;
        // For Lambertian, we importance sample the cosine hemisphere domain.
        if (dot(i.geometric_normal, i.wi) < 0) {
            // Incoming direction is below the surface.
            o.bsdf = float3(0);
            o.wo = float3(0);
            o.pdf = 0;
            return o;
        }

        AnisotropicTrowbridgeReitzParameter params;
        params.alpha_x = material.alpha_x;
        params.alpha_y = material.alpha_y;

        const Frame frame = i.shading_frame;
        const float3 wi = i.shading_frame.to_local(i.wi);
        if (AnisotropicTrowbridgeReitzDistribution::effectively_smooth(params)) {
            float f = FresnelComplex(theta_phi_coord::AbsCosTheta(wi),
                                     complex(material.eta, material.k));
            o.wo = frame.to_world(float3(-wi.x, -wi.y, wi.z));
            o.pdf = 1.f;
            o.bsdf = float3(f) / o.pdf;
            return o;
        }
        // Sample rough conductor BRDF
        // Sample microfacet normal wm and reflected direction wi
        o = microfacet_reflection::sample_vnormal<
            AnisotropicTrowbridgeReitzDistribution>(i, params);
        // Evaluate Fresnel factor F for conductor BRDF
        float F = FresnelComplex(abs(dot(i.wi, o.wh)), complex(material.eta, material.k));
        float3 wh = i.shading_frame.to_local(o.wh);
        float3 wo = i.shading_frame.to_local(o.wo);
        float3 f = AnisotropicTrowbridgeReitzDistribution::D(wh, params)
                    * AnisotropicTrowbridgeReitzDistribution::G(wo, wi, params)
                    * F / (4 * theta_phi_coord::AbsCosTheta(wi));
        o.bsdf = f / o.pdf;
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

#endif // !_SRENDERER_CONDUCTOR_BRDF_