#ifndef _SRENDERER_CONDUCTOR_BRDF_
#define _SRENDERER_CONDUCTOR_BRDF_

#include "bxdf.hlsli"
#include "srenderer/spt.hlsli"

struct ConductorMaterial : IBxDFParameter {
    float3 eta;  // real component of IoR
    float3 k;    // imaginary component of IoR
    float alpha;
    
    __init() {}
    __init(MaterialData data) {
        k = data.floatvec_0.xyz;
        alpha = data.floatvec_0.w;
        eta = data.floatvec_2.xyz;
    }
};

struct ConductorBRDF : IBxDF {
    typedef ConductorMaterial TParam;

    // Evaluate the BSDF
    static float3 eval(ibsdf::eval_in i, ConductorMaterial material) {
        const Frame frame = i.shading_frame;
        const float3 wi = i.shading_frame.to_local(i.wi);
        const float3 wo = i.shading_frame.to_local(i.wo);
        const float3 wh = normalize(wi + wo);
        
        // Sample rough conductor BRDF
        // Sample microfacet normal wm and reflected direction wi
        IsotropicTrowbridgeReitzParameter params;
        params.alpha = material.alpha;
        float3 f = eval_isotropic_ggx_conductor(wi, wo, wh,
            material.eta, material.k, params);
        return f;
    }

    // importance sample the BSDF
    static ibsdf::sample_out sample(ibsdf::sample_in i, ConductorMaterial material) {
        const Frame frame = i.shading_frame;
        float3 wi = i.shading_frame.to_local(i.wi);
        if (wi.z < 0) { wi.z = -wi.z; i.wi = frame.to_world(wi); }
        ibsdf::sample_out o;
        if (IsotropicTrowbridgeReitzDistribution::effectively_smooth(material.alpha)) {
            float3 f = FresnelComplex(theta_phi_coord::AbsCosTheta(wi),
                                      complex3(material.eta, material.k));
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
        float3 F = FresnelComplex(abs(dot(i.wi, o.wh)), complex3(material.eta, material.k));
        float3 wh = i.shading_frame.to_local(o.wh);
        float3 wo = i.shading_frame.to_local(o.wo);
        float3 f = IsotropicTrowbridgeReitzDistribution::D(wh, params)
                    * IsotropicTrowbridgeReitzDistribution::G(wo, wi, params)
                    * F / (4 * theta_phi_coord::AbsCosTheta(wi));
        o.bsdf = f / o.pdf;
        return o;
    }

    // Evaluate the PDF of the BSDF sampling
    static float pdf(ibsdf::pdf_in i, ConductorMaterial material) {
        const Frame frame = i.shading_frame;
        float3 wi = i.shading_frame.to_local(i.wi);
        const float3 wo = i.shading_frame.to_local(i.wo);
        if (wi.z < 0) { wi.z = -wi.z; i.wi = frame.to_world(wi); }
        const float3 wh = normalize(wi + wo);
        if (IsotropicTrowbridgeReitzDistribution::effectively_smooth(material.alpha)) {
            return 1.f;
        }
        
        IsotropicTrowbridgeReitzParameter params;
        params.alpha = material.alpha;
        const float pdf = IsotropicTrowbridgeReitzDistribution::pdf_vnormal(wi, wh, params);
        const float VdotH = abs(dot(wi, wh));
        return pdf / (4 * abs(VdotH));
    }

    static ConductorMaterial.Differential bwd_eval(
        ibsdf::eval_in i, ConductorMaterial material,
        float3.Differential d_output
    ) {
        const Frame frame = i.shading_frame;
        const float3 wi = i.shading_frame.to_local(i.wi);
        const float3 wo = i.shading_frame.to_local(i.wo);
        const float3 wh = normalize(wi + wo);

        IsotropicTrowbridgeReitzParameter params;
        params.alpha = material.alpha;
        var params_pair = diffPair(params);
        var eta_pair = diffPair(material.eta);
        var k_pair = diffPair(material.k);
        // bwd_diff(eval_isotropic_ggx_conductor)(wi, wo, wh,
        //                                        eta_pair, k_pair, params_pair, d_output);
        bwd_diff(IsotropicTrowbridgeReitzDistribution::D)(wh, params_pair, dot(d_output, float3(1)));
        ConductorMaterial.Differential d_material;
        d_material.alpha = params_pair.d.alpha;
        // d_material.eta = eta_pair.d;
        // d_material.k = k_pair.d;
        return d_material;
    }

    [Differentiable]
    static float3 eval_isotropic_ggx_conductor(
        no_diff float3 wi,
        no_diff float3 wo,
        no_diff float3 wh,
        float3 eta,
        float3 k,
        IsotropicTrowbridgeReitzParameter params
    ) {
        // Evaluate Fresnel factor F for conductor BRDF
        float3 F = FresnelComplex(abs(dot(wi, wh)), complex3(eta, k));
        float3 f = IsotropicTrowbridgeReitzDistribution::D(wh, params)
                    * IsotropicTrowbridgeReitzDistribution::G(wo, wi, params)
                    * F / (4 * theta_phi_coord::AbsCosTheta(wi));
        return f;
    }
}

// struct ConductorBRDFDerivative : IBxDFDerivative {
//     typedef ConductorMaterial TParam;

//     /** Backward derivative of bxdf evaluation */
//     static void bwd_eval(const ibsdf::eval_in i, inout DifferentialPair<ConductorMaterial> param) {
//         IsotropicTrowbridgeReitzParameter ggx_param;
//         ggx_param.alpha = param.p.alpha;
//         DielectricFresnel fresnel = DielectricFresnel(1.0, 2.0);
//         var d_ggx_param = microfacet_reflection::bwd_eval<IsotropicTrowbridgeReitzDistribution>(
//             i, fresnel, ggx_param, float3(1.f));
//         // accumulate the derivatives
//         ConductorMaterial.Differential dparam = param.d;
//         dparam.alpha += d_ggx_param.alpha;
//         param = diffPair(param.p, dparam);
//     }
//     /** sample and compute brdfd with primal importance sample sampling */
//     static ibsdf::dsample_out<ConductorMaterial> sample_primal(
//         const ibsdf::sample_in i, ConductorMaterial material) {
//         // sample the primal BRDF
//         ibsdf::sample_out sample_o = ConductorBRDF::sample(i, material);
//         ibsdf::dsample_out<ConductorMaterial> o;
//         o.wo = sample_o.wo;
//         o.pdf = sample_o.pdf;
//         // evaluate the BRDF derivative
//         ibsdf::eval_in eval_in;
//         eval_in.wi = i.wi;
//         eval_in.wo = o.wo;
//         eval_in.geometric_normal = i.geometric_normal;
//         eval_in.shading_frame = i.shading_frame;
//         DifferentialPair<ConductorMaterial> material_pair = diffPair(material, ConductorMaterial::dzero());
//         ConductorBRDFDerivative::bwd_eval(eval_in, material_pair);
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
//         const ibsdf::sample_in i, ConductorMaterial material) {
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
//     static ibsdf::dsample_out<ConductorMaterial> sample_pos_derivative(
//         const ibsdf::sample_in i, ConductorMaterial material) {
//         IsotropicTrowbridgeReitzParameter ggx_param;
//         ggx_param.alpha = material.alpha;

//         float4 sample_pdf = microfacet_reflection::sample_pos<
//             IsotropicTrowbridgeReitzDerivative>(i, ggx_param);
//         ibsdf::dsample_out<ConductorMaterial> o;
//         o.wo = sample_pdf.xyz;
//         o.pdf = sample_pdf.w;

//         ibsdf::eval_in eval_in;
//         eval_in.wi = i.wi;
//         eval_in.wo = o.wo;
//         eval_in.geometric_normal = i.geometric_normal;
//         eval_in.shading_frame = i.shading_frame;

//         DifferentialPair<ConductorMaterial> material_pair = diffPair(material, ConductorMaterial::dzero());
//         ConductorBRDFDerivative::bwd_eval(eval_in, material_pair);

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
//         const ibsdf::sample_in i, ConductorMaterial material) {
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
//     static ibsdf::dsample_out<ConductorMaterial> sample_neg_derivative(
//         const ibsdf::sample_in i, ConductorMaterial material) {
//         IsotropicTrowbridgeReitzParameter ggx_param;
//         ggx_param.alpha = material.alpha;

//         float4 sample_pdf = microfacet_reflection::sample_neg<
//             IsotropicTrowbridgeReitzDerivative>(i, ggx_param);
//         ibsdf::dsample_out<ConductorMaterial> o;
//         o.wo = sample_pdf.xyz;
//         o.pdf = sample_pdf.w;

//         ibsdf::eval_in eval_in;
//         eval_in.wi = i.wi;
//         eval_in.wo = o.wo;
//         eval_in.geometric_normal = i.geometric_normal;
//         eval_in.shading_frame = i.shading_frame;

//         DifferentialPair<ConductorMaterial> material_pair = diffPair(material, ConductorMaterial::dzero());
//         ConductorBRDFDerivative::bwd_eval(eval_in, material_pair);
        
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
//         const ibsdf::sample_in i, ConductorMaterial material) {
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
//     static float pdf_pos_derivative(const ibsdf::pdf_in i, ConductorMaterial material) {
//         IsotropicTrowbridgeReitzParameter ggx_param;
//         ggx_param.alpha = material.alpha;
//         return microfacet_reflection::pdf_pos<
//             IsotropicTrowbridgeReitzDerivative>(i, ggx_param);
//     }
//     /** The pdf of postivized derivative sampling, negative part */
//     static float pdf_neg_derivative(const ibsdf::pdf_in i, ConductorMaterial material) {
//         IsotropicTrowbridgeReitzParameter ggx_param;
//         ggx_param.alpha = material.alpha;
//         return microfacet_reflection::pdf_neg<
//             IsotropicTrowbridgeReitzDerivative>(i, ggx_param);
//     }
// };

// struct AnisoConductorMaterial : IBxDFParameter {
//     float eta; // real component of IoR
//     float k;   // imaginary component of IoR
//     float alpha_x;
//     float alpha_y;
// };

// struct AnisoConductorBRDF : IBxDF {
//     typedef AnisoConductorMaterial TParam;

//     // Evaluate the BSDF
//     static float3 eval(ibsdf::eval_in i, AnisoConductorMaterial material) {
//         if (dot(i.geometric_normal, i.wi) < 0 ||
//             dot(i.geometric_normal, i.wo) < 0) {
//             // No light below the surface
//             return float3(0);
//         }
//         Frame frame = i.shading_frame;
//         // Lambertian BRDF
//         return 0.f;
//     }
//     // importance sample the BSDF
//     static ibsdf::sample_out sample(ibsdf::sample_in i, AnisoConductorMaterial material) {
//         ibsdf::sample_out o;
//         // For Lambertian, we importance sample the cosine hemisphere domain.
//         if (dot(i.geometric_normal, i.wi) < 0) {
//             // Incoming direction is below the surface.
//             o.bsdf = float3(0);
//             o.wo = float3(0);
//             o.pdf = 0;
//             return o;
//         }

//         AnisotropicTrowbridgeReitzParameter params;
//         params.alpha_x = material.alpha_x;
//         params.alpha_y = material.alpha_y;

//         const Frame frame = i.shading_frame;
//         const float3 wi = i.shading_frame.to_local(i.wi);
//         if (AnisotropicTrowbridgeReitzDistribution::effectively_smooth(params)) {
//             float f = FresnelComplex(theta_phi_coord::AbsCosTheta(wi),
//                                      complex(material.eta, material.k));
//             o.wo = frame.to_world(float3(-wi.x, -wi.y, wi.z));
//             o.pdf = 1.f;
//             o.bsdf = float3(f) / o.pdf;
//             return o;
//         }
//         // Sample rough conductor BRDF
//         // Sample microfacet normal wm and reflected direction wi
//         o = microfacet_reflection::sample_vnormal<
//             AnisotropicTrowbridgeReitzDistribution>(i, params);
//         // Evaluate Fresnel factor F for conductor BRDF
//         float F = FresnelComplex(abs(dot(i.wi, o.wh)), complex(material.eta, material.k));
//         float3 wh = i.shading_frame.to_local(o.wh);
//         float3 wo = i.shading_frame.to_local(o.wo);
//         float3 f = AnisotropicTrowbridgeReitzDistribution::D(wh, params)
//                     * AnisotropicTrowbridgeReitzDistribution::G(wo, wi, params)
//                     * F / (4 * theta_phi_coord::AbsCosTheta(wi));
//         o.bsdf = f / o.pdf;
//         return o;
//     }
//     // Evaluate the PDF of the BSDF sampling
//     float pdf(ibsdf::pdf_in i, AnisoConductorMaterial material) {
//         if (dot(i.geometric_normal, i.wi) < 0 ||
//             dot(i.geometric_normal, i.wo) < 0) {
//             // No light below the surface
//             return float(0);
//         }
//         // Flip the shading frame if it is
//         // inconsistent with the geometry normal.
//         Frame frame = i.shading_frame;

//         return 0.f;
//     }
// }

#endif // !_SRENDERER_CONDUCTOR_BRDF_