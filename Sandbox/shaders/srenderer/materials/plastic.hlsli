#ifndef _SRENDERER_GGX_BRDF_MATERIAL_
#define _SRENDERER_GGX_BRDF_MATERIAL_

#include "bxdf.hlsli"
#include "common/math.hlsli"
#include "common/sampling.hlsli"
#include "common/microfacet.hlsli"

///////////////////////////////////////////////////////////////////////////////////////////
// Plastic Material
// ----------------------------------------------------------------------------------------
// Plastic can be modeled as a mixture of a diffuse and glossy scattering
// function with parameters controlling the particular colors and
// specular highlight size. The parameters to PlasticMaterial are
// two reflectivities, Kd and Ks, which respectively control the
// amounts of diffuse reflection and glossy specular reflection.
///////////////////////////////////////////////////////////////////////////////////////////

struct PlasticMaterial : IBxDFParameter {
    float alpha;
};

struct PlasticBRDF : IBxDF {
    typedef PlasticMaterial TParam;

    float3 R;
    IsotropicTrowbridgeReitzParameter params;
    IsotropicTrowbridgeReitzDistribution distribution;
    DielectricFresnel fresnel;

    __init() {
        params = IsotropicTrowbridgeReitzParameter(0.5);
        R = float3(1);
        fresnel = DielectricFresnel(1.0, 2.0);
    }
    // Evaluate the BSDF
    static float3 eval(ibsdf::eval_in i, PlasticMaterial material) {
        IsotropicTrowbridgeReitzParameter params;
        params.alpha = material.alpha;
        DielectricFresnel fresnel = DielectricFresnel(1.0, 2.0);
        float3 R = float3(1);
        return microfacet_reflection::eval<IsotropicTrowbridgeReitzDistribution>(i, fresnel, params, R);
    }
    // importance sample the BSDF
    static ibsdf::sample_out sample(ibsdf::sample_in i, PlasticMaterial material) {
        IsotropicTrowbridgeReitzParameter params;
        params.alpha = material.alpha;
        ibsdf::sample_out o = microfacet_reflection::sample_vnormal<
            IsotropicTrowbridgeReitzDistribution>(i, params);
        
        // evaluate the BSDF
        ibsdf::eval_in eval_in;
        eval_in.wi = i.wi;
        eval_in.wo = o.wo;
        eval_in.geometric_normal = i.geometric_normal;
        eval_in.shading_frame = i.shading_frame;
        o.bsdf = eval(eval_in, material) / o.pdf;
        
        // // reject samples below the surface
        // const float3 wo = i.shading_frame.to_local(o.wo);
        // if (wo.z < 0.f || o.pdf == 0.f) {
        //     o.bsdf = float3(0);
        //     o.pdf = 0.f;
        // }
        
        return o;
    }
    // Evaluate the PDF of the BSDF sampling
    float pdf(ibsdf::pdf_in i) {
        return 0.f;
        // return microfacet_reflection::pdf(i, distribution, params);
    }
};

struct PlasticBRDFDerivative : IBxDFDerivative {
    typedef PlasticMaterial TParam;

    /** Backward derivative of bxdf evaluation */
    static void bwd_eval(const ibsdf::eval_in i, inout DifferentialPair<PlasticMaterial> param) {
        IsotropicTrowbridgeReitzParameter ggx_param;
        ggx_param.alpha = param.p.alpha;
        DielectricFresnel fresnel = DielectricFresnel(1.0, 2.0);
        var d_ggx_param = microfacet_reflection::bwd_eval<IsotropicTrowbridgeReitzDistribution>(
            i, fresnel, ggx_param, float3(1.f));
        // accumulate the derivatives
        PlasticMaterial.Differential dparam = param.d;
        dparam.alpha += d_ggx_param.alpha;
        param = diffPair(param.p, dparam);
    }
    /** sample and compute brdfd with primal importance sample sampling */
    static ibsdf::dsample_out<PlasticMaterial> sample_primal(
        const ibsdf::sample_in i, PlasticMaterial material) {
        // sample the primal BRDF
        ibsdf::sample_out sample_o = PlasticBRDF::sample(i, material);
        ibsdf::dsample_out<PlasticMaterial> o;
        o.wo = sample_o.wo;
        o.pdf = sample_o.pdf;
        // evaluate the BRDF derivative
        ibsdf::eval_in eval_in;
        eval_in.wi = i.wi;
        eval_in.wo = o.wo;
        eval_in.geometric_normal = i.geometric_normal;
        eval_in.shading_frame = i.shading_frame;
        DifferentialPair<PlasticMaterial> material_pair = diffPair(material, PlasticMaterial::dzero());
        PlasticBRDFDerivative::bwd_eval(eval_in, material_pair);
        o.dparam = material_pair.d;
        o.dparam.alpha /= o.pdf; // divide by the pdf
        if (o.pdf == 0) o.dparam.alpha = 0.f;
        if (isnan(o.dparam.alpha)) o.dparam.alpha = 0.f;
        // reject samples below the surface
        const float3 wo = i.shading_frame.to_local(o.wo);
        if (wo.z < 0.f || o.pdf == 0.f) {
            o.dparam.alpha = 0.f;
        }
        return o;
    }

    /** sample but not compute brdfd with primal importance sample sampling */
    static ibsdf::dsample_noeval_out sample_noeval_primal(
        const ibsdf::sample_in i, PlasticMaterial material) {
        // sample the primal BRDF
        IsotropicTrowbridgeReitzParameter params;
        params.alpha = material.alpha;
        ibsdf::sample_out sample_o = microfacet_reflection::sample_normal<
            IsotropicTrowbridgeReitzDistribution>(i, params);
        ibsdf::dsample_noeval_out o;
        o.wo = sample_o.wo;
        o.pdf = sample_o.pdf;
        o.wh = sample_o.wh;
        o.pdf_wh = sample_o.pdf_wh;
        return o;
    }

    /** The pdf of postivized derivative sampling, positive part */
    static ibsdf::dsample_out<PlasticMaterial> sample_pos_derivative(
        const ibsdf::sample_in i, PlasticMaterial material) {
        IsotropicTrowbridgeReitzParameter ggx_param;
        ggx_param.alpha = material.alpha;
        
        float4 sample_pdf = microfacet_reflection::sample_pos<
            IsotropicTrowbridgeReitzDerivative>(i, ggx_param);
        ibsdf::dsample_out<PlasticMaterial> o;
        o.wo = sample_pdf.xyz;
        o.pdf = sample_pdf.w;
        
        ibsdf::eval_in eval_in;
        eval_in.wi = i.wi;
        eval_in.wo = o.wo;
        eval_in.geometric_normal = i.geometric_normal;
        eval_in.shading_frame = i.shading_frame;

        DifferentialPair<PlasticMaterial> material_pair = diffPair(material, PlasticMaterial::dzero());
        PlasticBRDFDerivative::bwd_eval(eval_in, material_pair);
        
        o.dparam = material_pair.d;
        o.dparam.alpha /= o.pdf;
        if (o.pdf == 0) o.dparam.alpha = 0.f;
        if (isnan(o.dparam.alpha)) o.dparam.alpha = 0.f;

        // reject samples below the surface
        const float3 wo = i.shading_frame.to_local(o.wo);
        if (wo.z < 0.f || o.pdf == 0.f) {
            o.dparam.alpha = 0.f;
        }

        return o;
    }

    /** sample but not compute brdfd with postivized derivative sampling, positive */
    static ibsdf::dsample_noeval_out sample_noeval_pos_derivative(
        const ibsdf::sample_in i, PlasticMaterial material) {
        IsotropicTrowbridgeReitzParameter ggx_param;
        ggx_param.alpha = material.alpha;
        
        float4 sample_pdf = microfacet_reflection::sample_pos<
            IsotropicTrowbridgeReitzDerivative>(i, ggx_param);
        ibsdf::dsample_noeval_out o;
        
        const float3 wi = i.shading_frame.to_local(i.wi);
        const float3 wh = sample_pdf.xyz;
        o.wo = i.shading_frame.to_world(reflect(-wi, wh));
        o.pdf = sample_pdf.w / (4 * abs(dot(wi, wh)));
        o.wh = i.shading_frame.to_world(wh);
        o.pdf_wh = sample_pdf.w;
        return o;
    }

    /** The pdf of postivized derivative sampling, negative part */
    static ibsdf::dsample_out<PlasticMaterial> sample_neg_derivative(
        const ibsdf::sample_in i, PlasticMaterial material) {
        IsotropicTrowbridgeReitzParameter ggx_param;
        ggx_param.alpha = material.alpha;

        float4 sample_pdf = microfacet_reflection::sample_neg<
            IsotropicTrowbridgeReitzDerivative>(i, ggx_param);
        ibsdf::dsample_out<PlasticMaterial> o;
        o.wo = sample_pdf.xyz;
        o.pdf = sample_pdf.w;

        ibsdf::eval_in eval_in;
        eval_in.wi = i.wi;
        eval_in.wo = o.wo;
        eval_in.geometric_normal = i.geometric_normal;
        eval_in.shading_frame = i.shading_frame;

        DifferentialPair<PlasticMaterial> material_pair = diffPair(material, PlasticMaterial::dzero());
        PlasticBRDFDerivative::bwd_eval(eval_in, material_pair);
        
        o.dparam = material_pair.d;
        o.dparam.alpha /= o.pdf;
        if (o.pdf == 0) o.dparam.alpha = 0.f;
        if (isnan(o.dparam.alpha)) o.dparam.alpha = 0.f;
        
        // reject samples below the surface
        const float3 wo = i.shading_frame.to_local(o.wo);
        if (wo.z < 0.f || o.pdf == 0.f) {
            o.dparam.alpha = 0.f;
        }
        
        return o;
    }

    /** sample but not compute brdfd with postivized derivative sampling, negative */
    static ibsdf::dsample_noeval_out sample_noeval_neg_derivative(
        const ibsdf::sample_in i, PlasticMaterial material) {
        IsotropicTrowbridgeReitzParameter ggx_param;
        ggx_param.alpha = material.alpha;
        
        float4 sample_pdf = microfacet_reflection::sample_neg<
            IsotropicTrowbridgeReitzDerivative>(i, ggx_param);
        ibsdf::dsample_noeval_out o;
        o.wo = sample_pdf.xyz;
        o.pdf = sample_pdf.w;
        return o;
    }

    /** The pdf of postivized derivative sampling, positive part */
    static float pdf_pos_derivative(const ibsdf::pdf_in i, PlasticMaterial material) {
        IsotropicTrowbridgeReitzParameter ggx_param;
        ggx_param.alpha = material.alpha;
        return microfacet_reflection::pdf_pos<
            IsotropicTrowbridgeReitzDerivative>(i, ggx_param);
    }
    /** The pdf of postivized derivative sampling, negative part */
    static float pdf_neg_derivative(const ibsdf::pdf_in i, PlasticMaterial material) {
        IsotropicTrowbridgeReitzParameter ggx_param;
        ggx_param.alpha = material.alpha;
        return microfacet_reflection::pdf_neg<
            IsotropicTrowbridgeReitzDerivative>(i, ggx_param);
    }
};

#endif // _SRENDERER_GGX_BRDF_MATERIAL_