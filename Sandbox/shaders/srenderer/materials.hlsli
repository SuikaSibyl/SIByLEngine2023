#ifndef _SRENDERER_MATERIAL_HEADER_
#define _SRENDERER_MATERIAL_HEADER_

#include "materials/lambertian.hlsli"
#include "materials/conductor.hlsli"

namespace materials {
ibsdf::sample_out bsdf_sample(ibsdf::sample_in i, MaterialData material, float2 uv) {
    ibsdf::sample_out o;
    switch (material.bxdf_type) {
    case 0: return LambertianBRDF::sample(i, LambertMaterial(material, uv));
    case 1: return ConductorBRDF::sample(i, ConductorMaterial(material));
    }
    return o;
}

float bsdf_sample_pdf(ibsdf::pdf_in i, MaterialData material, float2 uv) {
    ibsdf::sample_out o;
    switch (material.bxdf_type) {
    case 0: return LambertianBRDF::pdf(i, LambertMaterial(material, uv));
    case 1: return ConductorBRDF::pdf(i, ConductorMaterial(material));
    }
    return 0.f;
}

float3 bsdf_eval(ibsdf::eval_in i, MaterialData material, float2 uv) {
    switch (material.bxdf_type) {
    case 0: return LambertianBRDF::eval(i, LambertMaterial(material, uv));
    case 1: return ConductorBRDF::eval(i, ConductorMaterial(material));
    }
    return float3(0, 0, 0);
}

float3 albedo(MaterialData material, float2 uv) { return material.floatvec_0.xyz * sampleTexture(material.albedo_tex, uv).rgb; }
float3 emission(MaterialData material) { return material.floatvec_1.xyz; }
}

#endif // _SRENDERER_MATERIAL_HEADER_