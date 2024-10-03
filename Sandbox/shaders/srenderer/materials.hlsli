#ifndef _SRENDERER_MATERIAL_HEADER_
#define _SRENDERER_MATERIAL_HEADER_

#include "materials/lambertian.hlsli"

namespace materials {
ibsdf::sample_out bsdf_sample(ibsdf::sample_in i, MaterialData material) {
    ibsdf::sample_out o;
    switch (material.bxdf_type) {
    case 0: return LambertianBRDF::sample(i, LambertMaterial(material));
    }
    return o;
}

float bsdf_sample_pdf(ibsdf::pdf_in i, MaterialData material) {
    ibsdf::sample_out o;
    switch (material.bxdf_type) {
    case 0: return LambertianBRDF::pdf(i, LambertMaterial(material));
    }
    return 0.f;
}

float3 bsdf_eval(ibsdf::eval_in i, MaterialData material) {
    switch (material.bxdf_type) {
    case 0: return LambertianBRDF::eval(i, LambertMaterial(material));
    }
    return float3(0, 0, 0);
}

float3 albedo(MaterialData material) { return material.floatvec_0.xyz; }
float3 emission(MaterialData material) { return material.floatvec_1.xyz; }
}

#endif // _SRENDERER_MATERIAL_HEADER_