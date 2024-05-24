#ifndef _SRENDERER_PRIMITIVE_HEADER_
#define _SRENDERER_PRIMITIVE_HEADER_

namespace ibsdf {

}

interface IPrimitive {
    // Evaluate the BSDF
    float3 eval(ibsdf::eval_in i);
    // importance sample the BSDF
    ibsdf::sample_out sample(ibsdf::sample_in i);
    // Evaluate the PDF of the BSDF sampling
    float pdf(ibsdf::pdf_in i);
}


#endif // _SRENDERER_PRIMITIVE_HEADER_