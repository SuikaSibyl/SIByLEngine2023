#ifndef _SRENDERER_COMMON_RT_RGEN_LAYOUT_HEADER_
#define _SRENDERER_COMMON_RT_RGEN_LAYOUT_HEADER_

layout(location = 0) rayPayloadEXT PrimaryPayload   rPrimaryPld;
layout(location = 1) rayPayloadEXT ShadowPayload    rShadowPld;


layout(location = CALLABLE_LOC_SHAPE_SAMPLE)        callableDataEXT ShapeSampleQuery    cShapeSampleQuery;
layout(location = CALLABLE_LOC_SHAPE_SAMPLE_PDF)    callableDataEXT ShapeSamplePdfQuery cShapeSamplePdfQuery;
layout(location = CALLABLE_LOC_BSDF_EVAL)           callableDataEXT BSDFEvalQuery       cBSDFEvalQuery;
layout(location = CALLABLE_LOC_BSDF_SAMPLE)         callableDataEXT BSDFSampleQuery     cBSDFSampleQuery;
layout(location = CALLABLE_LOC_BSDF_PDF)            callableDataEXT BSDFSamplePDFQuery  cBSDFSamplePDFQuery;

#endif