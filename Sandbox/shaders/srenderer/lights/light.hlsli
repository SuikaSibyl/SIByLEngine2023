#ifndef _SRENDERER_LIGHTS_HEADER_
#define _SRENDERER_LIGHTS_HEADER_

interface ILightParameter : IDifferentiable {};

namespace ilight {
struct eval_le_in {
    float3 p;
    float3 dir;
};

struct sample_li_in {
    float3 p;
    float2 uv;
};

struct sample_li_out {
    float3 L;
    float pdf;
    float3 wi;
    float3 x;
};
}

interface ILight {
    // Associated a parameter type for each microfacet distribution
    associatedtype TParam : ILightParameter;

};

#endif // _SRENDERER_LIGHTS_HEADER_