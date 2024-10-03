#ifndef _SRENDERER_PRIMITIVE_HEADER_
#define _SRENDERER_PRIMITIVE_HEADER_

namespace iprimitive {
struct sample_in {
    float3 uv;
};

struct sample_out {
    float3 position;
    float3 normal;
    float pdf;
};
}

interface IPrimitive {

}


#endif // _SRENDERER_PRIMITIVE_HEADER_