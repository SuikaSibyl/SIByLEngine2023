#ifndef _SRENDERER_COMMON_VERT_HEADER_
#define _SRENDERER_COMMON_VERT_HEADER_

#include "../../include/common_descriptor_sets.h"

InterleavedVertex fetchVertex() {
    return vertices[gl_VertexIndex];
}

#endif