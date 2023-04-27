#ifndef _SRENDERER_COMMON_INDIRECT_HEADER_
#define _SRENDERER_COMMON_INDIRECT_HEADER_

struct DrawIndexedIndirectEX {
    uint    indexCount;
    uint    instanceCount;
    uint    firstIndex;
    int     vertexOffset;
    uint    firstInstance;
    uint    geometryID;
};

#endif