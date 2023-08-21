#ifndef _SRENDERER_INDIRECT_HEADER_
#define _SRENDERER_INDIRECT_HEADER_

struct DrawIndexedIndirectEX {
    uint indexCount;
    uint instanceCount;
    uint firstIndex;
    int vertexOffset;
    uint firstInstance;
    uint geometryID;
    uint padding0;
    uint padding1;
};

#endif // !_SRENDERER_INDIRECT_HEADER_