#ifndef _SRENDERER_CBT_HLSLI_   
#define _SRENDERER_CBT_HLSLI_
#define CBT_FLAG_WRITE

RWStructuredBuffer<uint> cbt_heap;

// data structures
struct cbt_Node {
    uint id;   // heapID
    int depth; // findMSB(heapID) := node depth
};

// manipulation
#ifdef CBT_FLAG_WRITE
void cbt_SplitNode_Fast(in const cbt_Node node);
void cbt_SplitNode(in const cbt_Node node);
void cbt_MergeNode_Fast(in const cbt_Node node);
void cbt_MergeNode(in const cbt_Node node);
#endif

// O(1) queries
int cbt_MaxDepth();
uint cbt_NodeCount();
uint cbt_HeapRead(in const cbt_Node node);
bool cbt_IsLeafNode(in const cbt_Node node);
bool cbt_IsCeilNode(in const cbt_Node node);
bool cbt_IsRootNode(in const cbt_Node node);
bool cbt_IsNullNode(in const cbt_Node node);

// O(depth) queries
uint cbt_EncodeNode(in const cbt_Node node);
cbt_Node cbt_DecodeNode(uint nodeID);

// node constructors
cbt_Node cbt_CreateNode(uint id);
cbt_Node cbt_CreateNode(uint id, int depth);
cbt_Node cbt_ParentNode_Fast(in const cbt_Node node);
cbt_Node cbt_ParentNode(in const cbt_Node node);
cbt_Node cbt_SiblingNode_Fast(in const cbt_Node node);
cbt_Node cbt_SiblingNode(in const cbt_Node node);
cbt_Node cbt_LeftSiblingNode_Fast(in const cbt_Node node);
cbt_Node cbt_LeftSiblingNode(in const cbt_Node node);
cbt_Node cbt_RightSiblingNode_Fast(in const cbt_Node node);
cbt_Node cbt_RightSiblingNode(in const cbt_Node node);
cbt_Node cbt_LeftChildNode_Fast(in const cbt_Node node);
cbt_Node cbt_LeftChildNode(in const cbt_Node node);
cbt_Node cbt_RightChildNode_Fast(in const cbt_Node node);
cbt_Node cbt_RightChildNode(in const cbt_Node node);

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

/*******************************************************************************
 * GetBitValue -- Returns the value of a bit stored in a 32-bit word */
uint cbt__GetBitValue(uint bitField, uint bitID) {
    return ((bitField >> bitID) & 1u); }
/*******************************************************************************
 * SetBitValue -- Sets the value of a bit stored in a 32-bit word */
#ifdef CBT_FLAG_WRITE
void cbt__SetBitValue(uint bufferID, uint bitID, uint bitValue) {
    const uint bitMask = ~(1u << bitID);
    InterlockedAnd(cbt_heap[bufferID], bitMask);
    InterlockedOr(cbt_heap[bufferID], bitValue << bitID); }
/*******************************************************************************
 * BitFieldInsert -- Returns the bit field after insertion of some bit data in range
 * [bitOffset, bitOffset + bitCount - 1] */
void cbt__BitFieldInsert(uint bufferID, uint bitOffset, uint bitCount, uint bitData) {
    uint bitMask = ~(~(0xFFFFFFFFu << bitCount) << bitOffset);
    InterlockedAnd(cbt_heap[bufferID], bitMask);
    InterlockedOr(cbt_heap[bufferID], bitData << bitOffset); }
#endif // CBT_FLAG_WRITE
/*******************************************************************************
 * BitFieldExtract -- Extracts bits [bitOffset, bitOffset + bitCount - 1] from
 * a bit field, returning them in the least significant bits of the result. */
uint cbt__BitFieldExtract(uint bitField, uint bitOffset, uint bitCount) {
    uint bitMask = ~(0xFFFFFFFFu << bitCount);
    return (bitField >> bitOffset) & bitMask; }
/*******************************************************************************
 * IsCeilNode -- Checks if a node is a ceil node, i.e., that can not split further */
bool cbt_IsCeilNode(in const cbt_Node node) {
    return (node.depth == cbt_MaxDepth()); }
/*******************************************************************************
 * IsRootNode -- Checks if a node is a root node */
bool cbt_IsRootNode(in const cbt_Node node) {
    return (node.id == 1u); }
/*******************************************************************************
 * IsNullNode -- Checks if a node is a null node */
bool cbt_IsNullNode(in const cbt_Node node) {
    return (node.id == 0u); }
/*******************************************************************************
 * CreateNode -- Constructor for the Node data structure */
cbt_Node cbt_CreateNode(uint id) {
    return cbt_CreateNode(id, firstbithigh(id)); }
cbt_Node cbt_CreateNode(uint id, int depth) {
    cbt_Node node;
    node.id = id;
    node.depth = depth;
    return node; }
/*******************************************************************************
 * ParentNode -- Computes the parent of the input node */
cbt_Node cbt_ParentNode_Fast(in const cbt_Node node) {
    return cbt_CreateNode(node.id >> 1, node.depth - 1); }
cbt_Node cbt_ParentNode(in const cbt_Node node) {
    if (cbt_IsNullNode(node)) return node;
    else return cbt_ParentNode_Fast(node); }
/*******************************************************************************
 * CeilNode -- Returns the associated ceil node, i.e., the deepest possible leaf */
cbt_Node cbt__CeilNode_Fast(in const cbt_Node node) {
    int maxDepth = cbt_MaxDepth();
    return cbt_CreateNode(node.id << (maxDepth - node.depth), maxDepth); }
cbt_Node cbt__CeilNode(in const cbt_Node node) {
    if (cbt_IsNullNode(node)) return node;
    else return cbt__CeilNode_Fast(node); }
/*******************************************************************************
 * SiblingNode -- Computes the sibling of the input node */
cbt_Node cbt_SiblingNode_Fast(in const cbt_Node node) {
    return cbt_CreateNode(node.id ^ 1u, node.depth); }
cbt_Node cbt_SiblingNode(in const cbt_Node node) {
    if (cbt_IsNullNode(node)) return node;
    else return cbt_SiblingNode_Fast(node); }
/*******************************************************************************
 * RightSiblingNode -- Computes the right sibling of the input node */
cbt_Node cbt_RightSiblingNode_Fast(in const cbt_Node node) {
    return cbt_CreateNode(node.id | 1u, node.depth); }
cbt_Node cbt_RightSiblingNode(in const cbt_Node node) {
    if (cbt_IsNullNode(node)) return node;
    else return cbt_RightSiblingNode_Fast(node); }
/*******************************************************************************
 * LeftSiblingNode -- Computes the left sibling of the input node */
cbt_Node cbt_LeftSiblingNode_Fast(in const cbt_Node node) {
    return cbt_CreateNode(node.id & (~1u), node.depth); }
cbt_Node cbt_LeftSiblingNode(in const cbt_Node node) {
    if (cbt_IsNullNode(node)) return node;
    else return cbt_LeftSiblingNode_Fast(node); }
/*******************************************************************************
 * RightChildNode -- Computes the right child of the input node */
cbt_Node cbt_RightChildNode_Fast(in const cbt_Node node) {
    return cbt_CreateNode((node.id << 1) | 1u, node.depth + 1); }
cbt_Node cbt_RightChildNode(in const cbt_Node node) {
    if (cbt_IsNullNode(node)) return node;
    else return cbt_RightChildNode_Fast(node); }
/*******************************************************************************
 * LeftChildNode -- Computes the left child of the input node */
cbt_Node cbt_LeftChildNode_Fast(in const cbt_Node node) {
    return cbt_CreateNode(node.id << 1, node.depth + 1); }
cbt_Node cbt_LeftChildNode(in const cbt_Node node) {
    if (cbt_IsNullNode(node)) return node;
    else return cbt_LeftChildNode_Fast(node); }
/*******************************************************************************
 * HeapByteSize -- Computes the number of Bytes to allocate for the bitfield
 *
 * For a tree of max depth D, the number of Bytes is 2^(D-1).
 * Note that 2 bits are "wasted" in the sense that they only serve
 * to round the required number of bytes to a power of two. */
uint cbt__HeapByteSize(uint maxDepth) {
    return 1u << (maxDepth - 1); }
/*******************************************************************************
 * HeapUint32Size -- Computes the number of uints to allocate for the bitfield */
uint cbt__HeapUint32Size(uint maxDepth) {
    return cbt__HeapByteSize(maxDepth) >> 2; }
/*******************************************************************************
 * NodeBitID -- Returns the bit index that stores data associated with a given node
 *
 * For a LEB of max depth D and given an index in [0, 2^(D+1) - 1], this
 * functions is used to emulate the behaviour of a lookup in an array, i.e.,
 * uint32_t[nodeID]. It provides the first bit in memory that stores
 * information associated with the element of index nodeID.
 *
 * For data located at level d, the bit offset is 2^d x (3 - d + D)
 * We then offset this quantity by the index by (nodeID - 2^d) x (D + 1 - d)
 * Note that the null index (nodeID = 0) is also supported. */
uint cbt__NodeBitID(in const cbt_Node node) {
    uint tmp1 = 2u << node.depth;
    uint tmp2 = uint(1 + cbt_MaxDepth() - node.depth);
    return tmp1 + node.id * tmp2; }
/*******************************************************************************
 * NodeBitID_BitField -- Computes the bitfield bit location associated to a node
 *
 * Here, the node is converted into a final node and its bit offset is
 * returned, which is finalNodeID + 2^{D + 1} */
uint cbt__NodeBitID_BitField(in const cbt_Node node) {
    return cbt__NodeBitID(cbt__CeilNode(node)); }
/*******************************************************************************
 * DataBitSize -- Returns the number of bits associated with a given node */
int cbt__NodeBitSize(in const cbt_Node node) {
    return cbt_MaxDepth() - node.depth + 1; }
/*******************************************************************************
 * HeapArgs
 *
 * The LEB heap data structure uses an array of 32-bit words to store its data.
 * Whenever we need to access a certain bit range, we need to query two such
 * words (because sometimes the requested bit range overlaps two 32-bit words).
 * The HeapArg data structure provides arguments for reading from and/or
 * writing to the two 32-bit words that bound the queries range. */
struct cbt__HeapArgs {
    uint heapIndexLSB, heapIndexMSB;
    uint bitOffsetLSB;
    uint bitCountLSB, bitCountMSB;
};

cbt__HeapArgs cbt__CreateHeapArgs(in const cbt_Node node, int bitCount) {
    uint alignedBitOffset = cbt__NodeBitID(node);
    uint maxHeapIndex = cbt__HeapUint32Size(cbt_MaxDepth()) - 1u;
    uint heapIndexLSB = (alignedBitOffset >> 5u);
    uint heapIndexMSB = min(heapIndexLSB + 1, maxHeapIndex);
    cbt__HeapArgs args;

    args.bitOffsetLSB = alignedBitOffset & 31u;
    args.bitCountLSB = min(32u - args.bitOffsetLSB, bitCount);
    args.bitCountMSB = bitCount - args.bitCountLSB;
    args.heapIndexLSB = heapIndexLSB;
    args.heapIndexMSB = heapIndexMSB;

    return args;
}

/*******************************************************************************
 * HeapWrite -- Sets bitCount bits located at nodeID to bitData
 *
 * Note that this procedure writes to at most two uint32 elements.
 * Two elements are relevant whenever the specified interval overflows 32-bit
 * words. */
#ifdef CBT_FLAG_WRITE
void cbt__HeapWriteExplicit(in const cbt_Node node, int bitCount, uint bitData) {
    cbt__HeapArgs args = cbt__CreateHeapArgs(node, bitCount);
    cbt__BitFieldInsert(args.heapIndexLSB,
                        args.bitOffsetLSB,
                        args.bitCountLSB,
                        bitData);
    cbt__BitFieldInsert(args.heapIndexMSB,
                        0u,
                        args.bitCountMSB,
                        bitData >> args.bitCountLSB);
}

void cbt__HeapWrite(in const cbt_Node node, uint bitData) {
    cbt__HeapWriteExplicit(node, cbt__NodeBitSize(node), bitData);
}
#endif // CBT_FLAG_WRITE

/*******************************************************************************
 * HeapRead -- Returns bitCount bits located at nodeID
 *
 * Note that this procedure writes to at most two uint32 elements.
 * Two elements are relevant whenever the specified interval overflows 32-bit
 * words.
 *
 */
uint cbt__HeapReadExplicit(in const cbt_Node node, int bitCount) {
    cbt__HeapArgs args = cbt__CreateHeapArgs(node, bitCount);
    uint lsb = cbt__BitFieldExtract(cbt_heap[args.heapIndexLSB],
                                    args.bitOffsetLSB,
                                    args.bitCountLSB);
    uint msb = cbt__BitFieldExtract(cbt_heap[args.heapIndexMSB],
                                    0u,
                                    args.bitCountMSB);

    return (lsb | (msb << args.bitCountLSB));
}

uint cbt_HeapRead(in const cbt_Node node) {
    return cbt__HeapReadExplicit(node, cbt__NodeBitSize(node));
}

/*******************************************************************************
 * HeapWrite_BitField -- Sets the bit associated to a leaf node to bitValue
 *
 * This is a dedicated routine to write directly to the bitfield.
 *
 */
#ifdef CBT_FLAG_WRITE
void cbt__HeapWrite_BitField(in const cbt_Node node, uint bitValue) {
    uint bitID = cbt__NodeBitID_BitField(node);
    cbt__SetBitValue(bitID >> 5u, bitID & 31u, bitValue);
}
#endif // CBT_FLAG_WRITE

/*******************************************************************************
 * HeapRead_BitField -- Returns the value of the bit associated to a leaf node
 *
 * This is a dedicated routine to read directly from the bitfield. */
uint cbt__HeapRead_BitField(in const cbt_Node node) {
    uint bitID = cbt__NodeBitID_BitField(node);
    return cbt__GetBitValue(cbt_heap[bitID >> 5u], bitID & 31u);
}

/*******************************************************************************
 * IsLeafNode -- Checks if a node is a leaf node */
bool cbt_IsLeafNode(in const cbt_Node node) {
    return (cbt_HeapRead(node) == 1u);
}

/*******************************************************************************
 * Split -- Subdivides a node in two */
#ifdef CBT_FLAG_WRITE
void cbt_SplitNode_Fast(in const cbt_Node node) {
    cbt__HeapWrite_BitField(cbt_RightChildNode(node), 1u);
}
void cbt_SplitNode(in const cbt_Node node) {
    if (!cbt_IsCeilNode(node))
        cbt_SplitNode_Fast(node);
}

/*******************************************************************************
 * Merge -- Merges the node with its neighbour */
void cbt_MergeNode_Fast(in const cbt_Node node) {
    cbt__HeapWrite_BitField(cbt_RightSiblingNode(node), 0u);
}
void cbt_MergeNode(in const cbt_Node node) {
    if (!cbt_IsRootNode(node))
        cbt_MergeNode_Fast(node);
}
#endif // CBT_FLAG_WRITE

/*******************************************************************************
 * MaxDepth -- Returns the maximum depth */
int cbt_MaxDepth() {
    return firstbitlow(cbt_heap[0]);
}

/*******************************************************************************
 * NodeCount -- Returns the number of triangles in the LEB */
uint cbt_NodeCount() {
    return cbt_HeapRead(cbt_CreateNode(1u, 0));
}

/*******************************************************************************
 * Decode the LEB Node associated to an index */
cbt_Node cbt_DecodeNode(uint nodeID) {
    cbt_Node node = cbt_CreateNode(1u, 0);

    while (cbt_HeapRead(node) > 1u) {
        cbt_Node leftChild = cbt_LeftChildNode_Fast(node);
        uint cmp = cbt_HeapRead(leftChild);
        uint b = nodeID < cmp ? 0u : 1u;

        node = leftChild;
        node.id |= b;
        nodeID -= cmp * b;
    }

    return node;
}

/*******************************************************************************
 * EncodeNode -- Returns the bit index associated with the Node
 *
 * This does the inverse of the DecodeNode routine. */
uint cbt_EncodeNode(in const cbt_Node node) {
    uint nodeID = 0u;
    cbt_Node nodeIterator = node;

    while (nodeIterator.id > 1u) {
        cbt_Node sibling = cbt_LeftSiblingNode_Fast(nodeIterator);
        uint nodeCount = cbt_HeapRead(sibling);

        nodeID += (nodeIterator.id & 1u) * nodeCount;
        nodeIterator = cbt_ParentNode(nodeIterator);
    }

    return nodeID;
}

#endif // _SRENDERER_CBT_HLSLI_