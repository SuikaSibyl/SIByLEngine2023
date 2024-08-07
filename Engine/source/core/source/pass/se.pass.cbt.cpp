#define DLIB_EXPORT
#include <passes/se.pass.cbt.hpp>
#undef DLIB_EXPORT
#include <imgui.h>

namespace se::cbt {
namespace host {
struct cbt_Tree {
  cbt_Tree(size_t size, size_t max_depth) {
    buffer.resize(size);
    heap = std::span<uint64_t>((uint64_t*)buffer.data(), 
      buffer.size() / sizeof(uint64_t));
    heap[0] = 1ULL << (max_depth); // store max Depth
  }
  std::vector<std::byte> buffer;
  std::span<uint64_t> heap;
};

struct cbt_Node {
  uint64_t id : 58; // heapID
  uint64_t depth : 6; // log2(heapID)
};

/*******************************************************************************
 * FindLSB -- Returns the position of the least significant bit */
inline int64_t cbt__FindLSB(uint64_t x) {
  int64_t lsb = 0; while (((x >> lsb) & 1u) == 0u) { ++lsb; } return lsb; }
/*******************************************************************************
 * MaxDepth -- Returns the max CBT depth */
inline int cbt_MaxDepth(cbt_Tree const& tree) { return cbt__FindLSB(tree.heap[0]); }
/*******************************************************************************
 * HeapByteSize -- Computes the number of Bytes to allocate for the bitfield
 *
 * For a tree of max depth D, the number of Bytes is 2^(D-1).
 * Note that 2 bits are "wasted" in the sense that they only serve
 * to round the required number of bytes to a power of two.  */
static int64_t cbt__HeapByteSize(uint64_t treeMaxDepth) {
  return 1LL << (treeMaxDepth - 1); }
/*******************************************************************************
 * HeapUint64Size -- Computes the number of uints to allocate for the bitfield */
inline int64_t cbt__HeapUint64Size(int64_t treeMaxDepth) {
  return cbt__HeapByteSize(treeMaxDepth) >> 3; }
/*******************************************************************************
 * CreateNode -- Constructor for the Node data structure */
cbt_Node cbt_CreateNode(uint64_t id, int64_t depth) {
  cbt_Node node; node.id = id; node.depth = depth; return node; }
/*******************************************************************************
 * IsCeilNode -- Checks if a node is a ceil node, i.e., that can not split further */
bool cbt_IsCeilNode(cbt_Tree const& tree, const cbt_Node node) {
  return (node.depth == cbt_MaxDepth(tree)); }
/*******************************************************************************
 * IsRootNode -- Checks if a node is a root node */
bool cbt_IsRootNode(cbt_Node const node) {
  return (node.id == 1u); }
/*******************************************************************************
 * IsNullNode -- Checks if a node is a null node */
bool cbt_IsNullNode(const cbt_Node node) {
  return (node.id == 0u); }
/*******************************************************************************
 * CeilNode -- Returns the associated ceil node, i.e., the deepest possible leaf */
cbt_Node cbt__CeilNode_Fast(cbt_Tree const& tree, const cbt_Node node) {
  int64_t maxDepth = cbt_MaxDepth(tree);  
  return cbt_CreateNode(node.id << (maxDepth - node.depth), maxDepth); }
cbt_Node cbt__CeilNode(cbt_Tree const& tree, const cbt_Node node) {
  return cbt_IsNullNode(node) ? node : cbt__CeilNode_Fast(tree, node); }
/*******************************************************************************
 * NodeBitID -- Returns the bit index that stores data associated with a given node
 *
 * For a tree of max depth D and given an index in [0, 2^(D+1) - 1], this
 * functions is used to emulate the behaviour of a lookup in an array, i.e.,
 * uint[nodeID]. It provides the first bit in memory that stores
 * information associated with the element of index nodeID.
 *
 * For data located at level d, the bit offset is 2^d x (3 - d + D)
 * We then offset this quantity by the index by (nodeID - 2^d) x (D + 1 - d)
 * Note that the null index (nodeID = 0) is also supported. */
inline int64_t cbt__NodeBitID(cbt_Tree const& tree, const cbt_Node node) {
  int64_t tmp1 = 2LL << node.depth;
  int64_t tmp2 = 1LL + cbt_MaxDepth(tree) - node.depth;
  return tmp1 + node.id * tmp2; }
/*******************************************************************************
 * NodeBitID_BitField -- Computes the bitfield bit location associated to a node
 *
 * Here, the node is converted into a final node and its bit offset is
 * returned, which is finalNodeID + 2^{D + 1} */
int64_t cbt__NodeBitID_BitField(cbt_Tree const& tree, const cbt_Node node) {
    return cbt__NodeBitID(tree, cbt__CeilNode(tree, node)); }
/*******************************************************************************
 * SetBitValue -- Sets the value of a bit stored in a bitfield */
void cbt__SetBitValue(uint64_t* bitField, int64_t bitID, uint64_t bitValue) {
  const uint64_t bitMask = ~(1ULL << bitID);
  __pragma("omp atomic")
  (*bitField) &= bitMask;
  __pragma("omp atomic")
  (*bitField) |= (bitValue << bitID);
}
/*******************************************************************************
 * HeapWrite_BitField -- Sets the bit associated to a leaf node to bitValue 
 * This is a dedicated routine to write directly to the bitfield. */
void cbt__HeapWrite_BitField(cbt_Tree& tree, const cbt_Node node, const uint64_t bitValue) {
  int64_t bitID = cbt__NodeBitID_BitField(tree, node);
  cbt__SetBitValue(&tree.heap[bitID >> 6], bitID & 63, bitValue); }
/*******************************************************************************
 * ClearBitField -- Clears the bitfield */
void cbt__ClearBitfield(cbt_Tree& tree) {
  int64_t maxDepth = cbt_MaxDepth(tree);
  int64_t bufferMinID = 1LL << (maxDepth - 5);
  int64_t bufferMaxID = cbt__HeapUint64Size(maxDepth);
  for (int bufferID = bufferMinID; bufferID < bufferMaxID; ++bufferID) {
    tree.heap[bufferID] = 0;
  }}
/*******************************************************************************
 * MinValue -- Returns the minimum value between two inputs */
inline uint64_t cbt__MinValue(uint64_t a, uint64_t b) {
  return a < b ? a : b; }
/*******************************************************************************
 * HeapArgs
 *
 * The CBT heap data structure uses an array of 64-bit words to store its data.
 * Whenever we need to access a certain bit range, we need to query two such
 * words (because sometimes the requested bit range overlaps two 64-bit words).
 * The HeapArg data structure provides arguments for reading from and/or
 * writing to the two 64-bit words that bound the queries range. */
struct cbt__HeapArgs {
  uint64_t* bitFieldLSB, * bitFieldMSB;
  int64_t bitOffsetLSB;
  int64_t bitCountLSB, bitCountMSB;
};

cbt__HeapArgs cbt__CreateHeapArgs(cbt_Tree const& tree, const cbt_Node node, int64_t bitCount) {
  int64_t alignedBitOffset = cbt__NodeBitID(tree, node);
  int64_t maxBufferIndex = cbt__HeapUint64Size(cbt_MaxDepth(tree)) - 1;
  int64_t bufferIndexLSB = (alignedBitOffset >> 6);
  int64_t bufferIndexMSB = cbt__MinValue(bufferIndexLSB + 1, maxBufferIndex);
  cbt__HeapArgs args;
  args.bitOffsetLSB = alignedBitOffset & 63;
  args.bitCountLSB = cbt__MinValue(64 - args.bitOffsetLSB, bitCount);
  args.bitCountMSB = bitCount - args.bitCountLSB;
  args.bitFieldLSB = &tree.heap[bufferIndexLSB];
  args.bitFieldMSB = &tree.heap[bufferIndexMSB];
  return args;
}
/*******************************************************************************
 * NodeBitSize -- Returns the number of bits storing the input node value */
static inline int64_t
cbt__NodeBitSize(cbt_Tree const& tree, const cbt_Node node) {
  return cbt_MaxDepth(tree) - node.depth + 1; }
/*******************************************************************************
 * BitFieldExtract -- Extracts bits [bitOffset, bitOffset + bitCount - 1] from
 * a bitfield, returning them in the least significant bits of the result. */
static inline uint64_t
cbt__BitFieldExtract(const uint64_t bitField, int64_t bitOffset, int64_t bitCount) {
  uint64_t bitMask = ~(0xFFFFFFFFFFFFFFFFULL << bitCount);
  return (bitField >> bitOffset) & bitMask; }
/*******************************************************************************
 * HeapRead -- Returns bitCount bits located at nodeID
 *
 * Note that this procedure reads from two uint64 elements.
 * This is because the data is not necessarily aligned with 64-bit
 * words. */
uint64_t cbt__HeapReadExplicit(cbt_Tree const& tree, const cbt_Node node, int64_t bitCount) {
  cbt__HeapArgs args = cbt__CreateHeapArgs(tree, node, bitCount);
  uint64_t lsb = cbt__BitFieldExtract(*args.bitFieldLSB, args.bitOffsetLSB, args.bitCountLSB);
  uint64_t msb = cbt__BitFieldExtract(*args.bitFieldMSB, 0u, args.bitCountMSB);
  return (lsb | (msb << args.bitCountLSB)); }
uint64_t cbt_HeapRead(cbt_Tree const& tree, const cbt_Node node) {
  return cbt__HeapReadExplicit(tree, node, cbt__NodeBitSize(tree, node)); }
/*******************************************************************************
 * ResetToDepth -- Initializes a CBT to its a specific subdivision level */
void cbt_ResetToDepth(cbt_Tree& tree, int64_t depth) {
  uint64_t minNodeID = 1ULL << depth;
  uint64_t maxNodeID = 2ULL << depth;
  cbt__ClearBitfield(tree);
  for (uint64_t nodeID = minNodeID; nodeID < maxNodeID; ++nodeID) {
    cbt_Node node = cbt_CreateNode(nodeID, depth);
    cbt__HeapWrite_BitField(tree, node, 1u);
  }
}
/*******************************************************************************
 * BitfieldInsert -- Inserts data in range [offset, offset + count - 1] */
static inline void
cbt__BitFieldInsert(
    uint64_t* bitField,
    int64_t  bitOffset,
    int64_t  bitCount,
    uint64_t bitData
) {
  uint64_t bitMask = ~(~(0xFFFFFFFFFFFFFFFFULL << bitCount) << bitOffset);
  __pragma("omp atomic")
  (*bitField) &= bitMask;
  __pragma("omp atomic")
  (*bitField) |= (bitData << bitOffset);
}
/*******************************************************************************
 * HeapWrite -- Sets bitCount bits located at nodeID to bitData
 *
 * Note that this procedure writes to at most two uint64 elements.
 * Two elements are relevant whenever the specified interval overflows 64-bit
 * words. */
static void
cbt__HeapWriteExplicit(
    cbt_Tree& tree,
    const cbt_Node node,
    int64_t bitCount,
    uint64_t bitData
) {
    cbt__HeapArgs args = cbt__CreateHeapArgs(tree, node, bitCount);

    cbt__BitFieldInsert(args.bitFieldLSB,
        args.bitOffsetLSB,
        args.bitCountLSB,
        bitData);
    cbt__BitFieldInsert(args.bitFieldMSB,
        0u,
        args.bitCountMSB,
        bitData >> args.bitCountLSB);
}

static void cbt__HeapWrite(cbt_Tree& tree, const cbt_Node node, uint64_t bitData) {
    cbt__HeapWriteExplicit(tree, node, cbt__NodeBitSize(tree, node), bitData); }
/*******************************************************************************
 * ComputeSumReduction -- Sums the 2 elements below the current slot */
static void cbt__ComputeSumReduction(cbt_Tree& tree) {
  int64_t depth = cbt_MaxDepth(tree);
  uint64_t minNodeID = (1ULL << depth);
  uint64_t maxNodeID = (2ULL << depth);
 
  // prepass: processes deepest levels in parallel
  __pragma("omp parallel for")
  for (uint64_t nodeID = minNodeID; nodeID < maxNodeID; nodeID+= 64u) {
    cbt_Node heapNode = cbt_CreateNode(nodeID, depth);
    int64_t alignedBitOffset = cbt__NodeBitID(tree, heapNode);
    uint64_t bitField = tree.heap[alignedBitOffset >> 6];
    uint64_t bitData = 0u;
    
    // 2-bits
    bitField = (bitField & 0x5555555555555555ULL)
             + ((bitField >>  1) & 0x5555555555555555ULL);
    bitData = bitField;
    tree.heap[(alignedBitOffset - minNodeID) >> 6] = bitData;
    
    // 3-bits
    bitField = (bitField & 0x3333333333333333ULL)
             + ((bitField >>  2) & 0x3333333333333333ULL);
    bitData = ((bitField >>  0) & (7ULL <<  0))
            | ((bitField >>  1) & (7ULL <<  3))
            | ((bitField >>  2) & (7ULL <<  6))
            | ((bitField >>  3) & (7ULL <<  9))
            | ((bitField >>  4) & (7ULL << 12))
            | ((bitField >>  5) & (7ULL << 15))
            | ((bitField >>  6) & (7ULL << 18))
            | ((bitField >>  7) & (7ULL << 21))
            | ((bitField >>  8) & (7ULL << 24))
            | ((bitField >>  9) & (7ULL << 27))
            | ((bitField >> 10) & (7ULL << 30))
            | ((bitField >> 11) & (7ULL << 33))
            | ((bitField >> 12) & (7ULL << 36))
            | ((bitField >> 13) & (7ULL << 39))
            | ((bitField >> 14) & (7ULL << 42))
            | ((bitField >> 15) & (7ULL << 45));
    cbt__HeapWriteExplicit(tree, cbt_CreateNode(nodeID >> 2, depth - 2), 48ULL, bitData);
    
    // 4-bits
    bitField = (bitField & 0x0F0F0F0F0F0F0F0FULL)
             + ((bitField >>  4) & 0x0F0F0F0F0F0F0F0FULL);
    bitData = ((bitField >>  0) & (15ULL <<  0))
            | ((bitField >>  4) & (15ULL <<  4))
            | ((bitField >>  8) & (15ULL <<  8))
            | ((bitField >> 12) & (15ULL << 12))
            | ((bitField >> 16) & (15ULL << 16))
            | ((bitField >> 20) & (15ULL << 20))
            | ((bitField >> 24) & (15ULL << 24))
            | ((bitField >> 28) & (15ULL << 28));
    cbt__HeapWriteExplicit(tree, cbt_CreateNode(nodeID >> 3, depth - 3), 32ULL, bitData);
    
    // 5-bits
    bitField = (bitField & 0x00FF00FF00FF00FFULL)
             + ((bitField >>  8) & 0x00FF00FF00FF00FFULL);
    bitData = ((bitField >>  0) & (31ULL <<  0))
            | ((bitField >> 11) & (31ULL <<  5))
            | ((bitField >> 22) & (31ULL << 10))
            | ((bitField >> 33) & (31ULL << 15));
    cbt__HeapWriteExplicit(tree, cbt_CreateNode(nodeID >> 4, depth - 4), 20ULL, bitData);
    
    // 6-bits
    bitField = (bitField & 0x0000FFFF0000FFFFULL)
             + ((bitField >> 16) & 0x0000FFFF0000FFFFULL);
    bitData = ((bitField >>  0) & (63ULL << 0))
            | ((bitField >> 26) & (63ULL << 6));
    cbt__HeapWriteExplicit(tree, cbt_CreateNode(nodeID >> 5, depth - 5), 12ULL, bitData);
    
    // 7-bits
    bitField = (bitField & 0x00000000FFFFFFFFULL)
             + ((bitField >> 32) & 0x00000000FFFFFFFFULL);
    bitData = bitField;
    cbt__HeapWriteExplicit(tree, cbt_CreateNode(nodeID >> 6, depth - 6),  7ULL, bitData);
  }
  __pragma("omp barrier")
  depth-= 6;

  // iterate over elements atomically
  while (--depth >= 0) {
     uint64_t minNodeID = 1ULL << depth;
     uint64_t maxNodeID = 2ULL << depth;
    
     __pragma("omp parallel for")
     for (uint64_t j = minNodeID; j < maxNodeID; ++j) {
       uint64_t x0 = cbt_HeapRead(tree, cbt_CreateNode(j << 1    , depth + 1));
       uint64_t x1 = cbt_HeapRead(tree, cbt_CreateNode(j << 1 | 1, depth + 1));
       cbt__HeapWrite(tree, cbt_CreateNode(j, depth), x0 + x1);
     }
    __pragma("omp barrier")
  }
}
}

CreateCBTPass::CreateCBTPass(int maxDepth, int initDepth)
  : maxDepth(maxDepth), initDepth(initDepth) { pReflection = reflect(); }

auto CreateCBTPass::reflect() noexcept -> rdg::PassReflection {
  size_t heap_byte_size = host::cbt__HeapByteSize(maxDepth);
  rdg::PassReflection reflector;
  reflector.addOutput("CBTree")
    .isBuffer().withSize(heap_byte_size)
    .withUsages((uint32_t)rhi::BufferUsageBit::COPY_DST)
    .consume(rdg::BufferInfo::ConsumeEntry{}
    .setAccess((uint32_t)rhi::AccessFlagBits::TRANSFER_WRITE_BIT)
    .addStage((uint32_t)rhi::PipelineStageBit::TRANSFER_BIT));
return reflector;
}

auto CreateCBTPass::execute(
  rdg::RenderContext* context,
  rdg::RenderData const& renderData) noexcept -> void {
  gfx::BufferHandle cbt = renderData.getBuffer("CBTree");
  if (!initialized) {
    initialized = true;
    // create the cbt tree
    host::cbt_Tree tree(host::cbt__HeapByteSize(maxDepth), maxDepth);
    // ResetToDepth -- Initializes a CBT to its a specific subdivision level
    uint64_t minNodeID = 1ULL << initDepth;
    uint64_t maxNodeID = 2ULL << initDepth;
    cbt__ClearBitfield(tree);
    for (uint64_t nodeID = minNodeID; nodeID < maxNodeID; ++nodeID) {
      host::cbt_Node node = host::cbt_CreateNode(nodeID, initDepth);
      cbt__HeapWrite_BitField(tree, node, 1u);
    }
    cbt__ComputeSumReduction(tree);
    auto* device = gfx::GFXContext::device;
    std::unique_ptr<rhi::Buffer> stageBuffer = device->createDeviceLocalBuffer(
      tree.buffer.data(), tree.buffer.size(), (uint32_t)rhi::BufferUsageBit::COPY_SRC);
    device->copyBufferToBuffer(stageBuffer.get(), 0, cbt->buffer.get(), 0, cbt->buffer->size());
    device->waitIdle();
  }
}

SumReductionFusedPass::SumReductionFusedPass(int maxDepth) :maxDepth(maxDepth) {
  auto [comp] = gfx::GFXContext::load_shader_slang(
    "cbt/sum-reduction-fuse.slang",
    std::array<std::pair<std::string, rhi::ShaderStageBit>, 1>{
      std::make_pair("ComputeMain", rhi::ShaderStageBit::COMPUTE),
  }, {}, true);
  rdg::ComputePass::init(comp.get());
}

auto SumReductionFusedPass::reflect() noexcept -> rdg::PassReflection {
  rdg::PassReflection reflector;
  reflector.addInputOutput("CBTree")
    .isBuffer().withUsages((uint32_t)rhi::BufferUsageBit::COPY_DST)
    .consume(rdg::BufferInfo::ConsumeEntry{}
    .setAccess((uint32_t)rhi::AccessFlagBits::SHADER_WRITE_BIT
             | (uint32_t)rhi::AccessFlagBits::SHADER_READ_BIT)
    .addStage((uint32_t)rhi::PipelineStageBit::COMPUTE_SHADER_BIT));
  return reflector;
}
  
auto SumReductionFusedPass::execute(rdg::RenderContext* context, rdg::RenderData const& renderData) noexcept -> void {
  gfx::BufferHandle cbt = renderData.getBuffer("CBTree");

  updateBindings(context, {
    {"cbt_heap", rhi::BindingResource{{cbt->buffer.get(), 0, cbt->buffer->size()}}},
  });

  int it = maxDepth;
  int cnt = ((1 << it) >> 5);
  int numGroup = (cnt >= 256) ? (cnt >> 8) : 1;
    
  se::rhi::ComputePassEncoder* encoder = beginPass(context);
  encoder->pushConstants(&maxDepth, (uint32_t)rhi::ShaderStageBit::COMPUTE, 0, sizeof(maxDepth));
  encoder->dispatchWorkgroups(numGroup, 1, 1);
  encoder->end();
}

SumReductionOneLayerPass::SumReductionOneLayerPass(int maxDepth) :maxDepth(maxDepth) {
  auto [comp] = gfx::GFXContext::load_shader_slang(
    "cbt/sum-reduction-one.slang",
    std::array<std::pair<std::string, rhi::ShaderStageBit>, 1>{
      std::make_pair("ComputeMain", rhi::ShaderStageBit::COMPUTE),
  }, {}, true);
  rdg::ComputePass::init(comp.get());
}

auto SumReductionOneLayerPass::reflect() noexcept -> rdg::PassReflection {
  rdg::PassReflection reflector;
  reflector.addInputOutput("CBTree")
    .isBuffer().withUsages((uint32_t)rhi::BufferUsageBit::COPY_DST)
    .consume(rdg::BufferInfo::ConsumeEntry{}
    .setAccess((uint32_t)rhi::AccessFlagBits::SHADER_WRITE_BIT
             | (uint32_t)rhi::AccessFlagBits::SHADER_READ_BIT)
    .addStage((uint32_t)rhi::PipelineStageBit::COMPUTE_SHADER_BIT));
  return reflector;
}
  
auto SumReductionOneLayerPass::execute(rdg::RenderContext* context, rdg::RenderData const& renderData) noexcept -> void {
  gfx::BufferHandle cbt = renderData.getBuffer("CBTree");

  updateBindings(context, {
    {"cbt_heap", rhi::BindingResource{{cbt->buffer.get(), 0, cbt->buffer->size()}}},
  });

  se::rhi::ComputePassEncoder* encoder = beginPass(context);
  int it = maxDepth;
  while (--it >= 0) {
    int cnt = 1 << it;
    int numGroup = (cnt >= 256) ? (cnt >> 8) : 1;
    encoder->pushConstants(&it, (uint32_t)rhi::ShaderStageBit::COMPUTE, 0, sizeof(it));
    encoder->dispatchWorkgroups(numGroup, 1, 1);
    context->cmdEncoder->pipelineBarrier(rhi::BarrierDescriptor{
      (uint32_t)rhi::PipelineStageBit::COMPUTE_SHADER_BIT,
      (uint32_t)rhi::PipelineStageBit::COMPUTE_SHADER_BIT, 0, {},
      { rhi::BufferMemoryBarrierDescriptor{cbt->buffer.get(), 
        (uint32_t)rhi::AccessFlagBits::SHADER_READ_BIT | (uint32_t)rhi::AccessFlagBits::SHADER_WRITE_BIT,
        (uint32_t)rhi::AccessFlagBits::SHADER_READ_BIT | (uint32_t)rhi::AccessFlagBits::SHADER_WRITE_BIT,
        0, cbt->buffer->size()}}, {}
    });
  }
  encoder->end();
}

CBTSpatialTreeVisualizePass::CBTSpatialTreeVisualizePass(size_t indirect_offset) 
  :indirect_offset(indirect_offset) {
  auto [vert, frag] = gfx::GFXContext::load_shader_slang(
    "cbt/sbt-visualize.slang",
    std::array<std::pair<std::string, rhi::ShaderStageBit>, 2>{
      std::make_pair("vertexMain", rhi::ShaderStageBit::VERTEX),
      std::make_pair("fragmentMain", rhi::ShaderStageBit::FRAGMENT),
    });
  rdg::RenderPass::init(vert.get(), frag.get());
}

auto CBTSpatialTreeVisualizePass::reflect() noexcept -> rdg::PassReflection {
  rdg::PassReflection reflector;
  reflector.addInputOutput("Color").isTexture()
    .withSize(vec3{ 1.f,1.f,1.f })
    .withFormat(rhi::TextureFormat::RGBA32_FLOAT)
    .withUsages((uint32_t)rhi::TextureUsageBit::COLOR_ATTACHMENT)
    .consume(rdg::TextureInfo::ConsumeEntry{
      rdg::TextureInfo::ConsumeType::ColorAttachment}
        .setAttachmentLoc(0));
  reflector.addInputOutput("Depth").isTexture()
    .withFormat(rhi::TextureFormat::DEPTH32_FLOAT)
    .withUsages((uint32_t)rhi::TextureUsageBit::DEPTH_ATTACHMENT)
    .consume(rdg::TextureInfo::ConsumeEntry{
      rdg::TextureInfo::ConsumeType::DepthStencilAttachment}
        .enableDepthWrite(true)
        .setAttachmentLoc(0)
        .setDepthCompareFn(rhi::CompareFunction::LESS_EQUAL));
  reflector.addInput("CBTree")
    .isBuffer().withUsages((uint32_t)rhi::BufferUsageBit::STORAGE)
    .consume(rdg::BufferInfo::ConsumeEntry{}
    .setAccess((uint32_t)rhi::AccessFlagBits::SHADER_READ_BIT)
    .addStage((uint32_t)rhi::PipelineStageBit::VERTEX_SHADER_BIT));
  reflector.addInput("Indirect")
    .isBuffer().withUsages((uint32_t)rhi::BufferUsageBit::INDIRECT)
    .consume(rdg::BufferInfo::ConsumeEntry{}
    .setAccess((uint32_t)rhi::AccessFlagBits::INDIRECT_COMMAND_READ_BIT)
    .addStage((uint32_t)rhi::PipelineStageBit::DRAW_INDIRECT_BIT));
  return reflector;
}

auto CBTSpatialTreeVisualizePass::execute(
  rdg::RenderContext* context,
  rdg::RenderData const& renderData) noexcept -> void {
  gfx::TextureHandle color = renderData.getTexture("Color");
  gfx::TextureHandle depth = renderData.getTexture("Depth");
  gfx::BufferHandle cbt = renderData.getBuffer("CBTree");
  gfx::BufferHandle idr = renderData.getBuffer("Indirect");
  gfx::SceneHandle scene = renderData.getScene();

  setRenderPassDescriptor(rhi::RenderPassDescriptor{
    { rhi::RenderPassColorAttachment{color->getRTV(0, 0, 1),
      nullptr, {0, 0, 0, 1}, rhi::LoadOp::LOAD, rhi::StoreOp::STORE},},
      rhi::RenderPassDepthStencilAttachment{
        depth->getDSV(0, 0, 1), 1, rhi::LoadOp::LOAD, rhi::StoreOp::STORE, false,
        0, rhi::LoadOp::DONT_CARE, rhi::StoreOp::DONT_CARE, false},
      });
  
  updateBindings(context, {
    {"GPUScene_camera", scene->getGPUScene()->bindingResourceCamera() } ,
    {"GPUScene_position", scene->getGPUScene()->bindingResourcePosition() },
    {"GPUScene_index", scene->getGPUScene()->bindingResourceIndex() },
    {"GPUScene_vertex", scene->getGPUScene()->bindingResourceVertex() },
    {"GPUScene_geometry", scene->getGPUScene()->bindingResourceGeometry() },
    {"cbt_heap", rhi::BindingResource{{cbt->buffer.get(), 0, cbt->buffer->size()}}},
  });

  pConst.resolution = { color->texture->width(), color->texture->width() };
  pConst.camera_index = scene->getEditorActiveCameraIndex();

  rhi::RenderPassEncoder* encoder = beginPass(context, color.get());
  encoder->pushConstants(&pConst, (uint32_t)rhi::ShaderStageBit::VERTEX, 0, sizeof(pConst));
  encoder->drawIndirect(idr->buffer.get(), 0, 1, 0);
  encoder->end();
}

auto CBTSpatialTreeVisualizePass::renderUI() noexcept -> void {
  ImGui::DragFloat("Line width", &pConst.line_width, 0.1, 0, 10);
}

TestCBTPass::TestCBTPass(int maxDepth) {
  auto [comp] = gfx::GFXContext::load_shader_slang(
    "cbt/test.slang",
    std::array<std::pair<std::string, rhi::ShaderStageBit>, 1>{
      std::make_pair("ComputeMain", rhi::ShaderStageBit::COMPUTE),
  }, {}, false);
  rdg::ComputePass::init(comp.get());
}

auto TestCBTPass::reflect() noexcept -> rdg::PassReflection {
  rdg::PassReflection reflector;
  reflector.addInput("CBTree")
    .isBuffer().withUsages((uint32_t)rhi::BufferUsageBit::STORAGE)
    .consume(rdg::BufferInfo::ConsumeEntry{}
    .setAccess((uint32_t)rhi::AccessFlagBits::SHADER_READ_BIT)
    .addStage((uint32_t)rhi::PipelineStageBit::COMPUTE_SHADER_BIT));
  reflector.addOutput("DebugBuffer")
    .isBuffer().withSize(32).withUsages((uint32_t)rhi::BufferUsageBit::STORAGE)
    .consume(rdg::BufferInfo::ConsumeEntry{}
    .setAccess((uint32_t)rhi::AccessFlagBits::SHADER_READ_BIT)
    .addStage((uint32_t)rhi::PipelineStageBit::COMPUTE_SHADER_BIT));
  return reflector;
}

auto TestCBTPass::execute(
  rdg::RenderContext* context,
  rdg::RenderData const& renderData) noexcept -> void {
  gfx::BufferHandle cbt = renderData.getBuffer("CBTree");
  gfx::BufferHandle dbgbuf = renderData.getBuffer("DebugBuffer");

  updateBindings(context, {
    {"cbt_heap", rhi::BindingResource{{cbt->buffer.get(), 0, cbt->buffer->size()}}},
    {"dbgbuf", dbgbuf->getBindingResource()},
  });

  se::rhi::ComputePassEncoder* encoder = beginPass(context);
  encoder->dispatchWorkgroups(1, 1, 1);
  encoder->end();
}
}