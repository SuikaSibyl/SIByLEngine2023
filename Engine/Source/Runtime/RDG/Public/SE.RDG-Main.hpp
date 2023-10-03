#pragma once
#include <array>
#include <format>
#include <functional>
#include <memory>
#include <optional>
#include <set>
#include <stack>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>
#include "../../Application/Public/SE.Application.Config.h"
#include "SE.RDG-DAG.hpp"
#include <SE.Math.hpp>
#include <SE.RHI.hpp>
#include <SE.GFX.hpp>
#include <Print/SE.Core.Log.hpp>
#include <Resource/SE.Core.Resource.hpp>

#include <SE.RHI.hpp>
#include <SE.GFX.hpp>
#include <SE.Editor.Core.hpp>
#include <SE.Math.Geometric.hpp>
#include <Print/SE.Core.Log.hpp>
#include <Resource/SE.Core.Resource.hpp>

import SE.Platform.Window;

namespace SIByL::Editor {
struct RDGViewerWidget;
}

namespace SIByL::RDG {
SE_EXPORT enum struct ResourceFlag : uint32_t {
  None = 0 << 0,
  Persistent = 1 << 0,
};
SE_EXPORT using ResourceFlags = uint32_t;

SE_EXPORT inline uint64_t MaxPossible64 = uint64_t(-1);

/**
 * Information that describe a buffer resource in RDG
 * @param * size		 : size of the buffer
 * @param * usages;	 : usage of the texture};
 * @param * flags	 : some flags describe the resource
 */
SE_EXPORT struct BufferInfo {
  uint32_t size = 0;
  RHI::BufferUsagesFlags usages = 0;
  ResourceFlags flags = 0;
  GFX::Buffer* reference = nullptr;

  inline auto withSize(uint32_t size) noexcept -> BufferInfo&;
  inline auto withUsages(RHI::BufferUsagesFlags usages) noexcept -> BufferInfo&;
  inline auto withFlags(ResourceFlags flags) noexcept -> BufferInfo&;

  struct ConsumeEntry {
    RHI::AccessFlags access;
    RHI::PipelineStageFlags stages;
    uint64_t offset = 0;
    uint64_t size = MaxPossible64;

    inline auto addStage(RHI::PipelineStageFlags stage) noexcept
        -> ConsumeEntry& {
      stages |= stage;
      return *this;
    }
    inline auto setAccess(uint32_t acc) noexcept -> ConsumeEntry& {
      access = acc;
      return *this;
    }
    inline auto setSubresource(uint64_t offset, uint64_t size) noexcept
        -> ConsumeEntry& {
      this->offset = offset;
      this->size = size;
      return *this;
    }
  };

  inline auto consume(ConsumeEntry const& entry) noexcept -> BufferInfo&;

  std::vector<ConsumeEntry> consumeHistories;
};

/**
 * Information that describe a texture resource in RDG
 * @param * size		 : size of the texture
 * @param * sizeDef;	 : how the size is defined
 * @param * exlevels  : levels of LoDport
 * @param * layers;	 : levels of layersenum
 * @param * samples;	 : number of samples for attachments
 * @param * format;	 : format of the texture
 * @param * Relative	 : define size by relative value
 * @param * usages;	 : usage of the texture};
 * @param * flag		 : some flags describe the resource
 */
SE_EXPORT inline uint32_t MaxPossible = uint32_t(-1);

SE_EXPORT struct TextureInfo {
  TextureInfo() = default;
  ~TextureInfo() = default;

  enum struct SizeDefine {
    Absolute,              // * define size by absolute value
    Relative,              // * define size by relative value
    RelativeToAnotherTex,  // * define size by relative value
  };
  union Size {
    Math::ivec3 absolute;  // * the absolute size
    Math::vec3 relative;   // * the relative size
  };
  Size size = Size{.relative = Math::vec3{1.f}};
  SizeDefine sizeDef = SizeDefine::Relative;
  uint32_t levels = 1;
  uint32_t layers = 1;
  uint32_t samples = 1;
  RHI::TextureFormat format = RHI::TextureFormat::RGBA8_UNORM;
  RHI::TextureUsagesFlags usages = 0;
  RHI::TextureFlags tflags = RHI::TextureFlags::NONE;
  ResourceFlags flags = 0;
  RHI::PipelineStageFlags stages = 0;
  RHI::AccessFlags access = 0;
  RHI::TextureLayout laytout = RHI::TextureLayout::GENERAL;
  RHI::ShaderStagesFlags sflags = 0;

  GFX::Texture* reference = nullptr;
  std::string sizeRefName;

  enum struct ConsumeType {
    ColorAttachment,
    DepthStencilAttachment,
    TextureBinding,
    StorageBinding,
  };

  struct ConsumeEntry {
    ConsumeType type;
    RHI::AccessFlags access;
    RHI::PipelineStageFlags stages;
    uint32_t level_beg = 0;
    uint32_t level_end = 1;
    uint32_t mip_beg = 0;
    uint32_t mip_end = 1;
    RHI::TextureLayout layout;
    bool depthWrite = false;
    RHI::CompareFunction depthCmp = RHI::CompareFunction::ALWAYS;
    uint32_t attachLoc = uint32_t(-1);

    inline auto addStage(RHI::PipelineStageFlags stage) noexcept
        -> ConsumeEntry& {
      stages |= stage;
      return *this;
    }
    inline auto setLayout(RHI::TextureLayout _layout) noexcept
        -> ConsumeEntry& {
      layout = _layout;
      return *this;
    }
    inline auto enableDepthWrite(bool set) noexcept -> ConsumeEntry& {
      depthWrite = set;
      return *this;
    }
    inline auto setDepthCompareFn(RHI::CompareFunction fn) noexcept
        -> ConsumeEntry& {
      depthCmp = fn;
      return *this;
    }
    inline auto setSubresource(uint32_t mip_beg, uint32_t mip_end,
                               uint32_t level_beg, uint32_t level_end) noexcept
        -> ConsumeEntry& {
      this->mip_beg = mip_beg;
      this->mip_end = mip_end;
      this->level_beg = level_beg;
      this->level_end = level_end;
      return *this;
    }
    inline auto setAttachmentLoc(uint32_t loc) noexcept -> ConsumeEntry& {
      attachLoc = loc;
      return *this;
    }
    inline auto setAccess(uint32_t acc) noexcept -> ConsumeEntry& {
      access = acc;
      return *this;
    }
  };

  inline auto consume(ConsumeEntry const& entry) noexcept -> TextureInfo&;

  inline auto setInfo(TextureInfo const& x) noexcept -> TextureInfo&;
  inline auto withSize(Math::ivec3 absolute) noexcept -> TextureInfo&;
  inline auto withSize(Math::vec3 relative) noexcept -> TextureInfo&;
  inline auto withSizeRelative(std::string const& src,
                               Math::vec3 relative = {1.}) noexcept
      -> TextureInfo&;
  inline auto withLevels(uint32_t levels) noexcept -> TextureInfo&;
  inline auto withLayers(uint32_t layers) noexcept -> TextureInfo&;
  inline auto withSamples(uint32_t samples) noexcept -> TextureInfo&;
  inline auto withFormat(RHI::TextureFormat format) noexcept -> TextureInfo&;
  inline auto withStages(RHI::ShaderStagesFlags flags) noexcept -> TextureInfo&;
  inline auto withUsages(RHI::TextureUsagesFlags usages) noexcept
      -> TextureInfo&;
  inline auto withFlags(ResourceFlags flags) noexcept -> TextureInfo&;

  inline auto getSize(Math::ivec3 ref) const noexcept -> RHI::Extend3D {
    if (sizeDef == SizeDefine::Absolute)
      return RHI::Extend3D{uint32_t(size.absolute.x), uint32_t(size.absolute.y),
                           uint32_t(size.absolute.z)};
    else
      return RHI::Extend3D{uint32_t(std::ceil(size.relative.x * ref.x)),
                           uint32_t(std::ceil(size.relative.y * ref.y)),
                           uint32_t(std::ceil(size.relative.z * ref.z))};
  }

  std::vector<ConsumeEntry> consumeHistories;
};

SE_EXPORT inline auto toTextureDescriptor(TextureInfo const& info,
                                       Math::ivec3 ref_size) noexcept
    -> RHI::TextureDescriptor {
  RHI::Extend3D size = info.getSize(ref_size);
  return RHI::TextureDescriptor{
      size,
      info.levels,
      info.layers,
      info.samples,
      size.depthOrArrayLayers == 1 ? RHI::TextureDimension::TEX2D
                                   : RHI::TextureDimension::TEX3D,
      info.format,
      info.usages | (uint32_t)RHI::TextureUsage::TEXTURE_BINDING |
          (uint32_t)RHI::TextureUsage::COPY_SRC,
      std::vector<RHI::TextureFormat>{info.format},
      info.tflags};
}

SE_EXPORT inline auto toBufferDescriptor(BufferInfo const& info) noexcept
    -> RHI::BufferDescriptor {
  return RHI::BufferDescriptor{info.size, info.usages};
}

/** Information that describe a resource in RDG */
SE_EXPORT struct ResourceInfo {
  ResourceInfo() : type(Type::UNKNOWN){};

  enum struct Type { UNKNOWN, Buffer, Texture } type = Type::UNKNOWN;
  struct Info {
    void* null;
    BufferInfo buffer;
    TextureInfo texture;
  } info;

  size_t devirtualizeID = (-1);
  /* declare the resource is a buffer */
  inline auto isBuffer() noexcept -> BufferInfo&;
  /* declare the resource is a texture */
  inline auto isTexture() noexcept -> TextureInfo&;

  ResourceInfo* prev = nullptr;
};

SE_EXPORT struct Resource {
  Core::GUID guid;
};

SE_EXPORT struct TextureResource : public Resource {
  RHI::TextureDescriptor desc;
  GFX::Texture* texture;
  std::string name;

  struct ConsumeHistory {
    size_t passID;
    std::vector<TextureInfo::ConsumeEntry> entries;
  };
  std::vector<ConsumeHistory> cosumeHistories;
};

SE_EXPORT struct BufferResource : public Resource {
  RHI::BufferDescriptor desc;
  GFX::Buffer* buffer;

  struct ConsumeHistory {
    size_t passID;
    std::vector<BufferInfo::ConsumeEntry> entries;
  };
  std::vector<ConsumeHistory> cosumeHistories;
};

SE_EXPORT struct PassReflection {
  PassReflection() {
    inputResources.clear();
    outputResources.clear();
    inputOutputResources.clear();
    internalResources.clear();
  }

  auto addInput(std::string const& name) noexcept -> ResourceInfo&;
  auto addOutput(std::string const& name) noexcept -> ResourceInfo&;
  auto addInputOutput(std::string const& name) noexcept -> ResourceInfo&;
  auto addInternal(std::string const& name) noexcept -> ResourceInfo&;

  std::unordered_map<std::string, ResourceInfo> inputResources = {};
  std::unordered_map<std::string, ResourceInfo> outputResources = {};
  std::unordered_map<std::string, ResourceInfo> inputOutputResources = {};
  std::unordered_map<std::string, ResourceInfo> internalResources = {};

  auto getDepthStencilState() noexcept -> RHI::DepthStencilState;
  auto getColorTargetState() noexcept -> std::vector<RHI::ColorTargetState>;

  auto getResourceInfo(std::string const& name) noexcept -> ResourceInfo* {
    if (inputResources.find(name) != inputResources.end()) {
      return &inputResources[name];
    } else if (outputResources.find(name) != outputResources.end()) {
      return &outputResources[name];
    } else if (inputOutputResources.find(name) != inputOutputResources.end()) {
      return &inputOutputResources[name];
    } else if (internalResources.find(name) != internalResources.end()) {
      return &internalResources[name];
    }
    Core::LogManager::Error(std::format(
        "RDG::PassReflection::getResourceInfo Failed to find resource \"{0}\"",
        name));
    return nullptr;
  }
};

SE_EXPORT struct RenderContext {
  size_t flightIdx = 0;
  RHI::CommandEncoder* cmdEncoder;
};

struct Graph;
struct Pass;
struct PipelinePass;

SE_EXPORT struct RenderData {
  struct DelegateData {
    RHI::CommandEncoder* cmdEncoder;
    union {
      RHI::RenderPassEncoder* render;
      RHI::ComputePassEncoder* compute;
      RHI::RayTracingPassEncoder* trace;
    } passEncoder;
    PipelinePass* pipelinePass;
    void* customData;
  };

  inline auto setDelegate(
      std::string const& name,
      std::function<void(DelegateData const&)> const& fn) noexcept -> void {
    delegates[name] = fn;
  }
  inline auto getDelegate(std::string const& name) const noexcept
      -> std::function<void(DelegateData const&)> const& {
    auto const& iter = delegates.find(name);
    if (iter == delegates.end()) return nullptr;
    return iter->second;
  }

  inline auto setBindingResource(
      std::string const& name,
      RHI::BindingResource const& bindGroupEntry) noexcept
      -> void {
    bindingResources[name]=bindGroupEntry;
  }
  inline auto getBindingResource(std::string const& name) const noexcept
      -> std::optional<RHI::BindingResource> {
    auto const& iter = bindingResources.find(name);
    if (iter == bindingResources.end()) return std::nullopt;
    return iter->second;
  }

  inline auto setBindGroupEntries(
      std::string const& name,
      std::vector<RHI::BindGroupEntry>* bindGroup) noexcept -> void {
    bindGroups[name] = bindGroup;
  }
  inline auto getBindGroupEntries(std::string const& name) const noexcept
      -> std::vector<RHI::BindGroupEntry>* {
    auto const& iter = bindGroups.find(name);
    if (iter == bindGroups.end()) return nullptr;
    return iter->second;
  }

  inline auto getTexture(std::string const& name) const noexcept
      -> GFX::Texture*;
  inline auto getBuffer(std::string const& name) const noexcept -> GFX::Buffer*;

  inline auto setUVec2(std::string const& name, Math::uvec2 v) noexcept
      -> void {
    uvec2s[name] = v;
  }
  inline auto getUVec2(std::string const& name) const noexcept -> Math::uvec2 {
    auto const& iter = uvec2s.find(name);
    if (iter == uvec2s.end()) return Math::uvec2{0};
    return iter->second;
  }

  inline auto setUInt(std::string const& name, uint32_t v) noexcept -> void {
    uints[name] = v;
  }
  inline auto getUInt(std::string const& name) const noexcept -> uint32_t {
    auto const& iter = uints.find(name);
    if (iter == uints.end()) return 0;
    return iter->second;
  }

  inline auto setPtr(std::string const& name, void* v) noexcept -> void {
    ptrs[name] = v;
  }
  inline auto getPtr(std::string const& name) const noexcept -> void* {
    auto const& iter = ptrs.find(name);
    if (iter == ptrs.end()) return 0;
    return iter->second;
  }

  inline auto setMat4(std::string const& name, Math::mat4 m) noexcept -> void {
    mat4s[name] = m;
  }
  inline auto getMat4(std::string const& name) const noexcept -> Math::mat4 {
    auto const& iter = mat4s.find(name);
    if (iter == mat4s.end()) return Math::mat4();
    return iter->second;
  }

  Graph* graph;
  Pass* pass;

  std::unordered_map<std::string, std::vector<RHI::BindGroupEntry>*> bindGroups;
  std::unordered_map<std::string, RHI::BindingResource> bindingResources;
  std::unordered_map<std::string, Math::uvec2> uvec2s;
  std::unordered_map<std::string, uint32_t> uints;
  std::unordered_map<std::string, std::function<void(DelegateData const&)>>
      delegates;
  std::unordered_map<std::string, void*> ptrs;
  std::unordered_map<std::string, Math::mat4> mat4s;
};

SE_EXPORT struct Pass {
  Pass() = default;
  virtual ~Pass() = default;
  virtual auto reflect() noexcept -> PassReflection = 0;

  virtual auto execute(RenderContext* context,
                       RenderData const& renderData) noexcept -> void = 0;

  virtual auto renderUI() noexcept -> void {}

  virtual auto onInteraction(Platform::Input* input,
                             Editor::Widget::WidgetInfo* info) noexcept
      -> void {}

  virtual auto generateMarker() noexcept -> void {
    marker.name = identifier;
    marker.color = {0.490, 0.607, 0.003, 1.};
  }

  PassReflection pReflection;
  RHI::DebugUtilLabelDescriptor marker;
  std::string identifier;
  std::vector<size_t> subgraphStack;

 protected:
  virtual auto init() noexcept -> void;
};

SE_EXPORT struct DummyPass : public Pass {
  virtual auto reflect() noexcept -> PassReflection = 0;
  virtual auto execute(RenderContext* context,
                       RenderData const& renderData) noexcept -> void override {
  }
};

SE_EXPORT struct AliasDict {
  struct Value {
    std::string pass;
    std::string resource;
  };
  std::unordered_map<std::string, Value> dict;

  inline auto addAlias(std::string const& subgraph_resource,
                       std::string const& pass,
                       std::string const& pass_resource) noexcept
      -> AliasDict& {
    dict[subgraph_resource] = Value{pass, pass_resource};
    return *this;
  }
};

SE_EXPORT struct Subgraph {
  virtual auto onRegister(Graph* graph) noexcept -> void = 0;

  virtual auto alias() noexcept -> AliasDict = 0;

  inline auto CONCAT(std::string const& name) noexcept -> std::string {
    return identifier + "." + name;
  }

  virtual auto generateMarker() noexcept -> void {
    marker.name = identifier;
    marker.color = {0.235, 0.321, 0.2, 1.};
  }

  std::string identifier;
  RHI::DebugUtilLabelDescriptor marker;
};

SE_EXPORT struct PipelinePass : public Pass {
  PipelinePass() = default;
  virtual ~PipelinePass() = default;
  GFX::ShaderReflection reflection;
  std::vector<std::unique_ptr<RHI::BindGroupLayout>> bindgroupLayouts;
  std::vector<
      std::array<std::unique_ptr<RHI::BindGroup>, MULTIFRAME_FLIGHTS_COUNT>>
      bindgroups;
  std::unique_ptr<RHI::PipelineLayout> pipelineLayout;

  inline auto getBindGroup(RenderContext* context, uint32_t size) noexcept
      -> RHI::BindGroup* {
    return bindgroups[size][context->flightIdx].get();
  }
  // Direct update binding with binding location find by string
  auto updateBinding(RenderContext* context, std::string const& name,
                     RHI::BindingResource const& resource) noexcept -> void;
  auto updateBindings(
      RenderContext* context,
      std::vector<std::pair<std::string, RHI::BindingResource>> const&
          bindings) noexcept -> void;
 protected:
  virtual auto init(std::vector<GFX::ShaderModule*> shaderModules) noexcept
      -> void;
};

SE_EXPORT struct RenderPass : public PipelinePass {
  RenderPass() = default;
  virtual ~RenderPass() = default;
  RHI::RenderPassDescriptor renderPassDescriptor;
  std::array<std::unique_ptr<RHI::RenderPipeline>, MULTIFRAME_FLIGHTS_COUNT>
      pipelines;
  std::array<std::unique_ptr<RHI::RenderPassEncoder>, MULTIFRAME_FLIGHTS_COUNT>
      passEncoders;

  inline auto beginPass(RDG::RenderContext* context,
                        GFX::Texture* target) noexcept
      -> RHI::RenderPassEncoder* {
    passEncoders[context->flightIdx] =
        context->cmdEncoder->beginRenderPass(renderPassDescriptor);
    passEncoders[context->flightIdx]->setPipeline(
        pipelines[context->flightIdx].get());
    prepareDispatch(context, target);
    return passEncoders[context->flightIdx].get();
  }

  inline auto beginPass(RDG::RenderContext* context, uint32_t width,
                        uint32_t height) noexcept -> RHI::RenderPassEncoder* {
    passEncoders[context->flightIdx] =
        context->cmdEncoder->beginRenderPass(renderPassDescriptor);
    passEncoders[context->flightIdx]->setPipeline(
        pipelines[context->flightIdx].get());
    prepareDispatch(context, width, height);
    return passEncoders[context->flightIdx].get();
  }

  inline auto prepareDelegateData(RenderContext* context,
                                  RenderData const& renderData) noexcept
      -> RenderData::DelegateData {
    RenderData::DelegateData data;
    data.cmdEncoder = context->cmdEncoder;
    data.passEncoder.render = passEncoders[context->flightIdx].get();
    data.pipelinePass = this;
    return data;
  }

  virtual auto generateMarker() noexcept -> void {
    marker.name = identifier;
    marker.color = {0.764, 0.807, 0.725, 1.};
  }

 protected:
  inline auto prepareDispatch(RDG::RenderContext* context,
                              GFX::Texture* target) noexcept -> void {
    for (size_t i = 0; i < bindgroups.size(); ++i)
      passEncoders[context->flightIdx]->setBindGroup(
          i, bindgroups[i][context->flightIdx].get());
    passEncoders[context->flightIdx]->setViewport(
        0, 0, target->texture->width(), target->texture->height(), 0, 1);
    passEncoders[context->flightIdx]->setScissorRect(
        0, 0, target->texture->width(), target->texture->height());
  }
  inline auto prepareDispatch(RDG::RenderContext* context, uint32_t width,
                              uint32_t height) noexcept -> void {
    for (size_t i = 0; i < bindgroups.size(); ++i)
      passEncoders[context->flightIdx]->setBindGroup(
          i, bindgroups[i][context->flightIdx].get());
    passEncoders[context->flightIdx]->setViewport(0, 0, width, height, 0, 1);
    passEncoders[context->flightIdx]->setScissorRect(0, 0, width, height);
  }

  using RenderPipelineDescCallback = std::function<void(RHI::RenderPipelineDescriptor&)>;
  virtual auto init(GFX::ShaderModule* vertex, GFX::ShaderModule* fragment,
                    std::optional<RenderPipelineDescCallback> callback =
                        std::nullopt) noexcept -> void;
  virtual auto init(GFX::ShaderModule* vertex, GFX::ShaderModule* geometry,
                    GFX::ShaderModule* fragment,
                    std::optional<RenderPipelineDescCallback> callback =
                        std::nullopt) noexcept -> void;
};

SE_EXPORT struct FullScreenPass : public RenderPass {
  FullScreenPass() = default;
  virtual ~FullScreenPass() = default;

  inline auto dispatchFullScreen(RDG::RenderContext* context) noexcept -> void {
    passEncoders[context->flightIdx]->draw(3, 1, 0, 0);
  }

 protected:
  static GFX::ShaderModule* fullscreen_vertex;
  virtual auto init(GFX::ShaderModule* fragment) noexcept -> void;

 private:
  virtual auto generateMarker() noexcept -> void {
    marker.name = identifier;
    marker.color = {0.682, 0.764, 0.752, 1.};
  }
};

SE_EXPORT struct ComputePass : public PipelinePass {
  ComputePass() = default;
  virtual ~ComputePass() = default;
  std::array<std::unique_ptr<RHI::ComputePipeline>, MULTIFRAME_FLIGHTS_COUNT>
      pipelines;
  std::array<std::unique_ptr<RHI::ComputePassEncoder>, MULTIFRAME_FLIGHTS_COUNT>
      passEncoders;

  inline auto beginPass(RDG::RenderContext* context) noexcept
      -> RHI::ComputePassEncoder* {
    passEncoders[context->flightIdx] =
        context->cmdEncoder->beginComputePass(RHI::ComputePassDescriptor{});
    passEncoders[context->flightIdx]->setPipeline(
        pipelines[context->flightIdx].get());
    prepareDispatch(context);
    return passEncoders[context->flightIdx].get();
  }

  virtual auto generateMarker() noexcept -> void {
    marker.name = identifier;
    marker.color = {0.6, 0.721, 0.780, 1.};
  }

 protected:
  inline auto prepareDispatch(RDG::RenderContext* context) noexcept -> void {
    for (size_t i = 0; i < bindgroups.size(); ++i)
      passEncoders[context->flightIdx]->setBindGroup(
          i, bindgroups[i][context->flightIdx].get());
  }

  virtual auto init(GFX::ShaderModule* comp) noexcept -> void;
};

SE_EXPORT struct RayTracingPass : public PipelinePass {
  RayTracingPass() = default;
  virtual ~RayTracingPass() = default;
  RHI::RayTracingPipelineDescriptor rtPipelineDescriptor;
  std::array<std::unique_ptr<RHI::RayTracingPipeline>, MULTIFRAME_FLIGHTS_COUNT>
      pipelines;
  std::array<std::unique_ptr<RHI::RayTracingPassEncoder>,
             MULTIFRAME_FLIGHTS_COUNT>
      passEncoders;

  inline auto beginPass(RDG::RenderContext* context) noexcept
      -> RHI::RayTracingPassEncoder* {
    passEncoders[context->flightIdx] = context->cmdEncoder->beginRayTracingPass(
        RHI::RayTracingPassDescriptor{});
    passEncoders[context->flightIdx]->setPipeline(
        pipelines[context->flightIdx].get());
    prepareDispatch(context);
    return passEncoders[context->flightIdx].get();
  }

  virtual auto generateMarker() noexcept -> void {
    marker.name = identifier;
    marker.color = {0.517, 0.674, 0.807, 1.};
  }

 protected:
  inline auto prepareDispatch(RDG::RenderContext* context) noexcept -> void {
    for (size_t i = 0; i < bindgroups.size(); ++i)
      passEncoders[context->flightIdx]->setBindGroup(
          i, bindgroups[i][context->flightIdx].get());
  }

  virtual auto init(GFX::SBTsDescriptor const& sbt, uint32_t max_depth) noexcept
      -> void;
};

SE_EXPORT struct Graph {
  /** validate the graph */
  auto validate() noexcept -> bool;

  /** build the graph */
  auto build() noexcept -> bool;

  /** execute the graph */
  auto execute(RHI::CommandEncoder* encoder) noexcept -> void;

  /** render graph ui */
  virtual auto renderUI() noexcept -> void {}

  /**
   * Add a RDPass to the RDGraph
   * @param pass		: the pass to add
   * @param identifier	: identifier of the pass added
   */
  auto addPass(std::unique_ptr<Pass>&& pass,
               std::string const& identifier) noexcept -> void;

  /**
   * Add a Subgraph to the RDGraph
   * @param subgraph	: the subgraph to add
   * @param identifier	: identifier of the pass added
   */
  auto addSubgraph(std::unique_ptr<Subgraph>&& subgraph,
                   std::string const& identifier) noexcept -> void;

  /**
   * Adding dependency from src_pass.src_resource to dst_pass.dst_resource.
   * @param src_pass: identifier of source pass
   * @param src_pass: identifier of the source output of source pass
   * @param dst_pass: identifier of destination pass
   * @param dst_pass: identifier of the destination input of destination pass
   */
  auto addEdge(std::string const& src_pass, std::string const& src_resource,
               std::string const& dst_pass,
               std::string const& dst_resource) noexcept -> void;

  auto addEdge(std::string const& src_pass,
               std::string const& dst_pass) noexcept -> void;

  /**
   * Set external resource
   * @param pass		: the pass where the resource is in
   * @param resource	: the resource to set
   */
  auto setExternal(std::string const& pass, std::string const& resource,
                   GFX::Texture* tex) noexcept -> void;
  auto setExternal(std::string const& pass, std::string const& resource,
                   GFX::Buffer* tex) noexcept -> void;

  /**
   * Mark the main output of the render graph.
   * @param pass		: the pass produce the output
   * @param identifier	: identifier of the output resource
   */
  auto markOutput(std::string const& pass, std::string const& output) noexcept
      -> void;

  inline auto getOutput() noexcept -> GFX::Texture*;

  inline auto getTextureResource(std::string const& pass,
                                 std::string const& output) noexcept
      -> GFX::Texture*;
  
  inline auto getBufferResource(std::string const& pass,
                                std::string const& output) noexcept
      -> GFX::Buffer*;

  RenderData renderData;

  inline auto getFlattenedPasses() noexcept -> std::vector<size_t> const& {
    return flattenedPasses;
  }
  inline auto getPass(size_t i) noexcept -> Pass* { return passes[i].get(); }

  inline auto getPass(std::string const& name) noexcept -> Pass* { 
    auto iter = passNameList.find(name);
    if (iter == passNameList.end())
      return nullptr;
    else
      return passes[iter->second].get();
  }

 private:
  friend RenderData;
  friend Editor::RDGViewerWidget;
  DAG dag;
  size_t passID = 0;
  size_t subgraphID = 0;
  std::unordered_map<std::string, size_t> passNameList;
  std::unordered_map<size_t, std::unique_ptr<Pass>> passes;
  std::unordered_map<size_t, std::string> subgraphNameList;
  std::unordered_map<std::string, std::unique_ptr<Subgraph>> subgraphs;
  std::unordered_map<std::string, AliasDict> subgraphsAlias;
  std::unordered_map<size_t, std::vector<RHI::BarrierDescriptor>> barriers;
  std::vector<size_t> flattenedPasses;
  std::vector<size_t> subgraphStack;

  std::string output_pass;
  std::string output_resource;

  Math::ivec3 standardSize = {1280, 720, 1};

  size_t resourceID = 0;
  std::unordered_map<size_t, std::unique_ptr<TextureResource>> textureResources;
  std::unordered_map<size_t, std::unique_ptr<BufferResource>> bufferResources;

  auto generateBufferBarriers() noexcept -> void;
  auto generateTextureBarriers() noexcept -> void;

  auto decodeAlias(std::string& pass, std::string& resource) noexcept -> void;
};

SE_EXPORT struct Pipeline {
  Pipeline() = default;
  virtual ~Pipeline() = default;

  virtual auto build() noexcept -> void = 0;

  /** render pipeline ui */
  virtual auto renderUI() noexcept -> void {}

  virtual auto execute(RHI::CommandEncoder* encoder) noexcept -> void = 0;

  virtual auto getActiveGraphs() noexcept -> std::vector<Graph*> = 0;

  virtual auto getOutput() noexcept -> GFX::Texture* = 0;
};

SE_EXPORT struct SingleGraphPipeline : public Pipeline {
  SingleGraphPipeline() = default;
  virtual ~SingleGraphPipeline() = default;

  virtual auto execute(RHI::CommandEncoder* encoder) noexcept -> void {
    pGraph->execute(encoder);
  }

  virtual auto getActiveGraphs() noexcept -> std::vector<Graph*> {
    return {pGraph};
  }

  virtual auto getOutput() noexcept -> GFX::Texture* {
    return pGraph->getOutput();
  }

  virtual auto build() noexcept -> void override { pGraph->build(); }

  Graph* pGraph;
};

#pragma region IMPL

inline auto BufferInfo::withSize(uint32_t _size) noexcept -> BufferInfo& {
  size = _size;
  return *this;
}
inline auto BufferInfo::withUsages(RHI::BufferUsagesFlags _usages) noexcept
    -> BufferInfo& {
  usages = _usages;
  return *this;
}
inline auto BufferInfo::withFlags(ResourceFlags _flags) noexcept
    -> BufferInfo& {
  flags = _flags;
  return *this;
}
inline auto BufferInfo::consume(ConsumeEntry const& entry) noexcept
    -> BufferInfo& {
  consumeHistories.emplace_back(entry);
  return *this;
}

inline auto TextureInfo::consume(ConsumeEntry const& _entry) noexcept
    -> TextureInfo& {
  ConsumeEntry entry = _entry;
  if (entry.type == ConsumeType::ColorAttachment) {
    entry.access |= uint32_t(RHI::AccessFlagBits::COLOR_ATTACHMENT_READ_BIT) |
                    uint32_t(RHI::AccessFlagBits::COLOR_ATTACHMENT_WRITE_BIT);
    entry.stages |= uint32_t(RHI::PipelineStages::COLOR_ATTACHMENT_OUTPUT_BIT);
    entry.layout = RHI::TextureLayout::COLOR_ATTACHMENT_OPTIMAL;
  } else if (entry.type == ConsumeType::DepthStencilAttachment) {
    entry.access |=
        uint32_t(RHI::AccessFlagBits::DEPTH_STENCIL_ATTACHMENT_READ_BIT) |
        uint32_t(RHI::AccessFlagBits::DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
    entry.stages |= uint32_t(RHI::PipelineStages::COLOR_ATTACHMENT_OUTPUT_BIT) |
                    uint32_t(RHI::PipelineStages::EARLY_FRAGMENT_TESTS_BIT) |
                    uint32_t(RHI::PipelineStages::LATE_FRAGMENT_TESTS_BIT);
    entry.layout = RHI::TextureLayout::DEPTH_ATTACHMENT_OPTIMAL;
  } else if (entry.type == ConsumeType::TextureBinding) {
    entry.access |= uint32_t(RHI::AccessFlagBits::SHADER_READ_BIT);
    entry.layout = RHI::TextureLayout::SHADER_READ_ONLY_OPTIMAL;
  } else if (entry.type == ConsumeType::StorageBinding) {
    entry.access |= uint32_t(RHI::AccessFlagBits::SHADER_READ_BIT) |
                    uint32_t(RHI::AccessFlagBits::SHADER_WRITE_BIT);
    entry.layout = RHI::TextureLayout::GENERAL;
  }
  consumeHistories.emplace_back(entry);
  return *this;
}
inline auto TextureInfo::setInfo(TextureInfo const& x) noexcept
    -> TextureInfo& {
  sizeDef = x.sizeDef;
  if (sizeDef == SizeDefine::Absolute)
    size.absolute = x.size.absolute;
  else if (sizeDef == SizeDefine::Relative)
    size.relative = x.size.relative;
  format = x.format;
  levels = x.levels;
  return *this;
}
inline auto TextureInfo::withSize(Math::ivec3 absolute) noexcept
    -> TextureInfo& {
  sizeDef = SizeDefine::Absolute;
  size.absolute = absolute;
  return *this;
}
inline auto TextureInfo::withSize(Math::vec3 relative) noexcept
    -> TextureInfo& {
  sizeDef = SizeDefine::Relative;
  size.relative = relative;
  return *this;
}
inline auto TextureInfo::withSizeRelative(std::string const& src,
                                          Math::vec3 relative) noexcept
    -> TextureInfo& {
  sizeDef = SizeDefine::RelativeToAnotherTex;
  sizeRefName = src;
  size.relative = relative;
  return *this;
}
inline auto TextureInfo::withLevels(uint32_t _levels) noexcept -> TextureInfo& {
  levels = _levels;
  return *this;
}
inline auto TextureInfo::withLayers(uint32_t _layers) noexcept -> TextureInfo& {
  layers = _layers;
  return *this;
}
inline auto TextureInfo::withSamples(uint32_t _samples) noexcept
    -> TextureInfo& {
  samples = _samples;
  return *this;
}
inline auto TextureInfo::withFormat(RHI::TextureFormat _format) noexcept
    -> TextureInfo& {
  format = _format;
  return *this;
}
inline auto TextureInfo::withStages(RHI::ShaderStagesFlags flags) noexcept
    -> TextureInfo& {
  sflags = flags;
  return *this;
}
inline auto TextureInfo::withUsages(RHI::TextureUsagesFlags _usages) noexcept
    -> TextureInfo& {
  usages = _usages;
  if (usages | uint32_t(RHI::TextureUsage::COLOR_ATTACHMENT)) {
    stages |=
        uint32_t(RHI::PipelineStages::COLOR_ATTACHMENT_OUTPUT_BIT) | sflags;
    access |= uint32_t(RHI::AccessFlagBits::COLOR_ATTACHMENT_READ_BIT) |
              uint32_t(RHI::AccessFlagBits::COLOR_ATTACHMENT_WRITE_BIT);
    laytout = RHI::TextureLayout::COLOR_ATTACHMENT_OPTIMAL;
  } else if (usages | uint32_t(RHI::TextureUsage::DEPTH_ATTACHMENT)) {
    stages |= uint32_t(RHI::PipelineStages::COLOR_ATTACHMENT_OUTPUT_BIT) |
              uint32_t(RHI::PipelineStages::EARLY_FRAGMENT_TESTS_BIT) |
              uint32_t(RHI::PipelineStages::EARLY_FRAGMENT_TESTS_BIT) | sflags;
    access |= uint32_t(RHI::AccessFlagBits::DEPTH_STENCIL_ATTACHMENT_READ_BIT) |
              uint32_t(RHI::AccessFlagBits::DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
    laytout = RHI::TextureLayout::DEPTH_ATTACHMENT_OPTIMAL;
  } else if (usages | uint32_t(RHI::TextureUsage::TEXTURE_BINDING)) {
    stages |= sflags;
    access |= uint32_t(RHI::AccessFlagBits::SHADER_READ_BIT);
    laytout = RHI::TextureLayout::SHADER_READ_ONLY_OPTIMAL;
  } else if (usages | uint32_t(RHI::TextureUsage::STORAGE_BINDING)) {
    stages |= sflags;
    access |= uint32_t(RHI::AccessFlagBits::SHADER_READ_BIT) |
              uint32_t(RHI::AccessFlagBits::SHADER_WRITE_BIT);
    laytout = RHI::TextureLayout::GENERAL;
  }
  return *this;
}
inline auto TextureInfo::withFlags(ResourceFlags _flags) noexcept
    -> TextureInfo& {
  flags = _flags;
  return *this;
}

inline auto ResourceInfo::isBuffer() noexcept -> BufferInfo& {
  type = Type::Buffer;
  info.buffer = BufferInfo{};
  return info.buffer;
}
inline auto ResourceInfo::isTexture() noexcept -> TextureInfo& {
  type = Type::Texture;
  info.texture = TextureInfo{};
  return info.texture;
}

inline auto Graph::getOutput() noexcept -> GFX::Texture* {
  auto const& iter_id = passNameList.find(output_pass);
  if (iter_id == passNameList.end()) return nullptr;
  size_t id = iter_id->second;
  auto const& iter_pass = passes.find(id);
  if (iter_pass == passes.end()) return nullptr;
  renderData.pass = iter_pass->second.get();
  return renderData.getTexture(output_resource);
}

inline auto Graph::getTextureResource(std::string const& _pass,
                                      std::string const& _output) noexcept
    -> GFX::Texture* {
  std::string pass = _pass, output = _output;
  decodeAlias(pass, output);
  auto const& iter_id = passNameList.find(pass);
  if (iter_id == passNameList.end()) return nullptr;
  size_t id = iter_id->second;
  auto const& iter_pass = passes.find(id);
  if (iter_pass == passes.end()) return nullptr;
  renderData.pass = iter_pass->second.get();
  return renderData.getTexture(output);
}

inline auto Graph::getBufferResource(std::string const& _pass,
                                     std::string const& _output) noexcept
    -> GFX::Buffer* {
  std::string pass = _pass, output = _output;
  decodeAlias(pass, output);
  auto const& iter_id = passNameList.find(pass);
  if (iter_id == passNameList.end()) return nullptr;
  size_t id = iter_id->second;
  auto const& iter_pass = passes.find(id);
  if (iter_pass == passes.end()) return nullptr;
  renderData.pass = iter_pass->second.get();
  return renderData.getBuffer(output);
}

SE_EXPORT auto collectAllShaders(GFX::SBTsDescriptor const& sbt) noexcept
    -> std::vector<GFX::ShaderModule*>;

struct TextureResourceVirtualMachine {
  struct TextureSubresourceRange {
    uint32_t level_beg;
    uint32_t level_end;
    uint32_t mip_beg;
    uint32_t mip_end;

    auto operator==(TextureSubresourceRange const& x) noexcept -> bool {
      return level_beg == x.level_beg && level_end == x.level_end &&
             mip_beg == x.mip_beg && mip_end == x.mip_end;
    }
  };

  struct TextureSubresourceState {
    RHI::PipelineStageFlags stageMask;
    RHI::AccessFlags access;
    RHI::TextureLayout layout;

    auto operator==(TextureSubresourceState const& x) noexcept -> bool {
      return stageMask == x.stageMask && access == x.access &&
             layout == x.layout;
    }
  };

  struct TextureSubresourceEntry {
    TextureSubresourceRange range;
    TextureSubresourceState state;
  };

  TextureResourceVirtualMachine(
      RHI::Texture* tex,
      std::vector<TextureResource::ConsumeHistory> const& cosumeHistories)
      : texture(tex) {
    bool depthBit = RHI::hasDepthBit(tex->format());
    bool stencilBit = RHI::hasStencilBit(tex->format());
    if (depthBit) aspects |= uint32_t(RHI::TextureAspect::DEPTH_BIT);
    if (stencilBit) aspects |= uint32_t(RHI::TextureAspect::STENCIL_BIT);
    if (!depthBit && !stencilBit)
      aspects |= uint32_t(RHI::TextureAspect::COLOR_BIT);

    // init state
    // TODO :: fix more complicated cases
    states.emplace_back(TextureSubresourceEntry{
        TextureSubresourceRange{0, tex->depthOrArrayLayers(), 0,
                                tex->mipLevelCount()},
        TextureSubresourceState{
            (uint32_t)RHI::PipelineStages::ALL_COMMANDS_BIT,
            (uint32_t)RHI::AccessFlagBits::MEMORY_READ_BIT |
                (uint32_t)RHI::AccessFlagBits::MEMORY_WRITE_BIT,
            RHI::TextureLayout::SHADER_READ_ONLY_OPTIMAL}});
    for (auto const& hentry : cosumeHistories) {
      for (auto const& subentry : hentry.entries) {
        updateSubresource(
            TextureResourceVirtualMachine::TextureSubresourceRange{
                subentry.level_beg, subentry.level_end, subentry.mip_beg,
                subentry.mip_end},
            TextureResourceVirtualMachine::TextureSubresourceState{
                subentry.stages, subentry.access, subentry.layout});
        if (subentry.type == TextureInfo::ConsumeType::ColorAttachment ||
            subentry.type == TextureInfo::ConsumeType::DepthStencilAttachment) {
          updateSubresource(
              TextureResourceVirtualMachine::TextureSubresourceRange{
                  subentry.level_beg, subentry.level_end, subentry.mip_beg,
                  subentry.mip_end},
              TextureResourceVirtualMachine::TextureSubresourceState{
                  subentry.stages, subentry.access,
                  RHI::TextureLayout::SHADER_READ_ONLY_OPTIMAL});
        }
      }
    }
  }

  auto valid(TextureSubresourceRange const& x) noexcept -> bool {
    return (x.level_beg < x.level_end && x.mip_beg < x.mip_end);
  }

  auto intersect(TextureSubresourceRange const& x,
                 TextureSubresourceRange const& y) noexcept
      -> std::optional<TextureSubresourceRange> {
    TextureSubresourceRange isect;
    isect.level_beg = std::max(x.level_beg, y.level_beg);
    isect.level_end = std::min(x.level_end, y.level_end);
    isect.mip_beg = std::max(x.mip_beg, y.mip_beg);
    isect.mip_end = std::min(x.mip_end, y.mip_end);
    if (valid(isect))
      return isect;
    else
      return std::nullopt;
  }

  auto merge(TextureSubresourceRange const& x,
             TextureSubresourceRange const& y) noexcept
      -> std::optional<TextureSubresourceRange> {
    if (x.level_beg == y.level_beg && x.level_end == y.level_end) {
      if (x.mip_beg == y.mip_end)
        return TextureSubresourceRange{x.level_beg, x.level_end, y.mip_beg,
                                       x.mip_end};
      else if (x.mip_end == y.mip_beg)
        return TextureSubresourceRange{x.level_beg, x.level_end, x.mip_beg,
                                       y.mip_end};
      else
        return std::nullopt;
    } else if (x.mip_beg == y.mip_beg && x.mip_end == y.mip_end) {
      if (x.level_beg == y.level_end)
        return TextureSubresourceRange{y.level_beg, x.level_end, x.mip_beg,
                                       x.mip_end};
      else if (x.level_end == y.level_beg)
        return TextureSubresourceRange{x.level_beg, y.level_end, x.mip_beg,
                                       x.mip_end};
      else
        return std::nullopt;
    } else
      return std::nullopt;
  }

  auto tryMerge() noexcept -> void {
    if (states.size() <= 1) return;
    while (true) {
      for (size_t i = 1; i <= states.size(); ++i) {
        if (i == states.size()) return;
        if (states[i].state == states[i - 1].state) {
          std::optional<TextureSubresourceRange> merged =
              merge(states[i - 1].range, states[i].range);
          if (merged.has_value()) {
            states[i - 1].range = merged.value();
            states.erase(states.begin() + i);
            break;
          }
        }
      }
    }
  }

  auto diff(TextureSubresourceRange const& x,
            TextureSubresourceRange const& y) noexcept
      -> std::vector<TextureSubresourceRange> {
    std::vector<TextureSubresourceRange> diffs;
    auto fn_subdivide_mip = [&](uint32_t level_beg, uint32_t level_end) {
      if (x.mip_beg == y.mip_beg && x.mip_end == y.mip_end) {
        // do nothing
      } else if (x.mip_beg == y.mip_beg) {
        diffs.emplace_back(TextureSubresourceRange{level_beg, level_end,
                                                   y.mip_end, x.mip_end});
      } else if (x.mip_end == y.mip_end) {
        diffs.emplace_back(TextureSubresourceRange{level_beg, level_end,
                                                   x.mip_beg, y.mip_beg});
      } else {
        diffs.emplace_back(TextureSubresourceRange{level_beg, level_end,
                                                   x.mip_beg, y.mip_beg});
        diffs.emplace_back(TextureSubresourceRange{level_beg, level_end,
                                                   y.mip_end, x.mip_end});
      }
    };
    if (x.level_beg == y.level_beg && x.level_end == y.level_end) {
      fn_subdivide_mip(x.level_beg, x.level_end);
    } else if (x.level_beg == y.level_beg) {
      diffs.emplace_back(TextureSubresourceRange{y.level_end, x.level_end,
                                                 x.mip_beg, x.mip_end});
      fn_subdivide_mip(y.level_beg, y.level_end);
    } else if (x.level_end == y.level_end) {
      diffs.emplace_back(TextureSubresourceRange{x.level_beg, y.level_beg,
                                                 x.mip_beg, x.mip_end});
      fn_subdivide_mip(y.level_beg, y.level_end);
    } else {
      diffs.emplace_back(TextureSubresourceRange{x.level_beg, y.level_beg,
                                                 x.mip_beg, x.mip_end});
      diffs.emplace_back(TextureSubresourceRange{y.level_end, x.level_end,
                                                 x.mip_beg, x.mip_end});
      fn_subdivide_mip(y.level_beg, y.level_end);
    }
    return diffs;
  }

  auto toBarrierDescriptor(TextureSubresourceRange const& range,
                           TextureSubresourceState const& prev,
                           TextureSubresourceState const& next) {
    RHI::BarrierDescriptor desc = RHI::BarrierDescriptor{
        prev.stageMask,
        next.stageMask,
        uint32_t(RHI::DependencyType::NONE),
        std::vector<RHI::MemoryBarrier*>{},
        std::vector<RHI::BufferMemoryBarrierDescriptor>{},
        std::vector<RHI::TextureMemoryBarrierDescriptor>{
            RHI::TextureMemoryBarrierDescriptor{
                texture,
                RHI::ImageSubresourceRange{
                    aspects, range.mip_beg, range.mip_end - range.mip_beg,
                    range.level_beg, range.level_end - range.level_beg},
                prev.access, next.access, prev.layout, next.layout}}};
    return desc;
  }

  auto updateSubresource(TextureSubresourceRange const& range,
                         TextureSubresourceState const& state) noexcept
      -> std::vector<RHI::BarrierDescriptor> {
    std::vector<RHI::BarrierDescriptor> barriers;

    std::vector<TextureSubresourceEntry> addedEntries;
    for (auto iter = states.begin();;) {
      if (iter == states.end()) break;
      if (iter->range == range) {
        barriers.emplace_back(toBarrierDescriptor(range, iter->state, state));
        iter->state = state;
        return barriers;
      }
      std::optional<TextureSubresourceRange> isect =
          intersect(iter->range, range);
      if (isect.has_value()) {
        TextureSubresourceRange const& isect_range = isect.value();
        barriers.emplace_back(
            toBarrierDescriptor(isect_range, iter->state, state));
        addedEntries.emplace_back(TextureSubresourceEntry{isect_range, state});
        std::vector<TextureSubresourceRange> diff_ranges =
            diff(iter->range, isect_range);
        for (auto const& drange : diff_ranges)
          addedEntries.emplace_back(
              TextureSubresourceEntry{drange, iter->state});
        iter = states.erase(iter);
      } else {
        iter++;
      }
    }
    states.insert(states.end(), addedEntries.begin(), addedEntries.end());
    tryMerge();
    return barriers;
  }

  RHI::Texture* texture;
  RHI::TextureAspectFlags aspects = 0;
  std::vector<TextureSubresourceEntry> states;
};

inline auto RenderData::getTexture(std::string const& name) const noexcept
    -> GFX::Texture* {
  if (pass->pReflection.inputResources.find(name) !=
      pass->pReflection.inputResources.end()) {
    return graph
        ->textureResources[pass->pReflection.inputResources[name]
                               .devirtualizeID]
        ->texture;
  }
  if (pass->pReflection.outputResources.find(name) !=
      pass->pReflection.outputResources.end()) {
    return graph
        ->textureResources[pass->pReflection.outputResources[name]
                               .devirtualizeID]
        ->texture;
  }
  if (pass->pReflection.inputOutputResources.find(name) !=
      pass->pReflection.inputOutputResources.end()) {
    return graph
        ->textureResources[pass->pReflection.inputOutputResources[name]
                               .devirtualizeID]
        ->texture;
  }
  if (pass->pReflection.internalResources.find(name) !=
      pass->pReflection.internalResources.end()) {
    return graph
        ->textureResources[pass->pReflection.internalResources[name]
                               .devirtualizeID]
        ->texture;
  }
}

inline auto RenderData::getBuffer(std::string const& name) const noexcept
    -> GFX::Buffer* {
  if (pass->pReflection.inputResources.find(name) !=
      pass->pReflection.inputResources.end()) {
    return graph
        ->bufferResources[pass->pReflection.inputResources[name].devirtualizeID]
        ->buffer;
  }
  if (pass->pReflection.outputResources.find(name) !=
      pass->pReflection.outputResources.end()) {
    return graph
        ->bufferResources[pass->pReflection.outputResources[name]
                              .devirtualizeID]
        ->buffer;
  }
  if (pass->pReflection.inputOutputResources.find(name) !=
      pass->pReflection.inputOutputResources.end()) {
    return graph
        ->bufferResources[pass->pReflection.inputOutputResources[name]
                              .devirtualizeID]
        ->buffer;
  }
  if (pass->pReflection.internalResources.find(name) !=
      pass->pReflection.internalResources.end()) {
    return graph
        ->bufferResources[pass->pReflection.internalResources[name]
                              .devirtualizeID]
        ->buffer;
  }
}

#pragma endregion

}  // namespace SIByL::RDG