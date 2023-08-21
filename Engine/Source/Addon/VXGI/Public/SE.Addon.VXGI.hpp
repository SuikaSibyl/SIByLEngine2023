#pragma once
#include <SE.SRenderer.hpp>

namespace SIByL::Addon::VXGI {
SE_EXPORT struct VXGISetting {
  enum struct DataStructure {
    Clipmap,  // Only support Clipmap for now
  } dataStructure = DataStructure::Clipmap;
  struct ClipmapSetting {
    uint32_t size = 64;  // length of one dim
    uint32_t lod = 3;    // number of lods
    uint32_t mip = 4;
  } clipmapSetting;
  struct SharedData {
    RHI::BufferBinding voxUniBinding;
    GFX::Sampler* tex3dSampler;
  } shared;
};

struct alignas(64) VoxelizeUniform {
  Math::vec3 aabbMin;
  int voxelSize;
  Math::vec3 aabbMax;
  int padding;
};

SE_EXPORT struct VoxelizePass : public RDG::RenderPass {
  VoxelizePass(VXGISetting* setting);
  virtual auto reflect() noexcept
      -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  VXGISetting* const setting;
  VoxelizeUniform gUniform;
  GFX::StructuredUniformBufferView<VoxelizeUniform> uniformBuffer;
};

SE_EXPORT struct VoxelMipPass : public RDG::ComputePass {
  VoxelMipPass(VXGISetting* setting);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  virtual auto renderUI() noexcept -> void override;
  VXGISetting* const setting;
  bool updateMip = true;
};

SE_EXPORT struct VoxelVisualizePass : public RDG::FullScreenPass {
  VoxelVisualizePass(VXGISetting* setting);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  virtual auto renderUI() noexcept -> void override;
  VXGISetting* setting;
  struct PushConstant {
    Math::ivec2 resolution;
    float edgeThreshold = 0.02f;
    int mipLevelShown = 0;
  } pConst;
};

SE_EXPORT struct ConeTraceDebuggerPass : public RDG::RayTracingPass {
  ConeTraceDebuggerPass(VXGISetting* setting);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto renderUI() noexcept -> void override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  virtual auto onInteraction(Platform::Input* input,
                             Editor::Widget::WidgetInfo* info) noexcept
      -> void override;
  VXGISetting* setting;
  Math::ivec2 debugPixel;
  float coneLength = 1.f;
  int invalidDebugPixel = 0;
  int voxelShown = 0;
  Math::ivec4 debugVoxel;
  int debug_mode = 0;
  float maximum_tan = 0.57735026919f;
};
}