#pragma once
#include <SE.SRenderer.hpp>
#include <SE.Addon.VXGI.hpp>
#include <SE.Addon.RestirGI.hpp>
#include <SE.Addon.BitonicSort.hpp>

namespace SIByL::Addon::VXGuiding {
SE_EXPORT struct DITestSetting {
  float ex = 1., ey = 1., intensity = 1.;
  float padding0;
  Math::vec3 position = Math::vec3(0);
  Math::vec3 rotation = Math::vec3(0);
  float padding1;
  struct SharedData {
    RHI::BufferBinding DITestBinding;
  } shared;
};

SE_EXPORT struct alignas(64) TestDIUniform {
  Math::mat4 worldMat;
  Math::mat4 localMat;
  Math::vec2 pMin;
  Math::vec2 pMax;
  float intensity;
};

struct VXGuidingSetting {
  bool respawn = true;
  bool inject_first_vertex = false;
  bool use_average_irradiance = false;
};

SE_EXPORT struct VXGuiderClearPass : public RDG::ComputePass {
  VXGuiderClearPass(VXGI::VXGISetting* setting, VXGuidingSetting* vxguiding_setting);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  virtual auto renderUI() noexcept -> void override;
  VXGI::VoxelizeUniform gUniform;
  GFX::StructuredUniformBufferView<VXGI::VoxelizeUniform> uniformBuffer;
  VXGI::VXGISetting* const setting;
  VXGuidingSetting* vxguiding_setting;
};

SE_EXPORT struct VXGuiderGIPass : public RDG::RayTracingPass {
  VXGuiderGIPass(VXGI::VXGISetting* vx_set);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  virtual auto renderUI() noexcept -> void override;
  VXGI::VXGISetting* voxel_setting;
  int strategy = 6;
  int traverse_mode = 0;
  int mis_mode = 0;
  int visibility_mode = 0;
  bool second = true;
  int splitbar = 1280;
};


SE_EXPORT struct VXGuiderGIASVGFPass : public RDG::RayTracingPass {
  VXGuiderGIASVGFPass(VXGI::VXGISetting* vx_set);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  virtual auto renderUI() noexcept -> void override;
  VXGI::VXGISetting* voxel_setting;
  int strategy = 6;
  int traverse_mode = 0;
  int mis_mode = 0;
  int visibility_mode = 0;
  bool second = true;
};

SE_EXPORT struct VXGuiderReSTIRGIPass : public RDG::RayTracingPass {
  VXGuiderReSTIRGIPass(VXGI::VXGISetting* vx_set,
                       RestirGI::GIResamplingRuntimeParameters* param);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  virtual auto renderUI() noexcept -> void override;
  VXGI::VXGISetting* voxel_setting;
  int strategy = 0;
  int traverse_mode = 0;
  int mis_mode = 0;
  int visibility_mode = 0;
  RestirGI::GIResamplingRuntimeParameters* param;
};

SE_EXPORT struct VXGuiderAccumPass : public RDG::RayTracingPass {
  VXGuiderAccumPass(VXGI::VXGISetting* vx_set);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  virtual auto renderUI() noexcept -> void override;
  virtual auto onInteraction(Platform::Input* input,
                             Editor::Widget::WidgetInfo* info) noexcept
      -> void override;
  VXGI::VXGISetting* voxel_setting;
  int strategy = 0;
  int traverse_mode = 0;
  int mis_mode = 0;
  int visibility_mode = 0;
  Math::ivec2 debugPixel;
};

SE_EXPORT struct VXGuiderCompactPass : public RDG::ComputePass {
  VXGuiderCompactPass(VXGI::VXGISetting* setting);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  VXGI::VXGISetting* const setting;
};

SE_EXPORT struct VXGuiderGeometryPass : public RDG::RenderPass {
  VXGuiderGeometryPass(VXGI::VXGISetting* setting);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  virtual auto renderUI() noexcept -> void override;
  VXGI::VXGISetting* const setting;
  struct alignas(64) VoxelSetting {
    int use_compact = 0;
    int z_conservative = 0;
    int clipping = 0;
  } gVoxelSetting;
  GFX::StructuredUniformBufferView<VoxelSetting> settingBuffer;
  bool injectGeometries = false;
};

SE_EXPORT struct VXGuiderGeometryDynamicPass : public RDG::RenderPass {
  VXGuiderGeometryDynamicPass(VXGI::VXGISetting* setting);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  virtual auto renderUI() noexcept -> void override;
  VXGI::VXGISetting* const setting;
  struct alignas(64) VoxelSetting {
    int use_compact = 0;
    int z_conservative = 0;
    int clipping = 0;
  } gVoxelSetting;
  GFX::StructuredUniformBufferView<VoxelSetting> settingBuffer;
  bool injectGeometries = false;
  int inject_id = 0;
};

SE_EXPORT struct VXGuiderBakeCleanPass : public RDG::ComputePass {
  VXGuiderBakeCleanPass(VXGI::VXGISetting* setting);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  VXGI::VXGISetting* const setting;
};

SE_EXPORT struct VXGuiderGeometryBakePass : public RDG::RenderPass {
  VXGuiderGeometryBakePass(VXGI::VXGISetting* setting);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  virtual auto renderUI() noexcept -> void override;
  VXGI::VXGISetting* const setting;
  struct alignas(64) VoxelSetting {
    int use_compact = 1;
    int z_conservative = 0;
    int clipping = 1;
  } gVoxelSetting;
  GFX::StructuredUniformBufferView<VoxelSetting> settingBuffer;
  GFX::StructuredUniformBufferView<VXGI::VoxelizeUniform> uniformBuffer;
};

SE_EXPORT struct PrebakeDummyPass : public RDG::DummyPass {
  PrebakeDummyPass(bool& need_rebake) : need_rebake(need_rebake) { RDG::Pass::init(); }
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto renderUI() noexcept -> void override;
  bool& need_rebake;
};

SE_EXPORT struct VXGuiderVisualizePass : public RDG::RenderPass {
  VXGuiderVisualizePass(VXGI::VXGISetting* setting);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  virtual auto renderUI() noexcept -> void override;

  struct PushConstantBuffer {
    Math::vec2 resolution;
    float line_width = 5.f;
  } pConst;

  VXGI::VXGISetting* setting;
  bool visualize = true;
};

SE_EXPORT struct VXGuiderDIInjection : public RDG::RenderPass {
  VXGuiderDIInjection(DITestSetting* test_set, VXGI::VXGISetting* setting);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  virtual auto renderUI() noexcept -> void override;
  DITestSetting* di_test_setting;
  VXGI::VXGISetting* voxel_setting;
  TestDIUniform gTestDI;
  GFX::StructuredUniformBufferView<TestDIUniform> testdiBuffer;
  struct alignas(64) VoxelSetting {
    int use_compact = 0;
    int z_conservative = 0;
    int clipping = 0;
  } gVoxelSetting;
  GFX::StructuredUniformBufferView<VoxelSetting> settingBuffer;
  VXGI::VoxelizeUniform gUniform;
  GFX::StructuredUniformBufferView<VXGI::VoxelizeUniform> uniformBuffer;
};

SE_EXPORT struct VXGuider1stBounceInjection : public RDG::RayTracingPass {
  VXGuider1stBounceInjection(VXGI::VXGISetting* setting,
                             VXGuidingSetting* vxguiding_setting);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  virtual auto renderUI() noexcept -> void override;
  VXGI::VXGISetting* voxel_setting;
  VXGuidingSetting* vxguiding_setting;
};

SE_EXPORT struct VXGuiderLightInjection : public RDG::ComputePass {
  VXGuiderLightInjection(VXGI::VXGISetting* setting,
                         VXGuidingSetting* vxguiding_setting);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  virtual auto renderUI() noexcept -> void override;
  VXGI::VXGISetting* voxel_setting;
  VXGuidingSetting* vxguiding_setting;
};

SE_EXPORT struct VoxelClear6DPass : public RDG::ComputePass {
  VoxelClear6DPass(VXGI::VXGISetting* setting);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  VXGI::VXGISetting* const setting;
};

SE_EXPORT struct VXGuiderViewPass : public RDG::FullScreenPass {
  VXGuiderViewPass(VXGI::VXGISetting* setting);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  virtual auto renderUI() noexcept -> void override;
  virtual auto onInteraction(Platform::Input* input,
                             Editor::Widget::WidgetInfo* info) noexcept
      -> void override;
  VXGI::VXGISetting* setting;
  struct PushConstant {
    Math::ivec2 resolution;
    float edgeThreshold = 0.02f;
    int showWhat = 0;
    Math::ivec2 mapSize = {40, 23};
    Math::ivec2 debugPixel = {0, 0};
    float scalar = 1.f;
  } pConst;
};

SE_EXPORT struct VoxelMip6DPass : public RDG::ComputePass {
  VoxelMip6DPass(VXGI::VXGISetting* setting, bool secondPass);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  virtual auto renderUI() noexcept -> void override;
  VXGI::VXGISetting* const setting;
  bool updateMip = true;
  bool saveNumericalError = true;
  bool secondPass = false;
};

SE_EXPORT struct Voxel6DRTInjection : public RDG::RayTracingPass {
  Voxel6DRTInjection(VXGI::VXGISetting* vx_set);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  VXGI::VXGISetting* voxel_setting;
};

SE_EXPORT struct Voxel6DVisualizePass : public RDG::FullScreenPass {
  Voxel6DVisualizePass(VXGI::VXGISetting* setting, bool isotropic = false);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  virtual auto renderUI() noexcept -> void override;
  VXGI::VXGISetting* setting;
  bool isotropic;
  struct PushConstant {
    Math::ivec2 resolution;
    float edgeThreshold = 0.02f;
    int mipLevelShown = 0;
    Math::ivec2 padding;
    int showWhat = 0;
    float scaling = 1.f;
    int useSampler = 0;
  } pConst;
};

SE_EXPORT struct DITestInjectPass : public RDG::RenderPass {
  DITestInjectPass(DITestSetting* test_set, VXGI::VXGISetting* setting);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  virtual auto renderUI() noexcept -> void override;
  DITestSetting* di_test_setting;
  VXGI::VXGISetting* voxel_setting;
  TestDIUniform gUniform;
  GFX::StructuredUniformBufferView<TestDIUniform> uniformBuffer;
};

SE_EXPORT struct DITestVoxelCheckPass : public RDG::FullScreenPass {
  DITestVoxelCheckPass(DITestSetting* test_set, VXGI::VXGISetting* setting);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  virtual auto renderUI() noexcept -> void override;
  DITestSetting* di_test_setting;
  VXGI::VXGISetting* setting;
  struct PushConstant {
    Math::ivec2 resolution;
    float edgeThreshold = 0.02f;
    int mipLevelShown = 0;
    Math::ivec2 padding;
    int showWhat = 0;
    float scaling = 1.f;
  } pConst;
};

SE_EXPORT struct DITestPass : public RDG::RayTracingPass {
  DITestPass(DITestSetting* test_set, VXGI::VXGISetting* vx_set);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto renderUI() noexcept -> void override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  virtual auto onInteraction(Platform::Input* input,
                             Editor::Widget::WidgetInfo* info) noexcept
      -> void override;
  DITestSetting* di_test_setting;
  VXGI::VXGISetting* voxel_setting;
  int strategy = 0;
};

SE_EXPORT struct GITestPass : public RDG::RayTracingPass {
  GITestPass(VXGI::VXGISetting* vx_set);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto renderUI() noexcept -> void override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  VXGI::VXGISetting* voxel_setting;
  int strategy = 0;
};

SE_EXPORT struct ImportInjectPass : public RDG::RayTracingPass {
  ImportInjectPass(VXGI::VXGISetting* vx_set);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto renderUI() noexcept -> void override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  virtual auto onInteraction(Platform::Input* input,
                             Editor::Widget::WidgetInfo* info) noexcept
      -> void override;

  // virtual auto onInteraction(Platform::Input* input,
  //                            Editor::Widget::WidgetInfo* info) noexcept
  //     -> void override;
  VXGI::VXGISetting* voxel_setting;
  Math::ivec2 debug_pixel;
  float scaling = 1;
  bool useOcclusion = 1.f;
};

SE_EXPORT struct VXClusterComputeInfoPass : public RDG::ComputePass {
  VXClusterComputeInfoPass(VXGI::VXGISetting* vx_set);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  VXGI::VXGISetting* voxel_setting;
};

SE_EXPORT struct VXClusterInitCenterPass : public RDG::ComputePass {
  VXClusterInitCenterPass(VXGI::VXGISetting* vx_set);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  virtual auto renderUI() noexcept -> void override;
  bool use_seed = false;
  VXGI::VXGISetting* voxel_setting;
};


SE_EXPORT struct VXClusterSeedingPass : public RDG::ComputePass {
  VXClusterSeedingPass(VXGI::VXGISetting* vx_set);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  VXGI::VXGISetting* voxel_setting;
};

SE_EXPORT struct VXClusterFindAssociatePass : public RDG::ComputePass {
  VXClusterFindAssociatePass(VXGI::VXGISetting* vx_set);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  VXGI::VXGISetting* voxel_setting;
  virtual auto renderUI() noexcept -> void override;
  bool noCluster = false;
};

SE_EXPORT struct VXClusterUpdateCenterPass : public RDG::ComputePass {
  VXClusterUpdateCenterPass(VXGI::VXGISetting* vx_set);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  VXGI::VXGISetting* voxel_setting;
};

SE_EXPORT struct VXTreeEncodePass : public RDG::ComputePass {
  VXTreeEncodePass(VXGI::VXGISetting* vx_set);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  VXGI::VXGISetting* voxel_setting;
};

SE_EXPORT struct VXTreeIIntializePass : public RDG::ComputePass {
  VXTreeIIntializePass(VXGI::VXGISetting* vx_set);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  VXGI::VXGISetting* voxel_setting;
};

SE_EXPORT struct VXTreeInternalPass : public RDG::ComputePass {
  VXTreeInternalPass();
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
};

SE_EXPORT struct VXTreeMergePass : public RDG::ComputePass {
  VXTreeMergePass();
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
};

SE_EXPORT struct VXTreeTopLevelPass : public RDG::ComputePass {
  VXTreeTopLevelPass();
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  virtual auto renderUI() noexcept -> void override;
  int visibility = 1;
};

SE_EXPORT struct SPixelClearPass : public RDG::ComputePass {
  SPixelClearPass();
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
};

SE_EXPORT struct SPixelGatherPass : public RDG::ComputePass {
  SPixelGatherPass(VXGI::VXGISetting* voxel_setting);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  VXGI::VXGISetting* voxel_setting;
};

SE_EXPORT struct VXInfoClearPass : public RDG::ComputePass {
  VXInfoClearPass(VXGI::VXGISetting* voxel_setting);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  VXGI::VXGISetting* voxel_setting;
};

SE_EXPORT struct RowColumnPresamplePass : public RDG::ComputePass {
  RowColumnPresamplePass(VXGI::VXGISetting* voxel_setting);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  VXGI::VXGISetting* voxel_setting;
};

SE_EXPORT struct RowVisibilityPass : public RDG::ComputePass {
  RowVisibilityPass();
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
};

SE_EXPORT struct RowKmppCenterPass : public RDG::ComputePass {
  RowKmppCenterPass(VXGI::VXGISetting* setting);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  VXGI::VXGISetting* setting;
};

SE_EXPORT struct RowFindCenterPass : public RDG::ComputePass {
  RowFindCenterPass(VXGI::VXGISetting* setting);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  VXGI::VXGISetting* setting;
};

SE_EXPORT struct VXInfoRearrangePass : public RDG::ComputePass {
  VXInfoRearrangePass();
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  virtual auto renderUI() noexcept -> void override;
  int threshold_use_vpl_bound = -1;
};

SE_EXPORT struct SPixelVisibilityPass : public RDG::RayTracingPass {
  SPixelVisibilityPass();
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  virtual auto renderUI() noexcept -> void override;
  bool do_execute = true;
};

SE_EXPORT struct SPixelVisibilityEXPass : public RDG::ComputePass {
  SPixelVisibilityEXPass();
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  virtual auto renderUI() noexcept -> void override;
  bool do_execute = true;
  bool use_bsdf = true;
  bool use_distance = false;
};

SE_EXPORT struct VPLVisualizePass : public RDG::RenderPass {
  VPLVisualizePass();
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  virtual auto renderUI() noexcept -> void override;

  struct PushConstantBuffer {
    float gVPLRenderScale = 0.05f;
  } pConst;
};

SE_EXPORT struct GeomIJConfig {
  struct alignas(64) DataPack {
    Math::vec3 aabbMin;      // min of world AABB
    uint32_t size = 64;      // length of one dim over world AABB
    Math::vec3 aabbMax;      // max of world AABB
    int HashmapSize = 65536; // hash map size
    enum struct Mode : uint32_t { Hash, Unroll } mode = Mode::Hash;
    int bucket_mode = 0;
    int bucket_size = 8;
    int use_compact = 0;
    int z_conservative = 0;
    int clipping = 0;
  } data;
  GFX::StructuredUniformBufferView<DataPack> ubo = {nullptr};
  // get the cell count under this config
  uint32_t get_cell_count() const noexcept;
};

SE_EXPORT struct GeometryBakeClearPass : public RDG::ComputePass {
  GeometryBakeClearPass(GeomIJConfig* config);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept -> void override;
  GeomIJConfig* config;
};

SE_EXPORT struct GeometryBakePass : public RDG::RenderPass {
  GeometryBakePass(GeomIJConfig* config);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept -> void override;
  GeomIJConfig* const config;
};

SE_EXPORT struct GeometryVisualizePass : public RDG::RenderPass {
  GeometryVisualizePass(GeomIJConfig* config);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto renderUI() noexcept -> void override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept -> void override;
  struct PushConstantBuffer {
    Math::vec2 resolution;
    float line_width = 5.f;
  } pConst;
  bool visualize = true;
  GeomIJConfig* const config;
};

SE_EXPORT struct GeometryVisualize2Pass : public RDG::RenderPass {
  GeometryVisualize2Pass(GeomIJConfig* config);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto renderUI() noexcept -> void override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept -> void override;
  struct PushConstantBuffer {
    int play_mode = 0;
  } pConst;
  bool visualize = true;
  GeomIJConfig* const config;
};

SE_EXPORT struct DataPreparePass : public RDG::ComputePass {
  DataPreparePass(GeomIJConfig* config);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  GeomIJConfig* const config;
};

SE_EXPORT struct LightIJPass : public RDG::RayTracingPass {
  LightIJPass(GeomIJConfig* config);
  virtual auto renderUI() noexcept -> void override;
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  GeomIJConfig* const config;
};

SE_EXPORT struct GeometryReloadPass : public RDG::ComputePass {
  GeometryReloadPass(GeomIJConfig* config);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  GeomIJConfig* const config;
};

SE_EXPORT struct GeometryIJPass : public RDG::RenderPass {
  GeometryIJPass(GeomIJConfig* config);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto renderUI() noexcept -> void override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept -> void override;
  GeomIJConfig* const config;
  int lod_level = 0;
};

SE_EXPORT struct VoxelCompactPass : public RDG::ComputePass {
  VoxelCompactPass(GeomIJConfig* config);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept -> void override;
  GeomIJConfig* const config;
};

SE_EXPORT struct SPixelCenterPass : public RDG::ComputePass {
  SPixelCenterPass();
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto renderUI() noexcept -> void override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept -> void override;
};

SE_EXPORT struct SPixelFindPass : public RDG::ComputePass {
  SPixelFindPass();
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept -> void override;
};

SE_EXPORT struct SPixelVisualizePass : public RDG::ComputePass {
  SPixelVisualizePass();
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept -> void override;
  virtual auto renderUI() noexcept -> void override;
  bool drawBoundary = true;
  int debug_mode = 0;
};
//
//SE_EXPORT struct RowColumnPresamplePass : public RDG::ComputePass {
//  RowColumnPresamplePass(VXGI::VXGISetting* voxel_setting);
//  virtual auto reflect() noexcept -> RDG::PassReflection override;
//  virtual auto execute(RDG::RenderContext* context,
//                       RDG::RenderData const& renderData) noexcept
//      -> void override;
//  VXGI::VXGISetting* voxel_setting;
//};

SE_EXPORT struct SVoxelRowPresamplePass : public RDG::ComputePass {
  SVoxelRowPresamplePass();
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept -> void override;
};

SE_EXPORT struct SVoxelColumnPresamplePass : public RDG::ComputePass {
  SVoxelColumnPresamplePass();
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept -> void override;
};

SE_EXPORT struct SVoxelTreeEncodePass : public RDG::ComputePass {
  SVoxelTreeEncodePass(GeomIJConfig* config);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept -> void override;
  GeomIJConfig* config;
};

SE_EXPORT struct SVoxelTreeIIntializePass : public RDG::ComputePass {
  SVoxelTreeIIntializePass(GeomIJConfig* config);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept -> void override;
  GeomIJConfig* config;
};

SE_EXPORT struct SVoxelInfoClearPass : public RDG::ComputePass {
  SVoxelInfoClearPass(GeomIJConfig* config);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  GeomIJConfig* config;
};

SE_EXPORT struct SVoxelRowColumnPresamplePass : public RDG::ComputePass {
  SVoxelRowColumnPresamplePass(GeomIJConfig* config);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  GeomIJConfig* config;
};

SE_EXPORT struct SPixelInfoGather : public RDG::ComputePass {
  SPixelInfoGather(GeomIJConfig* config);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  GeomIJConfig* config;
};

SE_EXPORT struct VXPGuidingPass : public RDG::RayTracingPass {
  VXPGuidingPass(GeomIJConfig* config);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  virtual auto renderUI() noexcept -> void override;
  GeomIJConfig* config;
  int strategy = 6;
  int traverse_mode = 0;
  int mis_mode = 0;
  int visibility_mode = 0;
  bool second = true;
  int splitbar = 1280;
};

SE_EXPORT struct GeometryPrebakePipeline : public RDG::Pipeline {
  GeometryPrebakePipeline();
  virtual auto execute(RHI::CommandEncoder* encoder) noexcept -> void override;
  virtual auto renderUI() noexcept -> void override;
  virtual auto getActiveGraphs() noexcept -> std::vector<RDG::Graph*> override;
  virtual auto getOutput() noexcept -> GFX::Texture* override;
  virtual auto build() noexcept -> void override;

  GeomIJConfig config;
  struct GeometryPrebakeGraph : public RDG::Graph {
    GeometryPrebakeGraph(GeomIJConfig* config); };
  struct VXPGGraph : public RDG::Graph {
    VXPGGraph(GeomIJConfig* config);
    Addon::BitonicSort::BitonicSortSetting sort_setting; };
  std::unique_ptr<GeometryPrebakeGraph> prebake_graph;
  std::unique_ptr<VXPGGraph> vxpg_graph;
  int graph_chosen = 0;
};
}