#include <se.rhi.hpp>
#include <se.gfx.hpp>
#include <se.rdg.hpp>

namespace se::cbt {
struct SIByL_API CreateCBTPass :public rdg::DummyPass {
  CreateCBTPass(int maxDepth, int initDepth);
  virtual auto reflect() noexcept -> rdg::PassReflection override;
  virtual auto execute(rdg::RenderContext* context, rdg::RenderData const& renderData) noexcept -> void;
  int maxDepth; int initDepth; bool initialized = false;
};

struct SIByL_API SumReductionFusedPass :public rdg::ComputePass {
  SumReductionFusedPass(int maxDepth);
  virtual auto reflect() noexcept -> rdg::PassReflection override;
  virtual auto execute(rdg::RenderContext* context, rdg::RenderData const& renderData) noexcept -> void;
  int maxDepth;
};

struct SIByL_API SumReductionOneLayerPass :public rdg::ComputePass {
  SumReductionOneLayerPass(int maxDepth);
  virtual auto reflect() noexcept -> rdg::PassReflection override;
  virtual auto execute(rdg::RenderContext* context, rdg::RenderData const& renderData) noexcept -> void;
  int maxDepth;
};

struct SIByL_API CBTSpatialTreeVisualizePass : public rdg::RenderPass {
  CBTSpatialTreeVisualizePass(size_t indirect_offset);
  virtual auto reflect() noexcept -> rdg::PassReflection;
  //virtual auto renderUI() noexcept -> void override;
  virtual auto execute(rdg::RenderContext* context,
    rdg::RenderData const& renderData) noexcept -> void;
  virtual auto renderUI() noexcept -> void override;
  struct PushConst {
    ivec2 resolution;
    float line_width = 1.f;
    int camera_index;
    int show_level;
  } pConst;
  size_t indirect_offset = 0;
};

// This pass is used for test only
struct SIByL_API TestCBTPass :public rdg::ComputePass {
  TestCBTPass(int maxDepth);
  virtual auto reflect() noexcept -> rdg::PassReflection override;
  virtual auto execute(rdg::RenderContext* context, rdg::RenderData const& renderData) noexcept -> void;
};
}