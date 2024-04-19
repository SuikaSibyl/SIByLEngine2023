#include <se.rhi.hpp>
#include <se.gfx.hpp>
#include <se.rdg.hpp>

namespace se {
  struct SIByL_API AccumulatePass :public rdg::ComputePass {
	AccumulatePass(ivec3 resolution);
	virtual auto reflect() noexcept -> rdg::PassReflection override;
	virtual auto renderUI() noexcept -> void override;
	virtual auto execute(rdg::RenderContext* context, rdg::RenderData const& renderData) noexcept -> void;

	struct PushConstant {
	  se::uvec2 resolution;
	  uint32_t gAccumCount;
	  uint32_t gAccumulate = 0;
	  uint32_t gMovingAverageMode;
	} pConst;
	int maxAccumCount = 5;
	ivec3 resolution;
  };
}