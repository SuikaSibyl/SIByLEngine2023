#include <filesystem>
#include <typeinfo>
#include <cstdint>
#include <memory>
#include "../../../../../Application/Public/SE.Application.Config.h"
#include <Resource/SE.Core.Resource.hpp>
#include <SE.RHI.hpp>
#include <SE.GFX.hpp>
#include <SE.RDG.hpp>
#include <SE.SRenderer.hpp>

namespace SIByL::SRP
{
	SE_EXPORT struct BarGTPass :public RDG::RayTracingPass {

		struct PushConstant {
			uint32_t width;
			uint32_t height;
			uint32_t sample_batch;
			uint32_t lightIndex;
		};

		BarGTPass();

		virtual auto reflect() noexcept -> RDG::PassReflection override;

		virtual auto execute(RDG::RenderContext* context,
                                     RDG::RenderData const& renderData) noexcept
                    -> void override;

		Core::GUID udpt_rgen;
	};

	SE_EXPORT struct BarGTGraph :public RDG::Graph {
		BarGTGraph();
	};

	SE_EXPORT struct BarGTPipeline : public RDG::SingleGraphPipeline {
		BarGTPipeline() { pGraph = &graph; }
		BarGTGraph graph;
	};
}