module;
#include <array>
#include <memory>
#include <compare>
#include <typeinfo>
#include <filesystem>
#include <functional>
#include "../../../Application/Public/SE.Application.Config.h"
export module SE.SRenderer:ScreenSpacePass;
import SE.Core.Resource;
import SE.RHI;
import SE.GFX.Core;
import SE.GFX.RDG;
import SE.RDG;

namespace SIByL
{
	export struct ScreenSpacePass :public RDG::RenderPass {

		virtual auto reflect() noexcept -> RDG::PassReflection override {
			RDG::PassReflection reflector;
			return reflector;
		}

		virtual auto execute(RDG::RenderContext* context, RDG::RenderData const& renderData) noexcept -> void {
			// set render pass
			passEncoders[context->flightIdx] = context->cmdEncoder->beginRenderPass(renderPassDescriptor);
			passEncoders[context->flightIdx]->setPipeline(pipelines[context->flightIdx].get());

			//passEncoders[context->flightIdx]->setBindGroup(0, bindgroup[i].get());
			//passEncoders[context->flightIdx]->pushConstants(&src_size,
			//	(uint32_t)RHI::ShaderStages::FRAGMENT,
			//	0, sizeof(uint32_t));

			//src_size /= 2;

			//mipPassEncoders[i][index]->setViewport(0, 0, src_size, src_size, 0, 1);
			//mipPassEncoders[i][index]->setScissorRect(0, 0, src_size, src_size);

			passEncoders[context->flightIdx]->draw(3, 1, 0, 0);
			passEncoders[context->flightIdx]->end();
		}
	};
}