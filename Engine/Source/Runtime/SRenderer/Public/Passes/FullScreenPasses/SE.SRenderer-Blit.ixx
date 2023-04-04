module;
#include <array>
#include <memory>
#include <compare>
#include <typeinfo>
#include <filesystem>
#include <functional>
#include "../../../../Application/Public/SE.Application.Config.h"
export module SE.SRenderer.BlitPass;
import SE.Math.Geometric;
import SE.Core.Resource;
import SE.RHI;
import SE.GFX;
import SE.RDG;

namespace SIByL
{
	export struct BlitPass :public RDG::FullScreenPass {

		struct Descriptor {
			RDG::TextureInfo targetInfo;
		} desc;

		BlitPass(Descriptor const desc) :desc(desc) {
			frag = GFX::GFXManager::get()->registerShaderModuleResource(
				"../Engine/Binaries/Runtime/spirv/SRenderer/rasterizer/fullscreen_pass/blit_image_frag.spv",
				{ nullptr, RHI::ShaderStages::FRAGMENT });
			RDG::FullScreenPass::init(Core::ResourceManager::get()->getResource<GFX::ShaderModule>(frag));
		}

		virtual auto reflect() noexcept -> RDG::PassReflection {
			RDG::PassReflection reflector;

			reflector.addInput("Input")
				.isTexture()
				.withUsages((uint32_t)RHI::TextureUsage::TEXTURE_BINDING)
				.consume(RDG::TextureInfo::ConsumeEntry{ RDG::TextureInfo::ConsumeType::TextureBinding }
					.addStage((uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT));

			reflector.addOutput("Output")
				.isTexture()
				.setInfo(desc.targetInfo)
				.withUsages((uint32_t)RHI::TextureUsage::COLOR_ATTACHMENT)
				.consume(RDG::TextureInfo::ConsumeEntry{ RDG::TextureInfo::ConsumeType::ColorAttachment }
					.enableDepthWrite(false)
					.setAttachmentLoc(0)
					.setDepthCompareFn(RHI::CompareFunction::ALWAYS));

			return reflector;
		}

		virtual auto execute(RDG::RenderContext* context, RDG::RenderData const& renderData) noexcept -> void {

			GFX::Texture* in = renderData.getTexture("Input");
			GFX::Texture* out = renderData.getTexture("Output");

			renderPassDescriptor = {
				{ RHI::RenderPassColorAttachment{
					out->getRTV(0, 0, 1),
					nullptr, {0,0,0,1}, RHI::LoadOp::CLEAR, RHI::StoreOp::STORE }},
				RHI::RenderPassDepthStencilAttachment{},
			};

			RHI::RenderPassEncoder* encoder = beginPass(context, out);

			getBindGroup(context, 0)->updateBinding(std::vector<RHI::BindGroupEntry>{
				RHI::BindGroupEntry{ 0,RHI::BindingResource(
					std::vector<RHI::TextureView*>{in->getSRV(0,1,0,1)},
					Core::ResourceManager::get()->getResource<GFX::Sampler>(GFX::GFXManager::get()->commonSampler.defaultSampler)->sampler.get())
				}
			});

			dispatchFullScreen(context);

			encoder->end();
		}

		Core::GUID vert, frag;
	};
}