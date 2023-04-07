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
			uint32_t src_mip;
			uint32_t src_array;
			uint32_t dst_mip;
			uint32_t dst_array;
		} desc;

		BlitPass(Descriptor const& desc) :desc(desc) {
			frag = GFX::GFXManager::get()->registerShaderModuleResource(
				"../Engine/Binaries/Runtime/spirv/SRenderer/rasterizer/fullscreen_pass/blit_image_frag.spv",
				{ nullptr, RHI::ShaderStages::FRAGMENT });
			RDG::FullScreenPass::init(Core::ResourceManager::get()->getResource<GFX::ShaderModule>(frag));
		}

		virtual auto reflect() noexcept -> RDG::PassReflection {
			RDG::PassReflection reflector;

			reflector.addInput("Source")
				.isTexture()
				.withUsages((uint32_t)RHI::TextureUsage::TEXTURE_BINDING)
				.consume(RDG::TextureInfo::ConsumeEntry{ RDG::TextureInfo::ConsumeType::TextureBinding }
					.setSubresource(desc.src_mip, desc.src_mip + 1, desc.src_array, desc.src_array + 1)
					.addStage((uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT));

			reflector.addInputOutput("Target")
				.isTexture()
				.withUsages((uint32_t)RHI::TextureUsage::COLOR_ATTACHMENT)
				.consume(RDG::TextureInfo::ConsumeEntry{ RDG::TextureInfo::ConsumeType::ColorAttachment }
					.setSubresource(desc.dst_mip, desc.dst_mip + 1, desc.dst_array, desc.dst_array + 1)
					.enableDepthWrite(false)
					.setAttachmentLoc(0)
					.setDepthCompareFn(RHI::CompareFunction::ALWAYS));

			return reflector;
		}

		virtual auto execute(RDG::RenderContext* context, RDG::RenderData const& renderData) noexcept -> void {

			GFX::Texture* in = renderData.getTexture("Source");
			GFX::Texture* out = renderData.getTexture("Target");

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