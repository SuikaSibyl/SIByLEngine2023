module;
#include <array>
#include <memory>
#include <compare>
#include <typeinfo>
#include <filesystem>
#include <functional>
#include "../../../../Application/Public/SE.Application.Config.h"
export module SE.SRenderer.ClearI32RGBA;
import SE.Math.Geometric;
import SE.Core.Resource;
import SE.RHI;
import SE.GFX;
import SE.RDG;

namespace SIByL
{
	export struct ClearI32RGBAPass :public RDG::FullScreenPass {

		ClearI32RGBAPass() {
			frag = GFX::GFXManager::get()->registerShaderModuleResource(
				"../Engine/Binaries/Runtime/spirv/SRenderer/rasterizer/fullscreen_pass/clear_i32a4_frag.spv",
				{ nullptr, RHI::ShaderStages::FRAGMENT });
			RDG::FullScreenPass::init(Core::ResourceManager::get()->getResource<GFX::ShaderModule>(frag));
		}

		virtual auto reflect() noexcept -> RDG::PassReflection {
			RDG::PassReflection reflector;

			reflector.addOutput("I32RGBA")
				.isTexture()
				.withSize(Math::vec3(1, 1, 1))
				.withFormat(RHI::TextureFormat::R32_FLOAT)
				.withUsages((uint32_t)RHI::TextureUsage::COLOR_ATTACHMENT | (uint32_t)RHI::TextureUsage::STORAGE_BINDING)
				.withLayers(4)
				.consume(RDG::TextureInfo::ConsumeEntry{ RDG::TextureInfo::ConsumeType::ColorAttachment }
					.setSubresource(0,1,0,1).setAttachmentLoc(0)
					.addStage((uint32_t)RHI::PipelineStages::COLOR_ATTACHMENT_OUTPUT_BIT))
				.consume(RDG::TextureInfo::ConsumeEntry{ RDG::TextureInfo::ConsumeType::ColorAttachment }
					.setSubresource(0, 1, 1, 2).setAttachmentLoc(1)
					.addStage((uint32_t)RHI::PipelineStages::COLOR_ATTACHMENT_OUTPUT_BIT))
				.consume(RDG::TextureInfo::ConsumeEntry{ RDG::TextureInfo::ConsumeType::ColorAttachment }
					.setSubresource(0, 1, 2, 3).setAttachmentLoc(2)
					.addStage((uint32_t)RHI::PipelineStages::COLOR_ATTACHMENT_OUTPUT_BIT))
				.consume(RDG::TextureInfo::ConsumeEntry{ RDG::TextureInfo::ConsumeType::ColorAttachment }
					.setSubresource(0, 1, 3, 4).setAttachmentLoc(3)
					.addStage((uint32_t)RHI::PipelineStages::COLOR_ATTACHMENT_OUTPUT_BIT));

			return reflector;
		}

		virtual auto execute(RDG::RenderContext* context, RDG::RenderData const& renderData) noexcept -> void {

			GFX::Texture* target = renderData.getTexture("I32RGBA");

			renderPassDescriptor = {
				{ RHI::RenderPassColorAttachment{target->getRTV(0, 0, 1), nullptr, {0,0,0,1}, RHI::LoadOp::CLEAR, RHI::StoreOp::STORE },
				  RHI::RenderPassColorAttachment{target->getRTV(0, 1, 1), nullptr, {0,0,0,1}, RHI::LoadOp::CLEAR, RHI::StoreOp::STORE },
				  RHI::RenderPassColorAttachment{target->getRTV(0, 2, 1), nullptr, {0,0,0,1}, RHI::LoadOp::CLEAR, RHI::StoreOp::STORE },
				  RHI::RenderPassColorAttachment{target->getRTV(0, 3, 1), nullptr, {0,0,0,1}, RHI::LoadOp::CLEAR, RHI::StoreOp::STORE }},
				RHI::RenderPassDepthStencilAttachment{},
			};

			RHI::RenderPassEncoder* encoder = beginPass(context, target);

			dispatchFullScreen(context);

			encoder->end();
		}

		Core::GUID vert, frag;
	};
}