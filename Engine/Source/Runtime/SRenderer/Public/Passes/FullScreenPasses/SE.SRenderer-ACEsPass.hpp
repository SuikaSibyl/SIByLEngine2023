#pragma once

#include <array>
#include <memory>
#include <compare>
#include <typeinfo>
#include <filesystem>
#include <functional>
#include "../../../../Application/Public/SE.Application.Config.h"
#include <SE.Editor.Core.hpp>
#include <SE.RHI.hpp>
#include <SE.GFX.hpp>
#include <SE.RDG.hpp>
#include <SE.SRenderer.hpp>
#include <SE.Math.Geometric.hpp>
#include <Resource/SE.Core.Resource.hpp>

namespace SIByL
{
	SE_EXPORT struct ACEsPass :public RDG::FullScreenPass {

		ACEsPass() {
			frag = GFX::GFXManager::get()->registerShaderModuleResource(
				"../Engine/Binaries/Runtime/spirv/SRenderer/rasterizer/fullscreen_pass/aces_frag.spv", 
				{ nullptr, RHI::ShaderStages::FRAGMENT });
			RDG::FullScreenPass::init(Core::ResourceManager::get()->getResource<GFX::ShaderModule>(frag));
		}

		virtual auto reflect() noexcept -> RDG::PassReflection {
			RDG::PassReflection reflector;

			reflector.addInput("HDR")
				.isTexture()
				.withSize(Math::vec3(1, 1, 1))
				.withFormat(RHI::TextureFormat::RGBA32_FLOAT)
				.withUsages((uint32_t)RHI::TextureUsage::TEXTURE_BINDING)
				.consume(RDG::TextureInfo::ConsumeEntry{ RDG::TextureInfo::ConsumeType::TextureBinding }
					.addStage((uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT));

			reflector.addOutput("LDR")
				.isTexture()
				.withSize(Math::vec3(1, 1, 1))
				.withFormat(RHI::TextureFormat::RGBA8_UNORM)
				.withUsages((uint32_t)RHI::TextureUsage::COLOR_ATTACHMENT)
				.consume(RDG::TextureInfo::ConsumeEntry{ RDG::TextureInfo::ConsumeType::ColorAttachment }
					.enableDepthWrite(false)
					.setAttachmentLoc(0)
					.setDepthCompareFn(RHI::CompareFunction::ALWAYS));

			return reflector;
		}

		virtual auto execute(RDG::RenderContext* context, RDG::RenderData const& renderData) noexcept -> void {

			GFX::Texture* hdr = renderData.getTexture("HDR");
			GFX::Texture* ldr = renderData.getTexture("LDR");

			renderPassDescriptor = {
				{ RHI::RenderPassColorAttachment{
					ldr->getRTV(0, 0, 1),
					nullptr, {0,0,0,1}, RHI::LoadOp::CLEAR, RHI::StoreOp::STORE }},
				RHI::RenderPassDepthStencilAttachment{},
			};

			RHI::RenderPassEncoder* encoder = beginPass(context, ldr);

			getBindGroup(context, 0)->updateBinding(std::vector<RHI::BindGroupEntry>{
				RHI::BindGroupEntry{ 0,RHI::BindingResource(
					std::vector<RHI::TextureView*>{hdr->getSRV(0,1,0,1)},
					Core::ResourceManager::get()->getResource<GFX::Sampler>(GFX::GFXManager::get()->commonSampler.defaultSampler)->sampler.get())
				}
			});

			dispatchFullScreen(context);

			encoder->end();
		}

		Core::GUID vert, frag;
	};
}