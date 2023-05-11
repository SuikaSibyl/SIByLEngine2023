#pragma once

#include <array>
#include <vector>
#include <memory>
#include <compare>
#include <typeinfo>
#include <filesystem>
#include <functional>
#include <imgui.h>
#include <imgui_internal.h>
#include "../../../../Application/Public/SE.Application.Config.h"
#include <SE.Math.Geometric.hpp>
#include <SE.Editor.Core.hpp>
#include <SE.RHI.hpp>
#include <SE.GFX.hpp>
#include <SE.RDG.hpp>
#include <SE.SRenderer.hpp>

namespace SIByL
{
	SE_EXPORT struct ForwardPass :public RDG::RenderPass {

		ForwardPass() {
			vert = GFX::GFXManager::get()->registerShaderModuleResource("../Engine/Binaries/Runtime/spirv/SRenderer/rasterizer/forward_pass/forward_vert.spv", { nullptr, RHI::ShaderStages::VERTEX });
			frag = GFX::GFXManager::get()->registerShaderModuleResource("../Engine/Binaries/Runtime/spirv/SRenderer/rasterizer/forward_pass/forward_frag.spv", { nullptr, RHI::ShaderStages::FRAGMENT });
			RDG::RenderPass::init(
				Core::ResourceManager::get()->getResource<GFX::ShaderModule>(vert),
				Core::ResourceManager::get()->getResource<GFX::ShaderModule>(frag));

			shadowmap_data_device = GFX::GFXManager::get()->createStructuredArrayMultiStorageBuffer<RACommon::ShadowmapInfo>(1);
		}

		GFX::StructuredArrayMultiStorageBufferView<RACommon::ShadowmapInfo> shadowmap_data_device;

		virtual auto reflect() noexcept -> RDG::PassReflection {
			RDG::PassReflection reflector;

			reflector.addOutput("Color")
				.isTexture()
				.withSize(Math::vec3(1, 1, 1))
				.withFormat(RHI::TextureFormat::RGBA32_FLOAT)
				.withUsages((uint32_t)RHI::TextureUsage::COLOR_ATTACHMENT)
				.consume(RDG::TextureInfo::ConsumeEntry{ RDG::TextureInfo::ConsumeType::ColorAttachment }
			.setAttachmentLoc(0));

			reflector.addInputOutput("Depth")
				.isTexture()
				.withSize(Math::vec3(1, 1, 1))
				.withFormat(RHI::TextureFormat::DEPTH32_FLOAT)
				.withUsages((uint32_t)RHI::TextureUsage::DEPTH_ATTACHMENT)
				.consume(RDG::TextureInfo::ConsumeEntry{ RDG::TextureInfo::ConsumeType::DepthStencilAttachment }
					.enableDepthWrite(false)
					.setAttachmentLoc(0)
					.setDepthCompareFn(RHI::CompareFunction::EQUAL));

			reflector.addInput("Shadowmap")
				.isTexture()
				.withUsages((uint32_t)RHI::TextureUsage::TEXTURE_BINDING)
				.consume(RDG::TextureInfo::ConsumeEntry{ RDG::TextureInfo::ConsumeType::TextureBinding }
					.setSubresource(0, 1, 0, 4)
					.addStage((uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT));

			return reflector;
		}

		struct PushConstants {
			uint32_t lightIndex;
			float bias = 0.005;
		} pConst;

		virtual auto renderUI() noexcept -> void override {
			{	float bias = pConst.bias;
				ImGui::DragFloat("Bias", &bias, 0.01);
				pConst.bias = bias;
			}
		}

		virtual auto execute(RDG::RenderContext* context, RDG::RenderData const& renderData) noexcept -> void {

			GFX::Texture* color = renderData.getTexture("Color");
			GFX::Texture* depth = renderData.getTexture("Depth");

			GFX::Texture* shadowmap = renderData.getTexture("Shadowmap");

			renderPassDescriptor = {
				{ RHI::RenderPassColorAttachment{
					color->getRTV(0, 0, 1),
					nullptr, {0,0,0,1}, RHI::LoadOp::CLEAR, RHI::StoreOp::STORE }},
				RHI::RenderPassDepthStencilAttachment{
					depth->getDSV(0,0,1),
					1, RHI::LoadOp::LOAD, RHI::StoreOp::DONT_CARE, false,
					0, RHI::LoadOp::LOAD, RHI::StoreOp::DONT_CARE, false
				},
			};

			shadowmap_data_device.setStructure(
				RACommon::get()->shadowmapData.data(),
				context->flightIdx,
				RACommon::get()->shadowmapData.size()
			);

			RHI::RenderPassEncoder* encoder = beginPass(context, color);
			pConst.lightIndex = RACommon::get()->mainDirectionalLight.value().lightID;
			encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::FRAGMENT, sizeof(uint32_t), sizeof(PushConstants));

			std::vector<RHI::BindGroupEntry>* set_0_entries = renderData.getBindGroupEntries("CommonScene");
			getBindGroup(context, 0)->updateBinding(*set_0_entries);
			getBindGroup(context, 1)->updateBinding(std::vector<RHI::BindGroupEntry>{
				RHI::BindGroupEntry{ 0, RHI::BindingResource(shadowmap->getSRV(0,1,0,4), Core::ResourceManager::get()->getResource<GFX::Sampler>(GFX::GFXManager::get()->commonSampler.clamp_nearest)->sampler.get()) },
				RHI::BindGroupEntry{ 1, shadowmap_data_device.getBufferBinding(context->flightIdx) },
				RHI::BindGroupEntry{ 2, RHI::BindingResource{RACommon::get()->csm_info_device.getBufferBinding(context->flightIdx)} }
			});

			renderData.getDelegate("IssueAllDrawcalls")(prepareDelegateData(context, renderData));

			encoder->end();
		}

		Core::GUID vert, frag;
	};
}