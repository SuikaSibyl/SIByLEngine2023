#pragma once

#include <array>
#include <memory>
#include <compare>
#include <typeinfo>
#include <filesystem>
#include <functional>
#include "../../../../Application/Public/SE.Application.Config.h"
#include <SE.Math.Geometric.hpp>
#include <SE.Editor.Core.hpp>
#include <SE.RHI.hpp>
#include <SE.GFX.hpp>
#include <SE.RDG.hpp>
#include <SE.SRenderer.hpp>

namespace SIByL
{
	SE_EXPORT struct PreZOpaquePass :public RDG::RenderPass {

		PreZOpaquePass() {
			vert = GFX::GFXManager::get()->registerShaderModuleResource("../Engine/Binaries/Runtime/spirv/SRenderer/rasterizer/prez_pass/prez_pass_indirect_vert.spv", { nullptr, RHI::ShaderStages::VERTEX });
			frag = GFX::GFXManager::get()->registerShaderModuleResource("../Engine/Binaries/Runtime/spirv/SRenderer/rasterizer/prez_pass/prez_pass_frag.spv", { nullptr, RHI::ShaderStages::FRAGMENT });
			RDG::RenderPass::init(
				Core::ResourceManager::get()->getResource<GFX::ShaderModule>(vert),
				Core::ResourceManager::get()->getResource<GFX::ShaderModule>(frag));
		}

		virtual auto reflect() noexcept -> RDG::PassReflection {
			RDG::PassReflection reflector;

			reflector.addOutput("Depth")
				.isTexture()
				.withSize(Math::vec3(1, 1, 1))
				.withFormat(RHI::TextureFormat::DEPTH32_FLOAT)
				.withUsages((uint32_t)RHI::TextureUsage::DEPTH_ATTACHMENT)
				.consume(RDG::TextureInfo::ConsumeEntry{ RDG::TextureInfo::ConsumeType::DepthStencilAttachment }
					.enableDepthWrite(true)
					.setAttachmentLoc(0)
					.setDepthCompareFn(RHI::CompareFunction::LESS));

			return reflector;
		}

		virtual auto execute(RDG::RenderContext* context, RDG::RenderData const& renderData) noexcept -> void {

			GFX::Texture* depth = renderData.getTexture("Depth");

			renderPassDescriptor = {
				{},
				RHI::RenderPassDepthStencilAttachment{
					depth->getDSV(0,0,1),
					1, RHI::LoadOp::CLEAR, RHI::StoreOp::STORE, false,
					0, RHI::LoadOp::CLEAR, RHI::StoreOp::STORE, false
				},
			};

			RHI::RenderPassEncoder* encoder = beginPass(context, depth);
			
			std::vector<RHI::BindGroupEntry>* set_0_entries = renderData.getBindGroupEntries("CommonScene");
			getBindGroup(context, 0)->updateBinding(*set_0_entries);
			RHI::Buffer* indirect_draw_buffer = RACommon::get()->structured_drawcalls.all_drawcall_device->buffer.get();


			auto const& drawcall_info = RACommon::get()->structured_drawcalls.opaque_drawcall;

			if (drawcall_info.drawCount != 0) {
				getBindGroup(context, 1)->updateBinding({
					RHI::BindGroupEntry {0, RHI::BindingResource{ RHI::BufferBinding{indirect_draw_buffer, drawcall_info.offset, sizeof(RACommon::DrawIndexedIndirectEX) * drawcall_info.drawCount} }}
				});

				renderData.getDelegate("PrepareDrawcalls")(prepareDelegateData(context, renderData));
				encoder->drawIndexedIndirect(indirect_draw_buffer, drawcall_info.offset, drawcall_info.drawCount, sizeof(RACommon::DrawIndexedIndirectEX));
			}
			// do not use indirect
			//getBindGroup(context, 1)->updateBinding(RACommon::get()->);
			//renderData.getDelegate("IssueAllDrawcalls")(prepareDelegateData(context, renderData));

			encoder->end();
		}

		Core::GUID vert, frag;
	};


	SE_EXPORT struct PreZAlphPass :public RDG::RenderPass {

		PreZAlphPass() {
			vert = GFX::GFXManager::get()->registerShaderModuleResource("../Engine/Binaries/Runtime/spirv/SRenderer/rasterizer/prez_pass/prez_pass_alpha_indirect_vert.spv", { nullptr, RHI::ShaderStages::VERTEX });
			frag = GFX::GFXManager::get()->registerShaderModuleResource("../Engine/Binaries/Runtime/spirv/SRenderer/rasterizer/prez_pass/prez_pass_alpha_indirect_frag.spv", { nullptr, RHI::ShaderStages::FRAGMENT });
			RDG::RenderPass::init(
				Core::ResourceManager::get()->getResource<GFX::ShaderModule>(vert),
				Core::ResourceManager::get()->getResource<GFX::ShaderModule>(frag));
		}

		virtual auto reflect() noexcept -> RDG::PassReflection {
			RDG::PassReflection reflector;

			reflector.addInputOutput("Depth")
				.isTexture()
				.withSize(Math::vec3(1, 1, 1))
				.withFormat(RHI::TextureFormat::DEPTH32_FLOAT)
				.withUsages((uint32_t)RHI::TextureUsage::DEPTH_ATTACHMENT)
				.consume(RDG::TextureInfo::ConsumeEntry{ RDG::TextureInfo::ConsumeType::DepthStencilAttachment }
					.enableDepthWrite(true)
					.setAttachmentLoc(0)
					.setDepthCompareFn(RHI::CompareFunction::LESS));

			return reflector;
		}

		virtual auto execute(RDG::RenderContext* context, RDG::RenderData const& renderData) noexcept -> void {

			GFX::Texture* depth = renderData.getTexture("Depth");

			renderPassDescriptor = {
				{},
				RHI::RenderPassDepthStencilAttachment{
					depth->getDSV(0,0,1),
					1, RHI::LoadOp::LOAD, RHI::StoreOp::STORE, false,
					0, RHI::LoadOp::LOAD, RHI::StoreOp::STORE, false
				},
			};

			RHI::RenderPassEncoder* encoder = beginPass(context, depth);

			std::vector<RHI::BindGroupEntry>* set_0_entries = renderData.getBindGroupEntries("CommonScene");
			getBindGroup(context, 0)->updateBinding(*set_0_entries);
			RHI::Buffer* indirect_draw_buffer = RACommon::get()->structured_drawcalls.all_drawcall_device->buffer.get();


			auto const& drawcall_info = RACommon::get()->structured_drawcalls.alphacut_drawcall;

			if (drawcall_info.drawCount != 0) {
				getBindGroup(context, 1)->updateBinding({
					RHI::BindGroupEntry {0, RHI::BindingResource{ RHI::BufferBinding{indirect_draw_buffer, drawcall_info.offset, sizeof(RACommon::DrawIndexedIndirectEX) * drawcall_info.drawCount} }}
					});

				uint32_t sample_batch = renderData.getUInt("AccumIdx");
				encoder->pushConstants(&sample_batch, (uint32_t)RHI::ShaderStages::FRAGMENT, 0, sizeof(uint32_t));

				renderData.getDelegate("PrepareDrawcalls")(prepareDelegateData(context, renderData));
				encoder->drawIndexedIndirect(indirect_draw_buffer, drawcall_info.offset, drawcall_info.drawCount, sizeof(RACommon::DrawIndexedIndirectEX));
			}
			// do not use indirect
			//getBindGroup(context, 1)->updateBinding(RACommon::get()->);
			//renderData.getDelegate("IssueAllDrawcalls")(prepareDelegateData(context, renderData));

			encoder->end();
		}

		Core::GUID vert, frag;
	};

	SE_EXPORT struct PreZPass :public RDG::Subgraph {

		PreZPass() {}

		virtual auto alias() noexcept -> RDG::AliasDict override {
			RDG::AliasDict dict;
			dict.addAlias("Depth", CONCAT("Alpha"), "Depth");
			return dict;
		}

		virtual auto onRegister(RDG::Graph* graph) noexcept -> void override {
			graph->addPass(std::make_unique<PreZOpaquePass>(), CONCAT("Opaque"));
			graph->addPass(std::make_unique<PreZAlphPass>(), CONCAT("Alpha"));

			graph->addEdge(CONCAT("Opaque"), "Depth", CONCAT("Alpha"), "Depth");
		}
	};
}