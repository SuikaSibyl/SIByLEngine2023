module;
#include <array>
#include <memory>
#include <compare>
#include <typeinfo>
#include <filesystem>
#include <functional>
#include <imgui.h>
#include <imgui_internal.h>
#include "../../../../Application/Public/SE.Application.Config.h"
export module SE.SRenderer.FooPass;
import SE.SRenderer;
import SE.Math.Geometric;
import SE.Core.Resource;
import SE.Math.Geometric;
import SE.RHI;
import SE.GFX;
import SE.RDG;

namespace SIByL
{
	export struct FooPass :public RDG::RenderPass {

		FooPass() {
			vert = GFX::GFXManager::get()->registerShaderModuleResource("../Engine/Binaries/Runtime/spirv/SRenderer/rasterizer/basecolor_only_pass/basecolor_only_vert.spv", { nullptr, RHI::ShaderStages::VERTEX });
			frag = GFX::GFXManager::get()->registerShaderModuleResource("../Engine/Binaries/Runtime/spirv/SRenderer/rasterizer/basecolor_only_pass/basecolor_lumin_frag.spv", { nullptr, RHI::ShaderStages::FRAGMENT });
			RDG::RenderPass::init(
				Core::ResourceManager::get()->getResource<GFX::ShaderModule>(vert),
				Core::ResourceManager::get()->getResource<GFX::ShaderModule>(frag));

			global_uniform_buffer = GFX::GFXManager::get()->createStructuredUniformBuffer<SRenderer::GlobalUniforms>();
		}

		virtual auto reflect() noexcept -> RDG::PassReflection {
			RDG::PassReflection reflector;

			reflector.addOutput("LightProjection")
				.isTexture()
				.withSize(Math::ivec3(512, 512, 1))
				.withFormat(RHI::TextureFormat::RGBA32_FLOAT)
				.withUsages((uint32_t)RHI::TextureUsage::COLOR_ATTACHMENT)
				.consume(RDG::TextureInfo::ConsumeEntry{ RDG::TextureInfo::ConsumeType::ColorAttachment }
					.setAttachmentLoc(0));

			reflector.addOutput("LightProjLumMIP")
				.isTexture()
				.withSize(Math::ivec3(512, 512, 1))
				.withFormat(RHI::TextureFormat::R32_FLOAT)
				.withUsages((uint32_t)RHI::TextureUsage::COLOR_ATTACHMENT)
				.withLevels(RDG::MaxPossible)
				.consume(RDG::TextureInfo::ConsumeEntry{ RDG::TextureInfo::ConsumeType::ColorAttachment }
					.setSubresource(0, 1, 0, 1)
					.setAttachmentLoc(1));

			reflector.addOutput("LightProjectionDepth")
				.isTexture()
				.withSize(Math::ivec3(512, 512, 1))
				.withFormat(RHI::TextureFormat::DEPTH32_FLOAT)
				.withUsages((uint32_t)RHI::TextureUsage::DEPTH_ATTACHMENT)
				.consume(RDG::TextureInfo::ConsumeEntry{ RDG::TextureInfo::ConsumeType::DepthStencilAttachment }
					.enableDepthWrite(true)
					.setAttachmentLoc(0)
					.setDepthCompareFn(RHI::CompareFunction::LESS));

			return reflector;
		}

		Math::mat4 viewProjMat;
		SRenderer::GlobalUniforms gUni;
		GFX::StructuredUniformBufferView<SRenderer::GlobalUniforms> global_uniform_buffer;

		float xy[2] = { 1.4, 3.4 };
		float wh[2] = { 2.2, 3.5 };

		virtual auto renderUI() noexcept -> void override {
			ImGui::DragFloat2("XY", xy, 0.1);
			ImGui::DragFloat2("WH", wh, 0.5, 0, 100);
		}

		virtual auto execute(RDG::RenderContext* context, RDG::RenderData const& renderData) noexcept -> void {

			GFX::Texture* color = renderData.getTexture("LightProjection");
			GFX::Texture* depth = renderData.getTexture("LightProjectionDepth");
			GFX::Texture* lumin = renderData.getTexture("LightProjLumMIP");

			renderPassDescriptor = {
				{ RHI::RenderPassColorAttachment{color->getRTV(0, 0, 1), nullptr, {0,0,0,1}, RHI::LoadOp::CLEAR, RHI::StoreOp::STORE }, 
				  RHI::RenderPassColorAttachment{lumin->getRTV(0, 0, 1), nullptr, {0,0,0,1}, RHI::LoadOp::CLEAR, RHI::StoreOp::STORE }},
				RHI::RenderPassDepthStencilAttachment{
					depth->getDSV(0,0,1),
					1, RHI::LoadOp::CLEAR, RHI::StoreOp::STORE, false,
					0, RHI::LoadOp::CLEAR, RHI::StoreOp::STORE, false
				},
			};

			RHI::RenderPassEncoder* encoder = beginPass(context, color);

			gUni.cameraData.viewMat = Math::transpose(Math::lookAt(Math::vec3(xy[0], xy[1], 10), Math::vec3(xy[0], xy[1], 0), Math::vec3(0, 1, 0)).m);
			gUni.cameraData.projMat = Math::transpose(Math::ortho(-wh[0], wh[0], -wh[1], wh[1], 0.001, 100).m);
			gUni.cameraData.viewProjMat = gUni.cameraData.viewMat * gUni.cameraData.projMat;
			global_uniform_buffer.setStructure(gUni, context->flightIdx);

			std::vector<RHI::BindGroupEntry> set_0_entries = *renderData.getBindGroupEntries("CommonScene");
			set_0_entries[0] = RHI::BindGroupEntry{ 0,RHI::BindingResource{global_uniform_buffer.getBufferBinding(context->flightIdx)} };
			getBindGroup(context, 0)->updateBinding(set_0_entries);

			renderData.getDelegate("IssueDrawcalls_LightOnly")(prepareDelegateData(context, renderData));

			encoder->end();
		}

		Core::GUID vert, frag;
	};
}