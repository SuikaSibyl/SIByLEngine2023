module;
#include <array>
#include <memory>
#include <compare>
#include <typeinfo>
#include <filesystem>
#include <functional>
#include "../../../../Application/Public/SE.Application.Config.h"
export module SE.SRenderer.AlbedoPass;
import SE.Math.Geometric;
import SE.Core.Resource;
import SE.RHI;
import SE.GFX;
import SE.RDG;

namespace SIByL
{
	export struct AlbedoPass :public RDG::RenderPass {

		AlbedoPass() {
			vert = GFX::GFXManager::get()->registerShaderModuleResource("../Engine/Binaries/Runtime/spirv/SRenderer/rasterizer/basecolor_only_pass/basecolor_only_vert.spv", { nullptr, RHI::ShaderStages::VERTEX });
			frag = GFX::GFXManager::get()->registerShaderModuleResource("../Engine/Binaries/Runtime/spirv/SRenderer/rasterizer/basecolor_only_pass/basecolor_only_frag.spv", { nullptr, RHI::ShaderStages::FRAGMENT });
			RDG::RenderPass::init(
				Core::ResourceManager::get()->getResource<GFX::ShaderModule>(vert),
				Core::ResourceManager::get()->getResource<GFX::ShaderModule>(frag));
		}

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

			return reflector;
		}

		virtual auto execute(RDG::RenderContext* context, RDG::RenderData const& renderData) noexcept -> void {

			GFX::Texture* color = renderData.getTexture("Color");
			GFX::Texture* depth = renderData.getTexture("Depth");

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

			RHI::RenderPassEncoder* encoder = beginPass(context, color);

 			std::vector<RHI::BindGroupEntry>* set_0_entries = renderData.getBindGroupEntries("CommonScene");
			getBindGroup(context, 0)->updateBinding(*set_0_entries);

			renderData.getDelegate("IssueAllDrawcalls")(prepareDelegateData(context, renderData));

			encoder->end();
		}

		Core::GUID vert, frag;
	};
}