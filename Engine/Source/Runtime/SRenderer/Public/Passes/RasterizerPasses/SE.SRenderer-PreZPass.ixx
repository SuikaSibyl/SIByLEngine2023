module;
#include <array>
#include <memory>
#include <compare>
#include <typeinfo>
#include <filesystem>
#include <functional>
#include "../../../../Application/Public/SE.Application.Config.h"
export module SE.SRenderer.PreZPass;
import SE.Math.Geometric;
import SE.Core.Resource;
import SE.RHI;
import SE.GFX;
import SE.RDG;

namespace SIByL
{
	export struct PreZPass :public RDG::RenderPass {

		PreZPass() {
			vert = GFX::GFXManager::get()->registerShaderModuleResource("../Engine/Binaries/Runtime/spirv/SRenderer/rasterizer/prez_pass/prez_pass_vert.spv", { nullptr, RHI::ShaderStages::VERTEX });
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

			renderData.getDelegate("IssueAllDrawcalls")(prepareDelegateData(context, renderData));

			encoder->end();
		}

		Core::GUID vert, frag;
	};
}