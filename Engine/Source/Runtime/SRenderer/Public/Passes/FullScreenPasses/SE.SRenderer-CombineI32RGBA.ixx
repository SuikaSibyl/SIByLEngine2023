module;
#include <array>
#include <memory>
#include <compare>
#include <typeinfo>
#include <filesystem>
#include <functional>
#include "../../../../Application/Public/SE.Application.Config.h"
export module SE.SRenderer.CombineI32RGBA;
import SE.Math.Geometric;
import SE.Core.Resource;
import SE.RHI;
import SE.GFX.Core;
import SE.RDG;

namespace SIByL
{
	export struct CombineI32RGBAPass :public RDG::ComputePass {

		CombineI32RGBAPass() {
			frag = GFX::GFXManager::get()->registerShaderModuleResource(
				"../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/bdpt/combine_i32a4_comp.spv",
				{ nullptr, RHI::ShaderStages::COMPUTE });
			RDG::ComputePass::init(Core::ResourceManager::get()->getResource<GFX::ShaderModule>(frag));
		}

		virtual auto reflect() noexcept -> RDG::PassReflection {
			RDG::PassReflection reflector;

			reflector.addOutput("HDRAccum")
				.isTexture()
				.withSize(Math::vec3(1, 1, 1))
				.withFormat(RHI::TextureFormat::RGBA32_FLOAT)
				.withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
				.consume(RDG::TextureInfo::ConsumeEntry{ RDG::TextureInfo::ConsumeType::StorageBinding }
					.addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));

			reflector.addInput("I32RGBA")
				.isTexture()
				.withSize(Math::vec3(1, 1, 1))
				.withFormat(RHI::TextureFormat::R32_FLOAT)
				.withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
				.withLayers(4)
				.consume(RDG::TextureInfo::ConsumeEntry{ RDG::TextureInfo::ConsumeType::StorageBinding }
					.setSubresource(0, 1, 0, 4)
					.addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));

			return reflector;
		}

		virtual auto execute(RDG::RenderContext* context, RDG::RenderData const& renderData) noexcept -> void {

			GFX::Texture* hdr = renderData.getTexture("HDRAccum");
			GFX::Texture* rgba = renderData.getTexture("I32RGBA");

			getBindGroup(context, 0)->updateBinding(std::vector<RHI::BindGroupEntry>{
				RHI::BindGroupEntry{ 0, RHI::BindingResource{ hdr->getUAV(0,0,1)} },
				RHI::BindGroupEntry{ 1, RHI::BindingResource{ rgba->getUAV(0,0,4)} }
			});

			RHI::ComputePassEncoder* encoder = beginPass(context);

			uint32_t width = hdr->texture->width();
			uint32_t height = hdr->texture->height();
			uint32_t batchIdx = renderData.getUInt("AccumIdx");

			prepareDispatch(context);

			struct PushConstant {
				Math::uvec2 resolution;
				uint32_t sample_batch;
			};
			PushConstant pconst = {
				Math::uvec2{width,height},
				batchIdx
			};
			encoder->pushConstants(&pconst,
				(uint32_t)RHI::ShaderStages::COMPUTE,
				0, sizeof(PushConstant));
			encoder->dispatchWorkgroups((width + 15) / 16, (height + 15) / 16, 1);

			encoder->end();
		}

		Core::GUID vert, frag;
	};
}