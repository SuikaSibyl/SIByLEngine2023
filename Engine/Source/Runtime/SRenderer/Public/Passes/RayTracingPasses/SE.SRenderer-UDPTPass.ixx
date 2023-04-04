module;
#include <filesystem>
#include <typeinfo>
#include <cstdint>
#include <memory>
#include "../../../../Application/Public/SE.Application.Config.h"
export module SE.SRenderer.UDPTPass;
import SE.Core.Resource;
import SE.RHI;
import SE.GFX;
import SE.RDG;
import SE.SRenderer;

namespace SIByL::SRP
{
	export struct UDPTPass :public RDG::RayTracingPass {

		struct PushConstant {
			uint32_t width;
			uint32_t height;
			uint32_t sample_batch;
		};

		UDPTPass() {
			udpt_rgen = GFX::GFXManager::get()->registerShaderModuleResource("../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/udpt/udpt_rgen.spv", { nullptr, RHI::ShaderStages::RAYGEN });

			GFX::SBTsDescriptor sbt = RTCommon::get()->getSBTDescriptor();
			sbt.rgenSBT = GFX::SBTsDescriptor::RayGenerationSBT{ { Core::ResourceManager::get()->getResource<GFX::ShaderModule>(udpt_rgen) } };

			RayTracingPass::init(sbt, 3);
		}

		virtual auto reflect() noexcept -> RDG::PassReflection override {
			RDG::PassReflection reflector;

			reflector.addOutput("Color")
				.isTexture()
				.withSize(Math::vec3(1, 1, 1))
				.withFormat(RHI::TextureFormat::RGBA32_FLOAT)
				.withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
				.consume(RDG::TextureInfo::ConsumeEntry{ RDG::TextureInfo::ConsumeType::StorageBinding}
					.addStage((uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));

			return reflector;
		}

		virtual auto execute(RDG::RenderContext* context, RDG::RenderData const& renderData) noexcept -> void override {

			GFX::Texture* color = renderData.getTexture("Color");

			std::vector<RHI::BindGroupEntry>* set_0_entries = renderData.getBindGroupEntries("CommonScene");
			getBindGroup(context, 0)->updateBinding(*set_0_entries);
			std::vector<RHI::BindGroupEntry>* set_1_entries = renderData.getBindGroupEntries("CommonRT");
			getBindGroup(context, 1)->updateBinding(*set_1_entries);
			getBindGroup(context, 2)->updateBinding({ RHI::BindGroupEntry{ 0, RHI::BindingResource{ color->getUAV(0,0,1)} } });


			RHI::RayTracingPassEncoder* encoder = beginPass(context);

			uint32_t batchIdx = renderData.getUInt("AccumIdx");

			PushConstant pConst = {
				color->texture->width(),
				color->texture->height(),
				batchIdx
			};
			encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::RAYGEN, 0, sizeof(PushConstant));
			encoder->traceRays(color->texture->width(), color->texture->height(), 1);

			encoder->end();
		}

		Core::GUID udpt_rgen;
	};
}