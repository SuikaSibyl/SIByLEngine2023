#include <filesystem>
#include <typeinfo>
#include <cstdint>
#include <memory>
#include "../../../../Application/Public/SE.Application.Config.h"
#include "../../../../Application/Public/SE.Application.Config.h"
#include <SE.Editor.Core.hpp>
#include <SE.RHI.hpp>
#include <SE.GFX.hpp>
#include <SE.RDG.hpp>
#include <SE.SRenderer.hpp>

namespace SIByL::SRP
{
	SE_EXPORT struct BDPTPass :public RDG::RayTracingPass {

		struct PushConstant {
			uint32_t width;
			uint32_t height;
			uint32_t sample_batch;
		};

		BDPTPass() {
			bdpt_rgen = GFX::GFXManager::get()->registerShaderModuleResource("../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/bdpt/bdpt_rgen.spv", { nullptr, RHI::ShaderStages::RAYGEN });

			GFX::SBTsDescriptor sbt = RTCommon::get()->getSBTDescriptor();
			sbt.rgenSBT = GFX::SBTsDescriptor::RayGenerationSBT{ { Core::ResourceManager::get()->getResource<GFX::ShaderModule>(bdpt_rgen) } };

			RayTracingPass::init(sbt, 3);
		}

		virtual auto reflect() noexcept -> RDG::PassReflection override {
			RDG::PassReflection reflector;

			reflector.addInputOutput("atomicRGBA")
				.isTexture()
				.withSize(Math::vec3(1, 1, 1))
				.withFormat(RHI::TextureFormat::R32_FLOAT)
				.withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
				.withLayers(4)
				.consume(RDG::TextureInfo::ConsumeEntry{ RDG::TextureInfo::ConsumeType::StorageBinding }
					.setSubresource(0,1,0,4)
					.addStage((uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));

			return reflector;
		}

		virtual auto execute(RDG::RenderContext* context, RDG::RenderData const& renderData) noexcept -> void override {

			GFX::Texture* atomicRGBA = renderData.getTexture("atomicRGBA");

			std::vector<RHI::BindGroupEntry>* set_0_entries = renderData.getBindGroupEntries("CommonScene");
			getBindGroup(context, 0)->updateBinding(*set_0_entries);
			std::vector<RHI::BindGroupEntry>* set_1_entries = renderData.getBindGroupEntries("CommonRT");
			getBindGroup(context, 1)->updateBinding(*set_1_entries);
			getBindGroup(context, 2)->updateBinding({ RHI::BindGroupEntry{ 0, RHI::BindingResource{ atomicRGBA->getUAV(0,0,4)} } });

			RHI::RayTracingPassEncoder* encoder = beginPass(context);

			uint32_t batchIdx = renderData.getUInt("AccumIdx");
			PushConstant pConst = {
				atomicRGBA->texture->width(),
				atomicRGBA->texture->height(),
				batchIdx
			};
			encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::RAYGEN, 0, sizeof(PushConstant));
			encoder->traceRays(atomicRGBA->texture->width(), atomicRGBA->texture->height(), 1);

			encoder->end();
		}

		Core::GUID bdpt_rgen;
	};
}