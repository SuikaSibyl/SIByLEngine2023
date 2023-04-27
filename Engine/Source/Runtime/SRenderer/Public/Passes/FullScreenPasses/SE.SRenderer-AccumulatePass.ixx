module;
#include <cstdint>
#include <typeinfo>
#include <string>
#include <imgui.h>
#include <imgui_internal.h>
export module SE.SRenderer.AccumulatePass;
import SE.Core.Resource;
import SE.RHI;
import SE.GFX;
import SE.RDG;
import SE.SRenderer;

namespace SIByL
{
	export struct AccumulatePass :public RDG::ComputePass {

		AccumulatePass() {
			comp = GFX::GFXManager::get()->registerShaderModuleResource(
				"../Engine/Binaries/Runtime/spirv/SRenderer/rasterizer/fullscreen_pass/accumulate_comp.spv",
				{ nullptr, RHI::ShaderStages::COMPUTE });
			RDG::ComputePass::init(Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp));
		}

		virtual auto reflect() noexcept -> RDG::PassReflection {
			RDG::PassReflection reflector;

			reflector.addOutput("Output")
				.isTexture()
				.withSize(Math::vec3(1, 1, 1))
				.withFormat(RHI::TextureFormat::RGBA32_FLOAT)
				.withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
				.consume(RDG::TextureInfo::ConsumeEntry{ RDG::TextureInfo::ConsumeType::StorageBinding }
					.setSubresource(0, 1, 0, 1)
					.addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));

			reflector.addInput("Input")
				.isTexture()
				.withSize(Math::vec3(1, 1, 1))
				.withFormat(RHI::TextureFormat::RGBA32_FLOAT)
				.withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
				.consume(RDG::TextureInfo::ConsumeEntry{ RDG::TextureInfo::ConsumeType::StorageBinding }
					.setSubresource(0, 1, 0, 1)
					.addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));

			reflector.addInternal("LastSum")
				.isTexture()
				.withSize(Math::vec3(1, 1, 1))
				.withFormat(RHI::TextureFormat::RGBA32_FLOAT)
				.withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
				.consume(RDG::TextureInfo::ConsumeEntry{ RDG::TextureInfo::ConsumeType::StorageBinding }
					.setSubresource(0, 1, 0, 1)
					.addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));

			return reflector;
		}

		struct PushConstant {
			Math::uvec2 resolution;
			uint32_t gAccumCount;
			uint32_t gAccumulate;
			uint32_t gMovingAverageMode;
		};
		PushConstant pConst;

		int maxAccumCount = 5;

		virtual auto renderUI() noexcept -> void override {
			ImGui::DragInt("Max Accum", &maxAccumCount, 1, 0);
			if (ImGui::Button("Reset")) {
				pConst.gAccumCount = 0;
			}
			ImGui::SameLine();
			ImGui::Text(std::string("Accumulated Count: " + std::to_string(pConst.gAccumCount)).c_str());
		}
		virtual auto execute(RDG::RenderContext* context, RDG::RenderData const& renderData) noexcept -> void {

			GFX::Texture* output = renderData.getTexture("Output");
			GFX::Texture* input = renderData.getTexture("Input");
			GFX::Texture* sum = renderData.getTexture("LastSum");

			getBindGroup(context, 0)->updateBinding(std::vector<RHI::BindGroupEntry>{
				RHI::BindGroupEntry{ 0, RHI::BindingResource{ input->getUAV(0,0,1)} },
				RHI::BindGroupEntry{ 1, RHI::BindingResource{ sum->getUAV(0,0,1)} },
				RHI::BindGroupEntry{ 2, RHI::BindingResource{ output->getUAV(0,0,1)} },
			});

			RHI::ComputePassEncoder* encoder = beginPass(context);

			uint32_t width = input->texture->width();
			uint32_t height = input->texture->height();
			uint32_t batchIdx = renderData.getUInt("AccumIdx");

			prepareDispatch(context);

			pConst.resolution = Math::uvec2{ width,height };
			if (batchIdx == 0) {
				pConst.gAccumCount = 0;
			}
			pConst.gAccumulate = 1;
			pConst.gMovingAverageMode = pConst.gAccumCount > 0 ? 1 : 0;

			encoder->pushConstants(&pConst,
				(uint32_t)RHI::ShaderStages::COMPUTE,
				0, sizeof(PushConstant));
			encoder->dispatchWorkgroups((width + 15) / 16, (height + 15) / 16, 1);

			if (maxAccumCount == 0 || pConst.gAccumCount < maxAccumCount) {
				pConst.gAccumCount++;
			}

			encoder->end();
		}

		Core::GUID comp;
	};
}