#define DLIB_EXPORT
#include <passes/se.pass.postprocess.hpp>
#undef DLIB_EXPORT
#include <imgui.h>

namespace se {
  AccumulatePass::AccumulatePass(ivec3 resolution) :resolution(resolution) {
    auto [comp] = gfx::GFXContext::load_shader_slang(
      "passes/postprocess/accumulate.slang",
      std::array<std::pair<std::string, rhi::ShaderStageBit>, 1>{
        std::make_pair("ComputeMain", rhi::ShaderStageBit::COMPUTE),
    }, {}, true);
    rdg::ComputePass::init(comp.get());
  }

  auto AccumulatePass::reflect() noexcept -> rdg::PassReflection {
	rdg::PassReflection reflector;
	reflector.addOutput("Output")
	  .isTexture().withSize(resolution)
	  .withFormat(rhi::TextureFormat::RGBA32_FLOAT)
	  .withUsages((uint32_t)rhi::TextureUsageBit::STORAGE_BINDING)
	  .consume(rdg::TextureInfo::ConsumeEntry{ rdg::TextureInfo::ConsumeType::StorageBinding }
		  .setSubresource(0, 1, 0, 1)
		  .addStage((uint32_t)rhi::PipelineStageBit::COMPUTE_SHADER_BIT));
	reflector.addInput("Input")
	  .isTexture().withSize(resolution)
	  .withFormat(rhi::TextureFormat::RGBA32_FLOAT)
	  .withUsages((uint32_t)rhi::TextureUsageBit::STORAGE_BINDING)
	  .consume(rdg::TextureInfo::ConsumeEntry{ rdg::TextureInfo::ConsumeType::StorageBinding }
		  .setSubresource(0, 1, 0, 1)
		  .addStage((uint32_t)rhi::PipelineStageBit::COMPUTE_SHADER_BIT));
	reflector.addInternal("LastSum")
	  .isTexture().withSize(resolution)
	  .withFormat(rhi::TextureFormat::RGBA32_FLOAT)
	  .withUsages((uint32_t)rhi::TextureUsageBit::STORAGE_BINDING)
	  .consume(rdg::TextureInfo::ConsumeEntry{ rdg::TextureInfo::ConsumeType::StorageBinding }
		.setSubresource(0, 1, 0, 1)
		.addStage((uint32_t)rhi::PipelineStageBit::COMPUTE_SHADER_BIT));
	return reflector;
  }

  auto AccumulatePass::renderUI() noexcept -> void {
	ImGui::DragInt("Max Accum", &maxAccumCount, 1, 0);
	if (ImGui::Button("Reset")) {
	  pConst.gAccumCount = 0;
	}
	ImGui::SameLine();
	bool useAccum = pConst.gAccumulate != 0;
	if (ImGui::Checkbox("Use Accum", &useAccum)) {
	  pConst.gAccumulate = useAccum ? 1 : 0;
	}
	//pConst.gAccumulate = useAccum ? 1 : 0;
	ImGui::Text(std::string("Accumulated Count: " + std::to_string(pConst.gAccumCount)).c_str());
  }
  
  auto AccumulatePass::execute(rdg::RenderContext* context, rdg::RenderData const& renderData) noexcept -> void {
	gfx::TextureHandle output = renderData.getTexture("Output");
	gfx::TextureHandle input = renderData.getTexture("Input");
	gfx::TextureHandle sum = renderData.getTexture("LastSum");

	updateBindings(context, {
	  {"u_input", rhi::BindingResource{input->getUAV(0,0,1)}},
	  {"u_lastSum", rhi::BindingResource{sum->getUAV(0,0,1)}},
	  {"u_output", rhi::BindingResource{output->getUAV(0,0,1)}},
	});

	uint32_t width = input->texture->width();
	uint32_t height = input->texture->height();
	uint32_t batchIdx = 1;
	pConst.resolution = se::uvec2{ width,height };
	if (batchIdx == 0) pConst.gAccumCount = 0;
	pConst.gMovingAverageMode = pConst.gAccumCount > 0 ? 1 : 0;

	se::rhi::ComputePassEncoder* encoder = beginPass(context);
	encoder->pushConstants(&pConst, (uint32_t)rhi::ShaderStageBit::COMPUTE, 0, sizeof(pConst));
	encoder->dispatchWorkgroups((width + 15) / 16, (height + 15) / 16, 1);
	encoder->end();
	
	if (maxAccumCount == 0 || pConst.gAccumCount < maxAccumCount) pConst.gAccumCount++;
  }
}