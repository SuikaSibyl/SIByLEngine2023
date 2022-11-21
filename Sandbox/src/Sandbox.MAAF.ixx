module;
#include <array>
#include <filesystem>
#include <memory>
#include <utility>
export module Sandbox.MAAF;
import Core.Resource.RuntimeManage;
import Math.Vector;
import RHI;
import RHI.RHILayer;
import GFX.Resource;
import GFX.GFXManager;

using namespace SIByL;

namespace Sandbox
{
	export struct MAAF_Initial_Sample_Pass {
		MAAF_Initial_Sample_Pass(RHI::RHILayer* rhiLayer, RHI::PipelineLayout* rtPipelineLayout,
			std::array<RHI::BindGroup*, 2> const& rtBindGroups,
			std::array<RHI::BindGroup*, 2> const& camBindGroups,
			std::array<RHI::BindGroup*, 2> const& bufferBindGroups)
			: rtBindGroups(rtBindGroups)
			, camBindGroups(camBindGroups)
			, bufferBindGroups(bufferBindGroups) {
			// require GUID
			maaf_initial_sampling_rgen		 = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();	
			maaf_initial_sampling_rchit		 = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();	
			maaf_initial_sampling_rmiss		 = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();
			maaf_indirect_sampling_rchit	 = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();
			maaf_indirect_sampling_rmiss	 = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();
			maaf_shadow_dist_sampling_rahit	 = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();
			maaf_shadow_dist_sampling_rmiss	 = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();
			maaf_shadow_sampling_rchit		 = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();
			maaf_shadow_sampling_rmiss		 = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();
			// load Shaders
			GFX::GFXManager::get()->registerShaderModuleResource(maaf_initial_sampling_rgen, "../Engine/Binaries/Runtime/spirv/RayTracing/RayTrace/src/maaf_combined/maaf_initial_sampling_rgen.spv", { nullptr, RHI::ShaderStages::RAYGEN });
			GFX::GFXManager::get()->registerShaderModuleResource(maaf_initial_sampling_rchit, "../Engine/Binaries/Runtime/spirv/RayTracing/RayTrace/src/maaf_combined/maaf_initial_sampling_rchit.spv", { nullptr, RHI::ShaderStages::CLOSEST_HIT });
			GFX::GFXManager::get()->registerShaderModuleResource(maaf_initial_sampling_rmiss, "../Engine/Binaries/Runtime/spirv/RayTracing/RayTrace/src/maaf_combined/maaf_initial_sampling_rmiss.spv", { nullptr, RHI::ShaderStages::MISS });
			GFX::GFXManager::get()->registerShaderModuleResource(maaf_indirect_sampling_rchit, "../Engine/Binaries/Runtime/spirv/RayTracing/RayTrace/src/maaf_combined/maaf_indirect_sampling_rchit.spv", { nullptr, RHI::ShaderStages::CLOSEST_HIT });
			GFX::GFXManager::get()->registerShaderModuleResource(maaf_indirect_sampling_rmiss, "../Engine/Binaries/Runtime/spirv/RayTracing/RayTrace/src/maaf_combined/maaf_indirect_sampling_rmiss.spv", { nullptr, RHI::ShaderStages::MISS });
			GFX::GFXManager::get()->registerShaderModuleResource(maaf_shadow_dist_sampling_rahit, "../Engine/Binaries/Runtime/spirv/RayTracing/RayTrace/src/maaf_combined/maaf_shadow_dist_sampling_rahit.spv", { nullptr, RHI::ShaderStages::ANY_HIT });
			GFX::GFXManager::get()->registerShaderModuleResource(maaf_shadow_dist_sampling_rmiss, "../Engine/Binaries/Runtime/spirv/RayTracing/RayTrace/src/maaf_combined/maaf_shadow_dist_sampling_rmiss.spv", { nullptr, RHI::ShaderStages::MISS });
			GFX::GFXManager::get()->registerShaderModuleResource(maaf_shadow_sampling_rchit, "../Engine/Binaries/Runtime/spirv/RayTracing/RayTrace/src/maaf_combined/maaf_shadow_sampling_rchit.spv", { nullptr, RHI::ShaderStages::CLOSEST_HIT });
			GFX::GFXManager::get()->registerShaderModuleResource(maaf_shadow_sampling_rmiss, "../Engine/Binaries/Runtime/spirv/RayTracing/RayTrace/src/maaf_combined/maaf_shadow_sampling_rmiss.spv", { nullptr, RHI::ShaderStages::MISS });
			// Create rt pipeline
			for (int i = 0; i < 2; ++i) {
				raytracingPipeline[i] = rhiLayer->getDevice()->createRayTracingPipeline(RHI::RayTracingPipelineDescriptor{
					rtPipelineLayout, 3, RHI::SBTsDescriptor{
						RHI::SBTsDescriptor::RayGenerationSBT{{ Core::ResourceManager::get()->getResource<GFX::ShaderModule>(maaf_initial_sampling_rgen)->shaderModule.get() }},
						RHI::SBTsDescriptor::MissSBT{{
							{Core::ResourceManager::get()->getResource<GFX::ShaderModule>(maaf_initial_sampling_rmiss)->shaderModule.get()},
							{Core::ResourceManager::get()->getResource<GFX::ShaderModule>(maaf_shadow_dist_sampling_rmiss)->shaderModule.get()},
							{Core::ResourceManager::get()->getResource<GFX::ShaderModule>(maaf_indirect_sampling_rmiss)->shaderModule.get()},
							{Core::ResourceManager::get()->getResource<GFX::ShaderModule>(maaf_shadow_sampling_rmiss)->shaderModule.get()} }},
						RHI::SBTsDescriptor::HitGroupSBT{{
							{{Core::ResourceManager::get()->getResource<GFX::ShaderModule>(maaf_initial_sampling_rchit)->shaderModule.get()}, nullptr, nullptr},
							{nullptr, {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(maaf_shadow_dist_sampling_rahit)->shaderModule.get()}, nullptr},
							{{Core::ResourceManager::get()->getResource<GFX::ShaderModule>(maaf_indirect_sampling_rchit)->shaderModule.get()}, nullptr, nullptr},
							{{Core::ResourceManager::get()->getResource<GFX::ShaderModule>(maaf_shadow_sampling_rchit)->shaderModule.get()}, nullptr, nullptr} }}
					} });
			}
		}

		~MAAF_Initial_Sample_Pass() {
			raytracingPipeline[0] = nullptr;
			raytracingPipeline[1] = nullptr;
		}

		auto composeCommands(RHI::CommandEncoder* encoder, int index) noexcept -> void {
			static uint32_t batchIdx = 0;
			rtEncoder[index] = encoder->beginRayTracingPass(RHI::RayTracingPassDescriptor{});
			rtEncoder[index]->setPipeline(raytracingPipeline[index].get());
			rtEncoder[index]->setBindGroup(0, rtBindGroups[index], 0, 0);
			rtEncoder[index]->setBindGroup(1, camBindGroups[index], 0, 0);
			rtEncoder[index]->setBindGroup(2, bufferBindGroups[index], 0, 0);
			rtEncoder[index]->pushConstants(&batchIdx,
				(uint32_t)RHI::ShaderStages::RAYGEN
				| (uint32_t)RHI::ShaderStages::COMPUTE
				| (uint32_t)RHI::ShaderStages::CLOSEST_HIT
				| (uint32_t)RHI::ShaderStages::MISS
				| (uint32_t)RHI::ShaderStages::ANY_HIT,
				0, sizeof(uint32_t));
			rtEncoder[index]->traceRays(800, 600, 1);
			rtEncoder[index]->end();
			++batchIdx;
		}

		Core::GUID maaf_initial_sampling_rgen;
		Core::GUID maaf_initial_sampling_rchit;
		Core::GUID maaf_initial_sampling_rmiss;
		Core::GUID maaf_indirect_sampling_rchit;
		Core::GUID maaf_indirect_sampling_rmiss;
		Core::GUID maaf_shadow_dist_sampling_rahit;
		Core::GUID maaf_shadow_dist_sampling_rmiss;
		Core::GUID maaf_shadow_sampling_rchit;
		Core::GUID maaf_shadow_sampling_rmiss;

		std::unique_ptr<RHI::RayTracingPipeline> raytracingPipeline[2];
		std::unique_ptr<RHI::RayTracingPassEncoder> rtEncoder[2] = {};
		std::array<RHI::BindGroup*, 2> rtBindGroups = {};
		std::array<RHI::BindGroup*, 2> camBindGroups = {};
		std::array<RHI::BindGroup*, 2> bufferBindGroups = {};
	};

	export struct MAAF_Prefilter_Pass {
		MAAF_Prefilter_Pass(RHI::RHILayer* rhiLayer, RHI::PipelineLayout* slopleFilterPipelineLayout,
			std::array<RHI::BindGroup*, 2> const& bufferBindGroups)
			: slopleFilterPipelineLayout(slopleFilterPipelineLayout), bufferBindGroups(bufferBindGroups)
		{
			maaf_prefilter_x_comp = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();
			maaf_prefilter_y_comp = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();
			GFX::GFXManager::get()->registerShaderModuleResource(maaf_prefilter_x_comp,
				"../Engine/Binaries/Runtime/spirv/RayTracing/RayTrace/src/maaf_combined/maaf_prefilter_x_comp.spv",
				{ nullptr, RHI::ShaderStages::COMPUTE });
			GFX::GFXManager::get()->registerShaderModuleResource(maaf_prefilter_y_comp,
				"../Engine/Binaries/Runtime/spirv/RayTracing/RayTrace/src/maaf_combined/maaf_prefilter_y_comp.spv",
				{ nullptr, RHI::ShaderStages::COMPUTE });
			// create compute pipeline
			for (int i = 0; i < 2; ++i) {
				computePipelineX[i] = rhiLayer->getDevice()->createComputePipeline(RHI::ComputePipelineDescriptor{
					slopleFilterPipelineLayout,
					{Core::ResourceManager::get()->getResource<GFX::ShaderModule>(maaf_prefilter_x_comp)->shaderModule.get(), "main"}
					});
				computePipelineY[i] = rhiLayer->getDevice()->createComputePipeline(RHI::ComputePipelineDescriptor{
					slopleFilterPipelineLayout,
					{Core::ResourceManager::get()->getResource<GFX::ShaderModule>(maaf_prefilter_y_comp)->shaderModule.get(), "main"}
					});
			}

		}

		~MAAF_Prefilter_Pass() {
			for (int i = 0; i < 2; ++i) {
				computePipelineX[i] = nullptr;
				computePipelineY[i] = nullptr;
				compEncoderX[i] = nullptr;
				compEncoderY[i] = nullptr;
			}
		}

		auto composeCommands_x(RHI::CommandEncoder* encoder, int index) noexcept -> void {
			compEncoderX[index] = encoder->beginComputePass({});
			compEncoderX[index]->setPipeline(computePipelineX[index].get());
			compEncoderX[index]->setBindGroup(0, bufferBindGroups[index], 0, 0);
			compEncoderX[index]->dispatchWorkgroups((800 + 15) / 16, (600 + 15) / 16, 1);
			compEncoderX[index]->end();
		}

		auto composeCommands_y(RHI::CommandEncoder* encoder, int index) noexcept -> void {
			compEncoderY[index] = encoder->beginComputePass({});
			compEncoderY[index]->setPipeline(computePipelineY[index].get());
			compEncoderY[index]->setBindGroup(0, bufferBindGroups[index], 0, 0);
			compEncoderY[index]->dispatchWorkgroups((800 + 15) / 16, (600 + 15) / 16, 1);
			compEncoderY[index]->end();
		}

		Core::GUID maaf_prefilter_x_comp;
		Core::GUID maaf_prefilter_y_comp;
		RHI::PipelineLayout* slopleFilterPipelineLayout;
		std::unique_ptr<RHI::ComputePipeline> computePipelineX[2];
		std::unique_ptr<RHI::ComputePipeline> computePipelineY[2];
		std::array<RHI::BindGroup*, 2> bufferBindGroups;
		std::unique_ptr<RHI::ComputePassEncoder> compEncoderX[2] = {};
		std::unique_ptr<RHI::ComputePassEncoder> compEncoderY[2] = {};
	};

	export struct MAAF_Continue_Sample_Pass {
		MAAF_Continue_Sample_Pass(RHI::RHILayer* rhiLayer, RHI::PipelineLayout* rtPipelineLayout,
			std::array<RHI::BindGroup*, 2> const& rtBindGroups,
			std::array<RHI::BindGroup*, 2> const& camBindGroups,
			std::array<RHI::BindGroup*, 2> const& bufferBindGroups)
			: rtBindGroups(rtBindGroups)
			, camBindGroups(camBindGroups)
			, bufferBindGroups(bufferBindGroups) {
			// require GUID
			maaf_initial_sampling_rgen = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();
			maaf_initial_sampling_rchit = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();
			maaf_initial_sampling_rmiss = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();
			maaf_indirect_sampling_rchit = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();
			maaf_indirect_sampling_rmiss = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();
			maaf_shadow_dist_sampling_rahit = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();
			maaf_shadow_dist_sampling_rmiss = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();
			maaf_shadow_sampling_rchit = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();
			maaf_shadow_sampling_rmiss = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();
			// load Shaders
			GFX::GFXManager::get()->registerShaderModuleResource(maaf_initial_sampling_rgen, "../Engine/Binaries/Runtime/spirv/RayTracing/RayTrace/src/maaf_combined/maaf_continue_sample_rgen.spv", { nullptr, RHI::ShaderStages::RAYGEN });
			GFX::GFXManager::get()->registerShaderModuleResource(maaf_initial_sampling_rchit, "../Engine/Binaries/Runtime/spirv/RayTracing/RayTrace/src/maaf_combined/maaf_initial_sampling_rchit.spv", { nullptr, RHI::ShaderStages::CLOSEST_HIT });
			GFX::GFXManager::get()->registerShaderModuleResource(maaf_initial_sampling_rmiss, "../Engine/Binaries/Runtime/spirv/RayTracing/RayTrace/src/maaf_combined/maaf_initial_sampling_rmiss.spv", { nullptr, RHI::ShaderStages::MISS });
			GFX::GFXManager::get()->registerShaderModuleResource(maaf_indirect_sampling_rchit, "../Engine/Binaries/Runtime/spirv/RayTracing/RayTrace/src/maaf_combined/maaf_indirect_sampling_rchit.spv", { nullptr, RHI::ShaderStages::CLOSEST_HIT });
			GFX::GFXManager::get()->registerShaderModuleResource(maaf_indirect_sampling_rmiss, "../Engine/Binaries/Runtime/spirv/RayTracing/RayTrace/src/maaf_combined/maaf_indirect_sampling_rmiss.spv", { nullptr, RHI::ShaderStages::MISS });
			GFX::GFXManager::get()->registerShaderModuleResource(maaf_shadow_dist_sampling_rahit, "../Engine/Binaries/Runtime/spirv/RayTracing/RayTrace/src/maaf_combined/maaf_shadow_dist_sampling_rahit.spv", { nullptr, RHI::ShaderStages::ANY_HIT });
			GFX::GFXManager::get()->registerShaderModuleResource(maaf_shadow_dist_sampling_rmiss, "../Engine/Binaries/Runtime/spirv/RayTracing/RayTrace/src/maaf_combined/maaf_shadow_dist_sampling_rmiss.spv", { nullptr, RHI::ShaderStages::MISS });
			GFX::GFXManager::get()->registerShaderModuleResource(maaf_shadow_sampling_rchit, "../Engine/Binaries/Runtime/spirv/RayTracing/RayTrace/src/maaf_combined/maaf_shadow_sampling_rchit.spv", { nullptr, RHI::ShaderStages::CLOSEST_HIT });
			GFX::GFXManager::get()->registerShaderModuleResource(maaf_shadow_sampling_rmiss, "../Engine/Binaries/Runtime/spirv/RayTracing/RayTrace/src/maaf_combined/maaf_shadow_sampling_rmiss.spv", { nullptr, RHI::ShaderStages::MISS });
			// Create rt pipeline
			for (int i = 0; i < 2; ++i) {
				raytracingPipeline[i] = rhiLayer->getDevice()->createRayTracingPipeline(RHI::RayTracingPipelineDescriptor{
					rtPipelineLayout, 3, RHI::SBTsDescriptor{
						RHI::SBTsDescriptor::RayGenerationSBT{{ Core::ResourceManager::get()->getResource<GFX::ShaderModule>(maaf_initial_sampling_rgen)->shaderModule.get() }},
						RHI::SBTsDescriptor::MissSBT{{
							{Core::ResourceManager::get()->getResource<GFX::ShaderModule>(maaf_initial_sampling_rmiss)->shaderModule.get()},
							{Core::ResourceManager::get()->getResource<GFX::ShaderModule>(maaf_shadow_dist_sampling_rmiss)->shaderModule.get()},
							{Core::ResourceManager::get()->getResource<GFX::ShaderModule>(maaf_indirect_sampling_rmiss)->shaderModule.get()},
							{Core::ResourceManager::get()->getResource<GFX::ShaderModule>(maaf_shadow_sampling_rmiss)->shaderModule.get()} }},
						RHI::SBTsDescriptor::HitGroupSBT{{
							{{Core::ResourceManager::get()->getResource<GFX::ShaderModule>(maaf_initial_sampling_rchit)->shaderModule.get()}, nullptr, nullptr},
							{nullptr, {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(maaf_shadow_dist_sampling_rahit)->shaderModule.get()}, nullptr},
							{{Core::ResourceManager::get()->getResource<GFX::ShaderModule>(maaf_indirect_sampling_rchit)->shaderModule.get()}, nullptr, nullptr},
							{{Core::ResourceManager::get()->getResource<GFX::ShaderModule>(maaf_shadow_sampling_rchit)->shaderModule.get()}, nullptr, nullptr} }}
					} });
			}
		}

		~MAAF_Continue_Sample_Pass() {
			raytracingPipeline[0] = nullptr;
			raytracingPipeline[1] = nullptr;
		}

		auto composeCommands(RHI::CommandEncoder* encoder, int index) noexcept -> void {
			static uint32_t batchIdx = 0;
			rtEncoder[index] = encoder->beginRayTracingPass(RHI::RayTracingPassDescriptor{});
			rtEncoder[index]->setPipeline(raytracingPipeline[index].get());
			rtEncoder[index]->setBindGroup(0, rtBindGroups[index], 0, 0);
			rtEncoder[index]->setBindGroup(1, camBindGroups[index], 0, 0);
			rtEncoder[index]->setBindGroup(2, bufferBindGroups[index], 0, 0);
			rtEncoder[index]->pushConstants(&batchIdx,
				(uint32_t)RHI::ShaderStages::RAYGEN
				| (uint32_t)RHI::ShaderStages::COMPUTE
				| (uint32_t)RHI::ShaderStages::CLOSEST_HIT
				| (uint32_t)RHI::ShaderStages::MISS
				| (uint32_t)RHI::ShaderStages::ANY_HIT,
				0, sizeof(uint32_t));
			rtEncoder[index]->traceRays(800, 600, 1);
			rtEncoder[index]->end();
			++batchIdx;
		}

		Core::GUID maaf_initial_sampling_rgen;
		Core::GUID maaf_initial_sampling_rchit;
		Core::GUID maaf_initial_sampling_rmiss;
		Core::GUID maaf_indirect_sampling_rchit;
		Core::GUID maaf_indirect_sampling_rmiss;
		Core::GUID maaf_shadow_dist_sampling_rahit;
		Core::GUID maaf_shadow_dist_sampling_rmiss;
		Core::GUID maaf_shadow_sampling_rchit;
		Core::GUID maaf_shadow_sampling_rmiss;

		std::unique_ptr<RHI::RayTracingPipeline> raytracingPipeline[2];
		std::unique_ptr<RHI::RayTracingPassEncoder> rtEncoder[2] = {};
		std::array<RHI::BindGroup*, 2> rtBindGroups = {};
		std::array<RHI::BindGroup*, 2> camBindGroups = {};
		std::array<RHI::BindGroup*, 2> bufferBindGroups = {};
	};

	
	export struct MAAF_Filter_Pass {
		MAAF_Filter_Pass(RHI::RHILayer* rhiLayer, RHI::PipelineLayout* slopleFilterPipelineLayout,
			std::array<RHI::BindGroup*, 2> const& bufferBindGroups)
			: slopleFilterPipelineLayout(slopleFilterPipelineLayout), bufferBindGroups(bufferBindGroups)
		{
			maaf_prefilter_x_comp = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();
			maaf_prefilter_y_comp = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();
			GFX::GFXManager::get()->registerShaderModuleResource(maaf_prefilter_x_comp,
				"../Engine/Binaries/Runtime/spirv/RayTracing/RayTrace/src/maaf_combined/maaf_filter_x_comp.spv",
				{ nullptr, RHI::ShaderStages::COMPUTE });
			GFX::GFXManager::get()->registerShaderModuleResource(maaf_prefilter_y_comp,
				"../Engine/Binaries/Runtime/spirv/RayTracing/RayTrace/src/maaf_combined/maaf_filter_y_comp.spv",
				{ nullptr, RHI::ShaderStages::COMPUTE });
			// create compute pipeline
			for (int i = 0; i < 2; ++i) {
				computePipelineX[i] = rhiLayer->getDevice()->createComputePipeline(RHI::ComputePipelineDescriptor{
					slopleFilterPipelineLayout,
					{Core::ResourceManager::get()->getResource<GFX::ShaderModule>(maaf_prefilter_x_comp)->shaderModule.get(), "main"}
					});
				computePipelineY[i] = rhiLayer->getDevice()->createComputePipeline(RHI::ComputePipelineDescriptor{
					slopleFilterPipelineLayout,
					{Core::ResourceManager::get()->getResource<GFX::ShaderModule>(maaf_prefilter_y_comp)->shaderModule.get(), "main"}
					});
			}

		}

		~MAAF_Filter_Pass() {
			for (int i = 0; i < 2; ++i) {
				computePipelineX[i] = nullptr;
				computePipelineY[i] = nullptr;
				compEncoderX[i] = nullptr;
				compEncoderY[i] = nullptr;
			}
		}

		auto composeCommands_x(RHI::CommandEncoder* encoder, int index) noexcept -> void {
			compEncoderX[index] = encoder->beginComputePass({});
			compEncoderX[index]->setPipeline(computePipelineX[index].get());
			compEncoderX[index]->setBindGroup(0, bufferBindGroups[index], 0, 0);
			compEncoderX[index]->dispatchWorkgroups((800 + 15) / 16, (600 + 15) / 16, 1);
			compEncoderX[index]->end();
		}

		auto composeCommands_y(RHI::CommandEncoder* encoder, int index) noexcept -> void {
			compEncoderY[index] = encoder->beginComputePass({});
			compEncoderY[index]->setPipeline(computePipelineY[index].get());
			compEncoderY[index]->setBindGroup(0, bufferBindGroups[index], 0, 0);
			compEncoderY[index]->dispatchWorkgroups((800 + 15) / 16, (600 + 15) / 16, 1);
			compEncoderY[index]->end();
		}

		Core::GUID maaf_prefilter_x_comp;
		Core::GUID maaf_prefilter_y_comp;
		RHI::PipelineLayout* slopleFilterPipelineLayout;
		std::unique_ptr<RHI::ComputePipeline> computePipelineX[2];
		std::unique_ptr<RHI::ComputePipeline> computePipelineY[2];
		std::array<RHI::BindGroup*, 2> bufferBindGroups;
		std::unique_ptr<RHI::ComputePassEncoder> compEncoderX[2] = {};
		std::unique_ptr<RHI::ComputePassEncoder> compEncoderY[2] = {};
	};

	export struct MAAF_Display_Pass {
		MAAF_Display_Pass(RHI::RHILayer* rhiLayer, RHI::PipelineLayout* displayPipelineLayout,
			std::array<RHI::BindGroup*, 2> const& bufferBindGroups)
			: bufferBindGroups(bufferBindGroups) {
			// get the shader
			aaf_result_display_comp = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();
			GFX::GFXManager::get()->registerShaderModuleResource(aaf_result_display_comp,
				"../Engine/Binaries/Runtime/spirv/RayTracing/RayTrace/src/maaf_combined/maaf_display_pass_comp.spv",
				{ nullptr, RHI::ShaderStages::COMPUTE });
			// create compute pipeline
			for (int i = 0; i < 2; ++i)
				computePipeline[i] = rhiLayer->getDevice()->createComputePipeline(RHI::ComputePipelineDescriptor{
					displayPipelineLayout,
					{Core::ResourceManager::get()->getResource<GFX::ShaderModule>(aaf_result_display_comp)->shaderModule.get(), "main"}
					});
		}

		~MAAF_Display_Pass() {
			computePipeline[0] = nullptr;
			computePipeline[1] = nullptr;
		}

		auto composeCommands(RHI::CommandEncoder* encoder, int index) noexcept -> void {
			compEncoder[index] = encoder->beginComputePass({});
			compEncoder[index]->setPipeline(computePipeline[index].get());
			compEncoder[index]->setBindGroup(0, bufferBindGroups[index], 0, 0);
			compEncoder[index]->dispatchWorkgroups((800 + 15) / 16, (600 + 15) / 16, 1);
			compEncoder[index]->end();
		}
		Core::GUID aaf_result_display_comp;
		RHI::PipelineLayout* displayPipelineLayout;
		std::unique_ptr<RHI::ComputePipeline> computePipeline[2];
		std::array<RHI::BindGroup*, 2> bufferBindGroups;
		std::unique_ptr<RHI::ComputePassEncoder> compEncoder[2] = {};
	};

	struct PushConstantRay {
		Math::vec4 clearColor;
		Math::vec3 lightPosition;
		float lightIntensity;
		int   lightType;
	};

	export struct MAAF_Pipeline {
		MAAF_Pipeline(RHI::RHILayer* rhiLayer, GFX::ASGroup* asgroup, Core::GUID rtTarget,
			RHI::BindGroupLayout* cameraBindGroupLayout, std::array<RHI::BindGroup*, 2> const& camBindGroup)
			: rtTarget(rtTarget)
		{
			// allocate buffers
			defocusSlopeBuffer				= Core::ResourceManager::get()->requestRuntimeGUID<GFX::Buffer>();
			directSlopeBuffer				= Core::ResourceManager::get()->requestRuntimeGUID<GFX::Buffer>();
			indirectSlopeBuffer				= Core::ResourceManager::get()->requestRuntimeGUID<GFX::Buffer>();
			worldPositionBuffer				= Core::ResourceManager::get()->requestRuntimeGUID<GFX::Buffer>();
			worldNormalBuffer				= Core::ResourceManager::get()->requestRuntimeGUID<GFX::Buffer>();
			useFilterBuffer					= Core::ResourceManager::get()->requestRuntimeGUID<GFX::Buffer>();
			MAAFParametersBuffer			= Core::ResourceManager::get()->requestRuntimeGUID<GFX::Buffer>();
			MAAFIntermediateDirectBuffer	= Core::ResourceManager::get()->requestRuntimeGUID<GFX::Buffer>();
			MAAFIntermediateIndirectBuffer	= Core::ResourceManager::get()->requestRuntimeGUID<GFX::Buffer>();

			defocusSlopeTmpBuffer			= Core::ResourceManager::get()->requestRuntimeGUID<GFX::Buffer>();
			directSlopeTmpBuffer			= Core::ResourceManager::get()->requestRuntimeGUID<GFX::Buffer>();
			indirectSlopeTmpBuffer			= Core::ResourceManager::get()->requestRuntimeGUID<GFX::Buffer>();
			worldPositionTmpBuffer			= Core::ResourceManager::get()->requestRuntimeGUID<GFX::Buffer>();
			worldNormalTmpBuffer			= Core::ResourceManager::get()->requestRuntimeGUID<GFX::Buffer>();
			useFilterTmpBuffer				= Core::ResourceManager::get()->requestRuntimeGUID<GFX::Buffer>();
			MAAFIntermediateDirectTmpBuffer = Core::ResourceManager::get()->requestRuntimeGUID<GFX::Buffer>();
			MAAFIntermediateIndirectTmpBuffer = Core::ResourceManager::get()->requestRuntimeGUID<GFX::Buffer>();

			GFX::GFXManager::get()->registerBufferResource(defocusSlopeBuffer, RHI::BufferDescriptor{
				width * height * sizeof(float) * 2, // pixel size * sizeof(vec2)
				(uint32_t)RHI::BufferUsage::STORAGE });
			GFX::GFXManager::get()->registerBufferResource(directSlopeBuffer, RHI::BufferDescriptor{
				width * height * sizeof(float) * 2, // pixel size * sizeof(vec2)
				(uint32_t)RHI::BufferUsage::STORAGE });
			GFX::GFXManager::get()->registerBufferResource(indirectSlopeBuffer, RHI::BufferDescriptor{
				width * height * sizeof(float) * 2, // pixel size * sizeof(vec2)
				(uint32_t)RHI::BufferUsage::STORAGE });
			GFX::GFXManager::get()->registerBufferResource(worldPositionBuffer, RHI::BufferDescriptor{
				width * height * sizeof(float) * 4, // pixel size * sizeof(align(vec3))
				(uint32_t)RHI::BufferUsage::STORAGE });
			GFX::GFXManager::get()->registerBufferResource(worldNormalBuffer, RHI::BufferDescriptor{
				width * height * sizeof(float) * 3, // pixel size * sizeof(align(vec3))
				(uint32_t)RHI::BufferUsage::STORAGE });
			GFX::GFXManager::get()->registerBufferResource(useFilterBuffer, RHI::BufferDescriptor{
				width * height * sizeof(unsigned int) * 1, // pixel size * sizeof(align(unsigned int))
				(uint32_t)RHI::BufferUsage::STORAGE });
			GFX::GFXManager::get()->registerBufferResource(MAAFParametersBuffer, RHI::BufferDescriptor{
				width * height * sizeof(float) * 2 * 20, // pixel size * sizeof(float3)
				(uint32_t)RHI::BufferUsage::STORAGE });
			GFX::GFXManager::get()->registerBufferResource(MAAFIntermediateDirectBuffer, RHI::BufferDescriptor{
				width * height * sizeof(float) * 4 * 25, // pixel size * sizeof(align(vec3))
				(uint32_t)RHI::BufferUsage::STORAGE });
			GFX::GFXManager::get()->registerBufferResource(MAAFIntermediateIndirectBuffer, RHI::BufferDescriptor{
				width * height * sizeof(float) * 4 * 25, // pixel size * sizeof(align(vec3))
				(uint32_t)RHI::BufferUsage::STORAGE });
			GFX::GFXManager::get()->registerBufferResource(defocusSlopeTmpBuffer, RHI::BufferDescriptor{
				width * height * sizeof(float) * 2, // pixel size * sizeof(align(vec3))
				(uint32_t)RHI::BufferUsage::STORAGE });
			GFX::GFXManager::get()->registerBufferResource(directSlopeTmpBuffer, RHI::BufferDescriptor{
				width * height * sizeof(float) * 2, // pixel size * sizeof(align(vec3))
				(uint32_t)RHI::BufferUsage::STORAGE });;
			GFX::GFXManager::get()->registerBufferResource(indirectSlopeTmpBuffer, RHI::BufferDescriptor{
				width * height * sizeof(float) * 2, // pixel size * sizeof(align(vec3))
				(uint32_t)RHI::BufferUsage::STORAGE });
			GFX::GFXManager::get()->registerBufferResource(worldNormalTmpBuffer, RHI::BufferDescriptor{
				width * height * sizeof(float) * 3, // pixel size * sizeof(uint)
				(uint32_t)RHI::BufferUsage::STORAGE });
			GFX::GFXManager::get()->registerBufferResource(worldPositionTmpBuffer, RHI::BufferDescriptor{
				width * height * sizeof(float) * 4, // pixel size * sizeof(uint)
				(uint32_t)RHI::BufferUsage::STORAGE });
			GFX::GFXManager::get()->registerBufferResource(useFilterTmpBuffer, RHI::BufferDescriptor{
				width * height * sizeof(unsigned int) * 1, // pixel size * sizeof(uint)
				(uint32_t)RHI::BufferUsage::STORAGE });
			GFX::GFXManager::get()->registerBufferResource(MAAFIntermediateDirectTmpBuffer, RHI::BufferDescriptor{
				width * height * sizeof(float) * 4 * 25, // pixel size * sizeof(uint)
				(uint32_t)RHI::BufferUsage::STORAGE });
			GFX::GFXManager::get()->registerBufferResource(MAAFIntermediateIndirectTmpBuffer, RHI::BufferDescriptor{
				width * height * sizeof(float) * 4 * 25, // pixel size * sizeof(uint)
				(uint32_t)RHI::BufferUsage::STORAGE });

			GFX::Buffer* pdefocusSlopeBuffer			 = Core::ResourceManager::get()->getResource<GFX::Buffer>(defocusSlopeBuffer);
			GFX::Buffer* pdirectSlopeBuffer				 = Core::ResourceManager::get()->getResource<GFX::Buffer>(directSlopeBuffer);	
			GFX::Buffer* pindirectSlopeBuffer			 = Core::ResourceManager::get()->getResource<GFX::Buffer>(indirectSlopeBuffer);
			GFX::Buffer* pworldPositionBuffer			 = Core::ResourceManager::get()->getResource<GFX::Buffer>(worldPositionBuffer);
			GFX::Buffer* pworldNormalBuffer				 = Core::ResourceManager::get()->getResource<GFX::Buffer>(worldNormalBuffer);	
			GFX::Buffer* puseFilterBuffer				 = Core::ResourceManager::get()->getResource<GFX::Buffer>(useFilterBuffer);
			GFX::Buffer* pMAAFParametersBuffer			 = Core::ResourceManager::get()->getResource<GFX::Buffer>(MAAFParametersBuffer);
			GFX::Buffer* pMAAFIntermediateDirectBuffer	 = Core::ResourceManager::get()->getResource<GFX::Buffer>(MAAFIntermediateDirectBuffer);
			GFX::Buffer* pMAAFIntermediateIndirectBuffer = Core::ResourceManager::get()->getResource<GFX::Buffer>(MAAFIntermediateIndirectBuffer);

			GFX::Buffer* pdefocusSlopeTmpBuffer			 = Core::ResourceManager::get()->getResource<GFX::Buffer>(defocusSlopeTmpBuffer);	
			GFX::Buffer* pdirectSlopeTmpBuffer			 = Core::ResourceManager::get()->getResource<GFX::Buffer>(directSlopeTmpBuffer);
			GFX::Buffer* pindirectSlopeTmpBuffer		 = Core::ResourceManager::get()->getResource<GFX::Buffer>(indirectSlopeTmpBuffer);
			GFX::Buffer* pworldPositionTmpBuffer		 = Core::ResourceManager::get()->getResource<GFX::Buffer>(worldPositionTmpBuffer);
			GFX::Buffer* pworldNormalTmpBuffer			 = Core::ResourceManager::get()->getResource<GFX::Buffer>(worldNormalTmpBuffer);
			GFX::Buffer* puseFilterTmpBuffer			 = Core::ResourceManager::get()->getResource<GFX::Buffer>(useFilterTmpBuffer);
			GFX::Buffer* pIntermIndirectTmpBuffer		 = Core::ResourceManager::get()->getResource<GFX::Buffer>(MAAFIntermediateDirectTmpBuffer);
			GFX::Buffer* pIntermDirectTmpBuffer			 = Core::ResourceManager::get()->getResource<GFX::Buffer>(MAAFIntermediateIndirectTmpBuffer);

			// create bind group layout - buffers + rt
			RHI::ShaderStagesFlags stages =
				(uint32_t)RHI::ShaderStages::RAYGEN
				| (uint32_t)RHI::ShaderStages::CLOSEST_HIT
				| (uint32_t)RHI::ShaderStages::MISS
				| (uint32_t)RHI::ShaderStages::ANY_HIT
				| (uint32_t)RHI::ShaderStages::COMPUTE;
			buffersBindGroupLayout = rhiLayer->getDevice()->createBindGroupLayout(
				RHI::BindGroupLayoutDescriptor{ {
					RHI::BindGroupLayoutEntry{0, stages, RHI::BufferBindingLayout{RHI::BufferBindingType::STORAGE}},	// slope buffer
					RHI::BindGroupLayoutEntry{1, stages, RHI::BufferBindingLayout{RHI::BufferBindingType::STORAGE}},	// vis buffer
					RHI::BindGroupLayoutEntry{2, stages, RHI::BufferBindingLayout{RHI::BufferBindingType::STORAGE}},	// proj dist buffer
					RHI::BindGroupLayoutEntry{3, stages, RHI::BufferBindingLayout{RHI::BufferBindingType::STORAGE}},	// world pos buffer
					RHI::BindGroupLayoutEntry{4, stages, RHI::BufferBindingLayout{RHI::BufferBindingType::STORAGE}},	// world normal buffer
					RHI::BindGroupLayoutEntry{5, stages, RHI::BufferBindingLayout{RHI::BufferBindingType::STORAGE}},	// brdf buffer
					RHI::BindGroupLayoutEntry{6, stages, RHI::BufferBindingLayout{RHI::BufferBindingType::STORAGE}},	// use filter buffer
					RHI::BindGroupLayoutEntry{7, stages, RHI::BufferBindingLayout{RHI::BufferBindingType::STORAGE}},	// use filter buffer
					RHI::BindGroupLayoutEntry{8, stages, RHI::BufferBindingLayout{RHI::BufferBindingType::STORAGE}},	// use filter buffer
					RHI::BindGroupLayoutEntry{9, stages, RHI::StorageTextureBindingLayout{}},							// render target storage image
					RHI::BindGroupLayoutEntry{10, stages, RHI::BufferBindingLayout{RHI::BufferBindingType::STORAGE}},	// use filter buffer
					RHI::BindGroupLayoutEntry{11, stages, RHI::BufferBindingLayout{RHI::BufferBindingType::STORAGE}},	// use filter buffer
					RHI::BindGroupLayoutEntry{12, stages, RHI::BufferBindingLayout{RHI::BufferBindingType::STORAGE}},	// use filter buffer
					RHI::BindGroupLayoutEntry{13, stages, RHI::BufferBindingLayout{RHI::BufferBindingType::STORAGE}},	// use filter buffer
					RHI::BindGroupLayoutEntry{14, stages, RHI::BufferBindingLayout{RHI::BufferBindingType::STORAGE}},	// use filter buffer
					RHI::BindGroupLayoutEntry{15, stages, RHI::BufferBindingLayout{RHI::BufferBindingType::STORAGE}},	// use filter buffer
					RHI::BindGroupLayoutEntry{16, stages, RHI::BufferBindingLayout{RHI::BufferBindingType::STORAGE}},	// use filter buffer
					RHI::BindGroupLayoutEntry{17, stages, RHI::BufferBindingLayout{RHI::BufferBindingType::STORAGE}},	// use filter buffer
				} });
			// create bind groups
			for (int i = 0; i < 2; ++i) {
				bufferBindGroup[i] = rhiLayer->getDevice()->createBindGroup(RHI::BindGroupDescriptor{
					buffersBindGroupLayout.get(),
					std::vector<RHI::BindGroupEntry>{
						{0,RHI::BindingResource{{pdefocusSlopeBuffer			->buffer.get(), 0, pdefocusSlopeBuffer			  ->buffer->size()}}},
						{1,RHI::BindingResource{{pdirectSlopeBuffer				->buffer.get(), 0, pdirectSlopeBuffer				->buffer->size()}}}, 
						{2,RHI::BindingResource{{pindirectSlopeBuffer			->buffer.get(), 0, pindirectSlopeBuffer			  ->buffer->size()}}},
						{3,RHI::BindingResource{{pworldPositionBuffer			->buffer.get(), 0, pworldPositionBuffer			  ->buffer->size()}}},
						{4,RHI::BindingResource{{pworldNormalBuffer				->buffer.get(), 0, pworldNormalBuffer				->buffer->size()}}}, 
						{5,RHI::BindingResource{{puseFilterBuffer				->buffer.get(), 0, puseFilterBuffer				  ->buffer->size()}}},
						{6,RHI::BindingResource{{pMAAFParametersBuffer			->buffer.get(), 0, pMAAFParametersBuffer			->buffer->size()}}}, 
						{7,RHI::BindingResource{{pMAAFIntermediateDirectBuffer	->buffer.get(), 0, pMAAFIntermediateDirectBuffer	->buffer->size()}}}, 
						{8,RHI::BindingResource{{pMAAFIntermediateIndirectBuffer->buffer.get(), 0, pMAAFIntermediateIndirectBuffer->buffer->size()}}}, 
						{9,RHI::BindingResource{Core::ResourceManager::get()->getResource<GFX::Texture>(rtTarget)->originalView.get()}},
						{10,RHI::BindingResource{{pdefocusSlopeTmpBuffer	->buffer.get(), 0, pdefocusSlopeTmpBuffer->buffer->size()}}},
						{11,RHI::BindingResource{{pdirectSlopeTmpBuffer->buffer.get(), 0, pdirectSlopeTmpBuffer->buffer->size()}}},
						{12,RHI::BindingResource{{pindirectSlopeTmpBuffer->buffer.get(), 0, pindirectSlopeTmpBuffer->buffer->size()}}},
						{13,RHI::BindingResource{{pworldPositionTmpBuffer->buffer.get(), 0, pworldPositionTmpBuffer->buffer->size()}}},
						{14,RHI::BindingResource{{pworldNormalTmpBuffer->buffer.get(), 0, pworldNormalTmpBuffer->buffer->size()}}},
						{15,RHI::BindingResource{{puseFilterTmpBuffer->buffer.get(), 0, puseFilterTmpBuffer->buffer->size()}}},
						{16,RHI::BindingResource{{pIntermIndirectTmpBuffer->buffer.get(), 0, pIntermIndirectTmpBuffer->buffer->size()}}},
						{17,RHI::BindingResource{{pIntermDirectTmpBuffer->buffer.get(), 0, pIntermDirectTmpBuffer->buffer->size()}}},
				} });
			}
			// create bind group layout
			rtBindGroupLayout = rhiLayer->getDevice()->createBindGroupLayout(
				RHI::BindGroupLayoutDescriptor{ {
					RHI::BindGroupLayoutEntry{0, stages, RHI::AccelerationStructureBindingLayout{}},					// tlas
					RHI::BindGroupLayoutEntry{1, stages, RHI::StorageTextureBindingLayout{}},							// output image
					RHI::BindGroupLayoutEntry{2, stages, RHI::BufferBindingLayout{RHI::BufferBindingType::STORAGE}},	// vertex buffer
					RHI::BindGroupLayoutEntry{3, stages, RHI::BufferBindingLayout{RHI::BufferBindingType::STORAGE}},	// index buffer
					RHI::BindGroupLayoutEntry{4, stages, RHI::BufferBindingLayout{RHI::BufferBindingType::STORAGE}},	// index buffer
					} });
			// create bind groups
			for (int i = 0; i < 2; ++i) {
				rtBindGroup[i] = rhiLayer->getDevice()->createBindGroup(RHI::BindGroupDescriptor{
					rtBindGroupLayout.get(),
					std::vector<RHI::BindGroupEntry>{
						{0,RHI::BindingResource{asgroup->tlas.get()}},
						{1,RHI::BindingResource{Core::ResourceManager::get()->getResource<GFX::Texture>(rtTarget)->originalView.get()}},
						{2,RHI::BindingResource{{asgroup->vertexBufferArray.get(), 0, asgroup->vertexBufferArray->size()}}},
						{3,RHI::BindingResource{{asgroup->indexBufferArray.get(), 0, asgroup->indexBufferArray->size()}}},
						{4,RHI::BindingResource{{asgroup->GeometryInfoBuffer.get(), 0, asgroup->GeometryInfoBuffer->size()}}},
				} });
			}
			pipelineLayout = rhiLayer->getDevice()->createPipelineLayout(RHI::PipelineLayoutDescriptor{
				{ {stages, 0, sizeof(PushConstantRay)}},
				{ rtBindGroupLayout.get(), cameraBindGroupLayout, buffersBindGroupLayout.get() }
				});

			displayPipelineLayout = rhiLayer->getDevice()->createPipelineLayout(RHI::PipelineLayoutDescriptor{
				{}, { buffersBindGroupLayout.get() } });

			initial_sampling_pass = std::make_unique<MAAF_Initial_Sample_Pass>(rhiLayer, pipelineLayout.get(),
				std::array<RHI::BindGroup*, 2>{rtBindGroup[0].get(), rtBindGroup[1].get()},
				std::array<RHI::BindGroup*, 2>{camBindGroup[0], camBindGroup[1]},
				std::array<RHI::BindGroup*, 2>{bufferBindGroup[0].get(), bufferBindGroup[1].get()});
			prefilter_pass = std::make_unique<MAAF_Prefilter_Pass>(rhiLayer, displayPipelineLayout.get(),
				std::array<RHI::BindGroup*, 2>{bufferBindGroup[0].get(), bufferBindGroup[1].get()});
			continue_sampling_pass = std::make_unique<MAAF_Continue_Sample_Pass>(rhiLayer, pipelineLayout.get(),
				std::array<RHI::BindGroup*, 2>{rtBindGroup[0].get(), rtBindGroup[1].get()},
				std::array<RHI::BindGroup*, 2>{camBindGroup[0], camBindGroup[1]},
				std::array<RHI::BindGroup*, 2>{bufferBindGroup[0].get(), bufferBindGroup[1].get()});
			filter_pass = std::make_unique<MAAF_Filter_Pass>(rhiLayer, displayPipelineLayout.get(),
				std::array<RHI::BindGroup*, 2>{bufferBindGroup[0].get(), bufferBindGroup[1].get()});
			result_display_pass = std::make_unique<MAAF_Display_Pass>(rhiLayer, displayPipelineLayout.get(),
				std::array<RHI::BindGroup*, 2>{bufferBindGroup[0].get(), bufferBindGroup[1].get()});

		}

		~MAAF_Pipeline() {
			initial_sampling_pass = nullptr;
			prefilter_pass = nullptr;
			filter_pass = nullptr;
			continue_sampling_pass = nullptr;
			result_display_pass = nullptr;
			pipelineLayout = nullptr;
			displayPipelineLayout = nullptr;
			rtBindGroup[0] = nullptr;
			rtBindGroup[1] = nullptr;
			bufferBindGroup[0] = nullptr;
			bufferBindGroup[1] = nullptr;
			rtBindGroupLayout = nullptr;
			buffersBindGroupLayout = nullptr;
		}

		auto composeCommands(RHI::CommandEncoder* encoder, int index) noexcept -> void {
			encoder->pipelineBarrier(RHI::BarrierDescriptor{
				(uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT,
				(uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT,
				(uint32_t)RHI::DependencyType::NONE,
				{}, {},
				{ RHI::TextureMemoryBarrierDescriptor{
					Core::ResourceManager::get()->getResource<GFX::Texture>(rtTarget)->texture.get(),
					RHI::ImageSubresourceRange{(uint32_t)RHI::TextureAspect::COLOR_BIT, 0,1,0,1},
					(uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT,
					(uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT,
					RHI::TextureLayout::SHADER_READ_ONLY_OPTIMAL,
					RHI::TextureLayout::GENERAL
				}}
				});

			RHI::BarrierDescriptor bufferRT2Comp{
				(uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR,
				(uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT,
				(uint32_t)RHI::DependencyType::NONE,
				{}, { RHI::BufferMemoryBarrierDescriptor{
					nullptr,
					(uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT,
					(uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT,
				} }, {}
			};

			RHI::BarrierDescriptor bufferComp2Comp_R2W{
				(uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT,
				(uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT,
				(uint32_t)RHI::DependencyType::NONE,
				{}, { RHI::BufferMemoryBarrierDescriptor{
					nullptr,
					(uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT,
					(uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT,
				} }, {}
			};
			RHI::BarrierDescriptor bufferComp2Comp_W2R{
				(uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT,
				(uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT,
				(uint32_t)RHI::DependencyType::NONE,
				{}, { RHI::BufferMemoryBarrierDescriptor{
					nullptr,
					(uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT,
					(uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT,
				} }, {}
			};

			RHI::BarrierDescriptor bufferRT2RT_W2RW{
				(uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR,
				(uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR,
				(uint32_t)RHI::DependencyType::NONE,
				{}, { RHI::BufferMemoryBarrierDescriptor{
					nullptr,
					(uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT,
					(uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT | (uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT,
				} }, {}
			};

			RHI::BarrierDescriptor bufferCOMP2RT_RW2RW{
				(uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT,
				(uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR,
				(uint32_t)RHI::DependencyType::NONE,
				{}, { RHI::BufferMemoryBarrierDescriptor{
					nullptr,
					(uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT | (uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT,
					(uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT | (uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT,
				} }, {}
			};

			{
				bufferCOMP2RT_RW2RW.bufferMemoryBarriers[0].buffer = Core::ResourceManager::get()->getResource<GFX::Buffer>(useFilterBuffer)->buffer.get();
				encoder->pipelineBarrier(bufferCOMP2RT_RW2RW);
			}
			initial_sampling_pass->composeCommands(encoder, index);

			// pre filtering
			{
				// ready buffer from RT
				bufferRT2Comp.bufferMemoryBarriers[0].buffer = Core::ResourceManager::get()->getResource<GFX::Buffer>(defocusSlopeBuffer)->buffer.get();
				encoder->pipelineBarrier(bufferRT2Comp);
				bufferRT2Comp.bufferMemoryBarriers[0].buffer = Core::ResourceManager::get()->getResource<GFX::Buffer>(directSlopeBuffer)->buffer.get();
				encoder->pipelineBarrier(bufferRT2Comp);
				bufferRT2Comp.bufferMemoryBarriers[0].buffer = Core::ResourceManager::get()->getResource<GFX::Buffer>(indirectSlopeBuffer)->buffer.get();
				encoder->pipelineBarrier(bufferRT2Comp);
				bufferRT2Comp.bufferMemoryBarriers[0].buffer = Core::ResourceManager::get()->getResource<GFX::Buffer>(worldPositionBuffer)->buffer.get();
				encoder->pipelineBarrier(bufferRT2Comp);
				bufferRT2Comp.bufferMemoryBarriers[0].buffer = Core::ResourceManager::get()->getResource<GFX::Buffer>(worldNormalBuffer)->buffer.get();
				encoder->pipelineBarrier(bufferRT2Comp);
				bufferRT2Comp.bufferMemoryBarriers[0].buffer = Core::ResourceManager::get()->getResource<GFX::Buffer>(useFilterBuffer)->buffer.get();
				encoder->pipelineBarrier(bufferRT2Comp);

				bufferComp2Comp_R2W.bufferMemoryBarriers[0].buffer = Core::ResourceManager::get()->getResource<GFX::Buffer>(defocusSlopeTmpBuffer)->buffer.get();
				encoder->pipelineBarrier(bufferComp2Comp_R2W);
				bufferComp2Comp_R2W.bufferMemoryBarriers[0].buffer = Core::ResourceManager::get()->getResource<GFX::Buffer>(directSlopeTmpBuffer)->buffer.get();
				encoder->pipelineBarrier(bufferComp2Comp_R2W);
				bufferComp2Comp_R2W.bufferMemoryBarriers[0].buffer = Core::ResourceManager::get()->getResource<GFX::Buffer>(indirectSlopeTmpBuffer)->buffer.get();
				encoder->pipelineBarrier(bufferComp2Comp_R2W);
				bufferComp2Comp_R2W.bufferMemoryBarriers[0].buffer = Core::ResourceManager::get()->getResource<GFX::Buffer>(worldPositionTmpBuffer)->buffer.get();
				encoder->pipelineBarrier(bufferComp2Comp_R2W);
				bufferComp2Comp_R2W.bufferMemoryBarriers[0].buffer = Core::ResourceManager::get()->getResource<GFX::Buffer>(worldNormalTmpBuffer)->buffer.get();
				encoder->pipelineBarrier(bufferComp2Comp_R2W);
				bufferComp2Comp_R2W.bufferMemoryBarriers[0].buffer = Core::ResourceManager::get()->getResource<GFX::Buffer>(useFilterTmpBuffer)->buffer.get();
				encoder->pipelineBarrier(bufferComp2Comp_R2W);

				prefilter_pass->composeCommands_x(encoder, index);

				bufferComp2Comp_W2R.bufferMemoryBarriers[0].buffer = Core::ResourceManager::get()->getResource<GFX::Buffer>(defocusSlopeTmpBuffer)->buffer.get();
				encoder->pipelineBarrier(bufferComp2Comp_W2R);
				bufferComp2Comp_W2R.bufferMemoryBarriers[0].buffer = Core::ResourceManager::get()->getResource<GFX::Buffer>(directSlopeTmpBuffer)->buffer.get();
				encoder->pipelineBarrier(bufferComp2Comp_W2R);
				bufferComp2Comp_W2R.bufferMemoryBarriers[0].buffer = Core::ResourceManager::get()->getResource<GFX::Buffer>(indirectSlopeTmpBuffer)->buffer.get();
				encoder->pipelineBarrier(bufferComp2Comp_W2R);
				bufferComp2Comp_W2R.bufferMemoryBarriers[0].buffer = Core::ResourceManager::get()->getResource<GFX::Buffer>(worldPositionTmpBuffer)->buffer.get();
				encoder->pipelineBarrier(bufferComp2Comp_W2R);
				bufferComp2Comp_W2R.bufferMemoryBarriers[0].buffer = Core::ResourceManager::get()->getResource<GFX::Buffer>(worldNormalTmpBuffer)->buffer.get();
				encoder->pipelineBarrier(bufferComp2Comp_W2R);
				bufferComp2Comp_W2R.bufferMemoryBarriers[0].buffer = Core::ResourceManager::get()->getResource<GFX::Buffer>(useFilterTmpBuffer)->buffer.get();
				encoder->pipelineBarrier(bufferComp2Comp_W2R);

				bufferComp2Comp_R2W.bufferMemoryBarriers[0].buffer = Core::ResourceManager::get()->getResource<GFX::Buffer>(defocusSlopeBuffer)->buffer.get();
				encoder->pipelineBarrier(bufferComp2Comp_R2W);
				bufferComp2Comp_R2W.bufferMemoryBarriers[0].buffer = Core::ResourceManager::get()->getResource<GFX::Buffer>(directSlopeBuffer)->buffer.get();
				encoder->pipelineBarrier(bufferComp2Comp_R2W);
				bufferComp2Comp_R2W.bufferMemoryBarriers[0].buffer = Core::ResourceManager::get()->getResource<GFX::Buffer>(indirectSlopeBuffer)->buffer.get();
				encoder->pipelineBarrier(bufferComp2Comp_R2W);
				bufferComp2Comp_R2W.bufferMemoryBarriers[0].buffer = Core::ResourceManager::get()->getResource<GFX::Buffer>(worldPositionBuffer)->buffer.get();
				encoder->pipelineBarrier(bufferComp2Comp_R2W);
				bufferComp2Comp_R2W.bufferMemoryBarriers[0].buffer = Core::ResourceManager::get()->getResource<GFX::Buffer>(worldNormalBuffer)->buffer.get();
				encoder->pipelineBarrier(bufferComp2Comp_R2W);
				bufferComp2Comp_R2W.bufferMemoryBarriers[0].buffer = Core::ResourceManager::get()->getResource<GFX::Buffer>(useFilterBuffer)->buffer.get();
				encoder->pipelineBarrier(bufferComp2Comp_R2W);

				prefilter_pass->composeCommands_y(encoder, index);

				bufferCOMP2RT_RW2RW.bufferMemoryBarriers[0].buffer = Core::ResourceManager::get()->getResource<GFX::Buffer>(defocusSlopeBuffer)->buffer.get();
				encoder->pipelineBarrier(bufferCOMP2RT_RW2RW);
				bufferCOMP2RT_RW2RW.bufferMemoryBarriers[0].buffer = Core::ResourceManager::get()->getResource<GFX::Buffer>(directSlopeBuffer)->buffer.get();
				encoder->pipelineBarrier(bufferCOMP2RT_RW2RW);
				bufferCOMP2RT_RW2RW.bufferMemoryBarriers[0].buffer = Core::ResourceManager::get()->getResource<GFX::Buffer>(indirectSlopeBuffer)->buffer.get();
				encoder->pipelineBarrier(bufferCOMP2RT_RW2RW);
				bufferCOMP2RT_RW2RW.bufferMemoryBarriers[0].buffer = Core::ResourceManager::get()->getResource<GFX::Buffer>(worldPositionBuffer)->buffer.get();
				encoder->pipelineBarrier(bufferCOMP2RT_RW2RW);
				bufferCOMP2RT_RW2RW.bufferMemoryBarriers[0].buffer = Core::ResourceManager::get()->getResource<GFX::Buffer>(worldNormalBuffer)->buffer.get();
				encoder->pipelineBarrier(bufferCOMP2RT_RW2RW);
				bufferCOMP2RT_RW2RW.bufferMemoryBarriers[0].buffer = Core::ResourceManager::get()->getResource<GFX::Buffer>(useFilterBuffer)->buffer.get();
				encoder->pipelineBarrier(bufferCOMP2RT_RW2RW);
				bufferCOMP2RT_RW2RW.bufferMemoryBarriers[0].buffer = Core::ResourceManager::get()->getResource<GFX::Buffer>(MAAFParametersBuffer)->buffer.get();
				encoder->pipelineBarrier(bufferCOMP2RT_RW2RW);

				bufferRT2RT_W2RW.bufferMemoryBarriers[0].buffer = Core::ResourceManager::get()->getResource<GFX::Buffer>(MAAFIntermediateDirectBuffer)->buffer.get();
				encoder->pipelineBarrier(bufferRT2RT_W2RW);
				bufferRT2RT_W2RW.bufferMemoryBarriers[0].buffer = Core::ResourceManager::get()->getResource<GFX::Buffer>(MAAFIntermediateIndirectBuffer)->buffer.get();
				encoder->pipelineBarrier(bufferRT2RT_W2RW);

				// continue sample
				continue_sampling_pass->composeCommands(encoder, index);

				bufferRT2Comp.bufferMemoryBarriers[0].buffer = Core::ResourceManager::get()->getResource<GFX::Buffer>(MAAFParametersBuffer)->buffer.get();
				encoder->pipelineBarrier(bufferRT2Comp);
				bufferRT2Comp.bufferMemoryBarriers[0].buffer = Core::ResourceManager::get()->getResource<GFX::Buffer>(MAAFIntermediateDirectBuffer)->buffer.get();
				encoder->pipelineBarrier(bufferRT2Comp);
				bufferRT2Comp.bufferMemoryBarriers[0].buffer = Core::ResourceManager::get()->getResource<GFX::Buffer>(MAAFIntermediateIndirectBuffer)->buffer.get();
				encoder->pipelineBarrier(bufferRT2Comp);
			}
			// filter
			{
				bufferComp2Comp_R2W.bufferMemoryBarriers[0].buffer = Core::ResourceManager::get()->getResource<GFX::Buffer>(MAAFIntermediateDirectTmpBuffer)->buffer.get();
				encoder->pipelineBarrier(bufferComp2Comp_R2W);
				bufferComp2Comp_R2W.bufferMemoryBarriers[0].buffer = Core::ResourceManager::get()->getResource<GFX::Buffer>(MAAFIntermediateIndirectTmpBuffer)->buffer.get();
				encoder->pipelineBarrier(bufferComp2Comp_R2W);

				filter_pass->composeCommands_x(encoder, index);

				bufferComp2Comp_W2R.bufferMemoryBarriers[0].buffer = Core::ResourceManager::get()->getResource<GFX::Buffer>(MAAFIntermediateDirectTmpBuffer)->buffer.get();
				encoder->pipelineBarrier(bufferComp2Comp_W2R);
				bufferComp2Comp_W2R.bufferMemoryBarriers[0].buffer = Core::ResourceManager::get()->getResource<GFX::Buffer>(MAAFIntermediateIndirectTmpBuffer)->buffer.get();
				encoder->pipelineBarrier(bufferComp2Comp_W2R);
				bufferComp2Comp_R2W.bufferMemoryBarriers[0].buffer = Core::ResourceManager::get()->getResource<GFX::Buffer>(MAAFIntermediateDirectBuffer)->buffer.get();
				encoder->pipelineBarrier(bufferComp2Comp_R2W);
				bufferComp2Comp_R2W.bufferMemoryBarriers[0].buffer = Core::ResourceManager::get()->getResource<GFX::Buffer>(MAAFIntermediateIndirectBuffer)->buffer.get();
				encoder->pipelineBarrier(bufferComp2Comp_R2W);

				filter_pass->composeCommands_y(encoder, index);

				bufferComp2Comp_W2R.bufferMemoryBarriers[0].buffer = Core::ResourceManager::get()->getResource<GFX::Buffer>(MAAFIntermediateDirectBuffer)->buffer.get();
				encoder->pipelineBarrier(bufferComp2Comp_W2R);
				bufferComp2Comp_W2R.bufferMemoryBarriers[0].buffer = Core::ResourceManager::get()->getResource<GFX::Buffer>(MAAFIntermediateIndirectBuffer)->buffer.get();
				encoder->pipelineBarrier(bufferComp2Comp_W2R);
			}
			result_display_pass->composeCommands(encoder, index);

			// ready output image for display
			encoder->pipelineBarrier(RHI::BarrierDescriptor{
				(uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT,
				(uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT,
				(uint32_t)RHI::DependencyType::NONE,
				{}, {},
				{ RHI::TextureMemoryBarrierDescriptor{
					Core::ResourceManager::get()->getResource<GFX::Texture>(rtTarget)->texture.get(),
					RHI::ImageSubresourceRange{(uint32_t)RHI::TextureAspect::COLOR_BIT, 0,1,0,1},
					(uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT,
					(uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT,
					RHI::TextureLayout::GENERAL,
					RHI::TextureLayout::SHADER_READ_ONLY_OPTIMAL
				}}
				});
		}

		Core::GUID rtTarget;

		size_t width = 800, height = 600;

		Core::GUID defocusSlopeBuffer;
		Core::GUID directSlopeBuffer;
		Core::GUID indirectSlopeBuffer;
		Core::GUID worldPositionBuffer;
		Core::GUID worldNormalBuffer;
		Core::GUID useFilterBuffer;
		Core::GUID MAAFParametersBuffer;
		Core::GUID MAAFIntermediateDirectBuffer;
		Core::GUID MAAFIntermediateIndirectBuffer;
		Core::GUID storageImage;
		Core::GUID defocusSlopeTmpBuffer;
		Core::GUID directSlopeTmpBuffer;
		Core::GUID indirectSlopeTmpBuffer;
		Core::GUID worldPositionTmpBuffer;
		Core::GUID worldNormalTmpBuffer;
		Core::GUID useFilterTmpBuffer;
		Core::GUID MAAFIntermediateDirectTmpBuffer;
		Core::GUID MAAFIntermediateIndirectTmpBuffer;

		std::unique_ptr<RHI::BindGroupLayout> buffersBindGroupLayout = nullptr;
		std::unique_ptr<RHI::BindGroup> bufferBindGroup[2];
		std::unique_ptr<RHI::BindGroupLayout> rtBindGroupLayout = nullptr;
		std::unique_ptr<RHI::BindGroup> rtBindGroup[2];
		std::array<RHI::BindGroup*, 2> camBindGroup[2];

		std::unique_ptr<RHI::PipelineLayout> pipelineLayout;
		std::unique_ptr<RHI::PipelineLayout> displayPipelineLayout;

		std::unique_ptr<MAAF_Initial_Sample_Pass> initial_sampling_pass = nullptr;
		std::unique_ptr<MAAF_Prefilter_Pass> prefilter_pass = nullptr;
		std::unique_ptr<MAAF_Continue_Sample_Pass> continue_sampling_pass = nullptr;
		std::unique_ptr<MAAF_Filter_Pass> filter_pass = nullptr;
		std::unique_ptr<MAAF_Display_Pass> result_display_pass = nullptr;
	};
}