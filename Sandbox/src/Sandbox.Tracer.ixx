module;
#include <array>
#include <filesystem>
#include <memory>
#include <utility>
export module Sandbox.Tracer;
import SE.Core.Resource;
import SE.Math.Geometric;
import SE.GFX.Core;
import SE.RHI;

using namespace SIByL;

namespace Sandbox
{
	export struct DirectTracer {
		DirectTracer(RHI::RHILayer* rhiLayer, std::array<RHI::PipelineLayout*, 2> const& layout) {
			// require GUID
			rgen = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();
			diffuse_rchit = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();
			shadow_rmiss = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();
			sky_rmiss = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();
			// load Shaders
			GFX::GFXManager::get()->registerShaderModuleResource(rgen, "../Engine/Binaries/Runtime/spirv/RayTracing/RayTrace/src/direct_light_tracer/direct_light_integrator_rgen.spv", { nullptr, RHI::ShaderStages::RAYGEN });
			GFX::GFXManager::get()->registerShaderModuleResource(diffuse_rchit, "../Engine/Binaries/Runtime/spirv/RayTracing/RayTrace/src/direct_light_tracer/direct_light_diffuse_mat_rchit.spv", { nullptr, RHI::ShaderStages::CLOSEST_HIT });
			GFX::GFXManager::get()->registerShaderModuleResource(shadow_rmiss, "../Engine/Binaries/Runtime/spirv/RayTracing/RayTrace/src/direct_light_tracer/direct_light_shadow_rmiss.spv", { nullptr, RHI::ShaderStages::MISS });
			GFX::GFXManager::get()->registerShaderModuleResource(sky_rmiss, "../Engine/Binaries/Runtime/spirv/RayTracing/RayTrace/src/direct_light_tracer/direct_light_simple_sky_rmiss.spv", { nullptr, RHI::ShaderStages::MISS });
			// Create rt pipeline
			for (int i = 0; i < 2; ++i) {
				raytracingPipeline[i] = rhiLayer->getDevice()->createRayTracingPipeline(RHI::RayTracingPipelineDescriptor{ 
					layout[i], 2, RHI::SBTsDescriptor{
						RHI::SBTsDescriptor::RayGenerationSBT{{ Core::ResourceManager::get()->getResource<GFX::ShaderModule>(rgen)->shaderModule.get() }},
						RHI::SBTsDescriptor::MissSBT{{
							{Core::ResourceManager::get()->getResource<GFX::ShaderModule>(sky_rmiss)->shaderModule.get()},
							{Core::ResourceManager::get()->getResource<GFX::ShaderModule>(shadow_rmiss)->shaderModule.get()} }},
						RHI::SBTsDescriptor::HitGroupSBT{{
								{{Core::ResourceManager::get()->getResource<GFX::ShaderModule>(diffuse_rchit)->shaderModule.get()}, nullptr, nullptr},
							}}
					} });
			}
		}

		~DirectTracer() {
			raytracingPipeline[0] = nullptr;
			raytracingPipeline[1] = nullptr;
		}

		Core::GUID rgen;
		Core::GUID diffuse_rchit;
		Core::GUID shadow_rmiss;
		Core::GUID sky_rmiss;

		std::unique_ptr<RHI::RayTracingPipeline> raytracingPipeline[2];
	};

	export struct AAF_SoftShadow_Sparse_InitialSampling_Pass {
		AAF_SoftShadow_Sparse_InitialSampling_Pass(RHI::RHILayer* rhiLayer, RHI::PipelineLayout* rtPipelineLayout, 
			std::array<RHI::BindGroup*,2> const& rtBindGroups,
			std::array<RHI::BindGroup*,2> const& camBindGroups,
			std::array<RHI::BindGroup*,2> const& bufferBindGroups)
			: rtBindGroups(rtBindGroups)
			, camBindGroups(camBindGroups)
			, bufferBindGroups(bufferBindGroups) {
			// require GUID
			lwb_primary_ray_rgen = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();
			lwb_primary_ray_rmiss = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();
			aaf_secondary_rchit = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();
			aaf_secondary_rmiss = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();
			lwb_primary_ray_rchit = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();
			// load Shaders
			GFX::GFXManager::get()->registerShaderModuleResource(lwb_primary_ray_rgen, "../Engine/Binaries/Runtime/spirv/RayTracing/RayTrace/src/aaf_softshadow/aaf_initial_sample_rgen.spv", { nullptr, RHI::ShaderStages::RAYGEN });
			GFX::GFXManager::get()->registerShaderModuleResource(lwb_primary_ray_rmiss, "../Engine/Binaries/Runtime/spirv/RayTracing/RayTrace/src/aaf_softshadow/aaf_initial_sample_rmiss.spv", { nullptr, RHI::ShaderStages::MISS });
			GFX::GFXManager::get()->registerShaderModuleResource(aaf_secondary_rchit, "../Engine/Binaries/Runtime/spirv/RayTracing/RayTrace/src/aaf_softshadow/aaf_distance_shadowray_rahit.spv", { nullptr, RHI::ShaderStages::ANY_HIT });
			GFX::GFXManager::get()->registerShaderModuleResource(aaf_secondary_rmiss, "../Engine/Binaries/Runtime/spirv/RayTracing/RayTrace/src/aaf_softshadow/aaf_distance_shadowray_rmiss.spv", { nullptr, RHI::ShaderStages::MISS });
			GFX::GFXManager::get()->registerShaderModuleResource(lwb_primary_ray_rchit, "../Engine/Binaries/Runtime/spirv/RayTracing/RayTrace/src/aaf_softshadow/aaf_shadow_sampling_rchit.spv", { nullptr, RHI::ShaderStages::CLOSEST_HIT });
			// Create rt pipeline
			for (int i = 0; i < 2; ++i) {
				raytracingPipeline[i] = rhiLayer->getDevice()->createRayTracingPipeline(RHI::RayTracingPipelineDescriptor{
					rtPipelineLayout, 2, RHI::SBTsDescriptor{
						RHI::SBTsDescriptor::RayGenerationSBT{{ Core::ResourceManager::get()->getResource<GFX::ShaderModule>(lwb_primary_ray_rgen)->shaderModule.get() }},
						RHI::SBTsDescriptor::MissSBT{{
							{Core::ResourceManager::get()->getResource<GFX::ShaderModule>(lwb_primary_ray_rmiss)->shaderModule.get()},
							{Core::ResourceManager::get()->getResource<GFX::ShaderModule>(aaf_secondary_rmiss)->shaderModule.get()} }},
						RHI::SBTsDescriptor::HitGroupSBT{{
								{{Core::ResourceManager::get()->getResource<GFX::ShaderModule>(lwb_primary_ray_rchit)->shaderModule.get()}, nullptr, nullptr},
								{nullptr, {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(aaf_secondary_rchit)->shaderModule.get()}, nullptr} }}
					} });
			}
		}

		~AAF_SoftShadow_Sparse_InitialSampling_Pass() {
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

		Core::GUID lwb_primary_ray_rgen;
		Core::GUID lwb_primary_ray_rmiss;
		Core::GUID aaf_secondary_rchit;
		Core::GUID aaf_secondary_rmiss;
		Core::GUID lwb_primary_ray_rchit;

		std::unique_ptr<RHI::RayTracingPipeline> raytracingPipeline[2];
		std::unique_ptr<RHI::RayTracingPassEncoder> rtEncoder[2] = {};
		std::array<RHI::BindGroup*, 2> rtBindGroups = {};
		std::array<RHI::BindGroup*, 2> camBindGroups = {};
		std::array<RHI::BindGroup*, 2> bufferBindGroups = {};
	};

	export struct AAF_SoftShadow_Sparse_ContinueSampling_Pass {
		AAF_SoftShadow_Sparse_ContinueSampling_Pass(RHI::RHILayer* rhiLayer, RHI::PipelineLayout* rtPipelineLayout,
			std::array<RHI::BindGroup*, 2> const& rtBindGroups,
			std::array<RHI::BindGroup*, 2> const& camBindGroups,
			std::array<RHI::BindGroup*, 2> const& bufferBindGroups)
			: rtBindGroups(rtBindGroups)
			, camBindGroups(camBindGroups)
			, bufferBindGroups(bufferBindGroups) {
			// require GUID
			aaf_continue_rgen = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();
			aaf_continue_rchit = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();
			aaf_continue_rmiss = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();
			aaf_shadow_rchit = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();
			aaf_shadow_rmiss = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();
			// load Shaders
			GFX::GFXManager::get()->registerShaderModuleResource(aaf_continue_rgen, "../Engine/Binaries/Runtime/spirv/RayTracing/RayTrace/src/aaf_softshadow/aaf_continue_sample_rgen.spv", { nullptr, RHI::ShaderStages::RAYGEN });
			GFX::GFXManager::get()->registerShaderModuleResource(aaf_continue_rchit, "../Engine/Binaries/Runtime/spirv/RayTracing/RayTrace/src/aaf_softshadow/aaf_continue_sample_rchit.spv", { nullptr, RHI::ShaderStages::CLOSEST_HIT });
			GFX::GFXManager::get()->registerShaderModuleResource(aaf_continue_rmiss, "../Engine/Binaries/Runtime/spirv/RayTracing/RayTrace/src/aaf_softshadow/aaf_initial_sample_rmiss.spv", { nullptr, RHI::ShaderStages::MISS });
			GFX::GFXManager::get()->registerShaderModuleResource(aaf_shadow_rchit, "../Engine/Binaries/Runtime/spirv/RayTracing/RayTrace/src/aaf_softshadow/aaf_continue_shadow_rchit.spv", { nullptr, RHI::ShaderStages::CLOSEST_HIT });
			GFX::GFXManager::get()->registerShaderModuleResource(aaf_shadow_rmiss, "../Engine/Binaries/Runtime/spirv/RayTracing/RayTrace/src/aaf_softshadow/aaf_continue_shadow_rmiss.spv", { nullptr, RHI::ShaderStages::MISS });
			// Create rt pipeline
			for (int i = 0; i < 2; ++i) {
				raytracingPipeline[i] = rhiLayer->getDevice()->createRayTracingPipeline(RHI::RayTracingPipelineDescriptor{
					rtPipelineLayout, 2, RHI::SBTsDescriptor{
						RHI::SBTsDescriptor::RayGenerationSBT{{ Core::ResourceManager::get()->getResource<GFX::ShaderModule>(aaf_continue_rgen)->shaderModule.get() }},
						RHI::SBTsDescriptor::MissSBT{{
							{Core::ResourceManager::get()->getResource<GFX::ShaderModule>(aaf_continue_rmiss)->shaderModule.get()},
							{Core::ResourceManager::get()->getResource<GFX::ShaderModule>(aaf_shadow_rmiss)->shaderModule.get()} }},
						RHI::SBTsDescriptor::HitGroupSBT{{
								{{Core::ResourceManager::get()->getResource<GFX::ShaderModule>(aaf_continue_rchit)->shaderModule.get()}, nullptr, nullptr},
								{{Core::ResourceManager::get()->getResource<GFX::ShaderModule>(aaf_shadow_rchit)->shaderModule.get()}, nullptr, nullptr} }}
					} });
			}
		}

		~AAF_SoftShadow_Sparse_ContinueSampling_Pass() {
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

		Core::GUID aaf_continue_rgen;
		Core::GUID aaf_continue_rchit;
		Core::GUID aaf_continue_rmiss;
		Core::GUID aaf_shadow_rchit;
		Core::GUID aaf_shadow_rmiss;

		std::unique_ptr<RHI::RayTracingPipeline> raytracingPipeline[2];
		std::unique_ptr<RHI::RayTracingPassEncoder> rtEncoder[2] = {};
		std::array<RHI::BindGroup*, 2> rtBindGroups = {};
		std::array<RHI::BindGroup*, 2> camBindGroups = {};
		std::array<RHI::BindGroup*, 2> bufferBindGroups = {};
	};

	export struct AAF_SoftShadow_SlopeFilter_Pass {
		AAF_SoftShadow_SlopeFilter_Pass(RHI::RHILayer* rhiLayer, RHI::PipelineLayout* slopleFilterPipelineLayout,
			std::array<RHI::BindGroup*, 2> const& bufferBindGroups)
			: slopleFilterPipelineLayout(slopleFilterPipelineLayout), bufferBindGroups(bufferBindGroups)
		{
			aaf_slope_filter_x_comp = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();
			aaf_slope_filter_y_comp = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();
			GFX::GFXManager::get()->registerShaderModuleResource(aaf_slope_filter_x_comp, 
				"../Engine/Binaries/Runtime/spirv/RayTracing/RayTrace/src/aaf_softshadow/aaf_slope_filter_x_comp.spv", 
				{ nullptr, RHI::ShaderStages::COMPUTE });
			GFX::GFXManager::get()->registerShaderModuleResource(aaf_slope_filter_y_comp, 
				"../Engine/Binaries/Runtime/spirv/RayTracing/RayTrace/src/aaf_softshadow/aaf_slope_filter_y_comp.spv", 
				{ nullptr, RHI::ShaderStages::COMPUTE });
			// create compute pipeline
			for (int i = 0; i < 2; ++i) {
				computePipelineX[i] = rhiLayer->getDevice()->createComputePipeline(RHI::ComputePipelineDescriptor{
					slopleFilterPipelineLayout,
					{Core::ResourceManager::get()->getResource<GFX::ShaderModule>(aaf_slope_filter_x_comp)->shaderModule.get(), "main"}
					});
				computePipelineY[i] = rhiLayer->getDevice()->createComputePipeline(RHI::ComputePipelineDescriptor{
					slopleFilterPipelineLayout,
					{Core::ResourceManager::get()->getResource<GFX::ShaderModule>(aaf_slope_filter_y_comp)->shaderModule.get(), "main"}
					});
			}

		}

		~AAF_SoftShadow_SlopeFilter_Pass() {
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

		Core::GUID aaf_slope_filter_x_comp;
		Core::GUID aaf_slope_filter_y_comp;
		RHI::PipelineLayout* slopleFilterPipelineLayout;
		std::unique_ptr<RHI::ComputePipeline> computePipelineX[2];
		std::unique_ptr<RHI::ComputePipeline> computePipelineY[2];
		std::array<RHI::BindGroup*, 2> bufferBindGroups;
		std::unique_ptr<RHI::ComputePassEncoder> compEncoderX[2] = {};
		std::unique_ptr<RHI::ComputePassEncoder> compEncoderY[2] = {};
	};

	export struct AAF_SoftShadow_VisibilityFilter_Pass {
		AAF_SoftShadow_VisibilityFilter_Pass(RHI::RHILayer* rhiLayer, RHI::PipelineLayout* slopleFilterPipelineLayout,
			std::array<RHI::BindGroup*, 2> const& bufferBindGroups)
			: slopleFilterPipelineLayout(slopleFilterPipelineLayout), bufferBindGroups(bufferBindGroups)
		{
			maaf_prefilter_x_comp = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();
			maaf_prefilter_y_comp = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();
			GFX::GFXManager::get()->registerShaderModuleResource(maaf_prefilter_x_comp,
				"../Engine/Binaries/Runtime/spirv/RayTracing/RayTrace/src/aaf_softshadow/aaf_occlusion_filter_x_comp.spv",
				{ nullptr, RHI::ShaderStages::COMPUTE });
			GFX::GFXManager::get()->registerShaderModuleResource(maaf_prefilter_y_comp,
				"../Engine/Binaries/Runtime/spirv/RayTracing/RayTrace/src/aaf_softshadow/aaf_occlusion_filter_y_comp.spv",
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

		~AAF_SoftShadow_VisibilityFilter_Pass() {
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
	
	export struct AAF_SoftShadow_Display_Pass {

		AAF_SoftShadow_Display_Pass(RHI::RHILayer* rhiLayer, RHI::PipelineLayout* displayPipelineLayout,
			std::array<RHI::BindGroup*, 2> const& bufferBindGroups)
		: bufferBindGroups(bufferBindGroups) {
			// get the shader
			aaf_result_display_comp = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();
			GFX::GFXManager::get()->registerShaderModuleResource(aaf_result_display_comp, "../Engine/Binaries/Runtime/spirv/RayTracing/RayTrace/src/aaf_softshadow/aaf_result_display_comp.spv", { nullptr, RHI::ShaderStages::COMPUTE });
			// create compute pipeline
			for (int i = 0; i < 2; ++i)
				computePipeline[i] = rhiLayer->getDevice()->createComputePipeline(RHI::ComputePipelineDescriptor{
					displayPipelineLayout,
					{Core::ResourceManager::get()->getResource<GFX::ShaderModule>(aaf_result_display_comp)->shaderModule.get(), "main"}
				});
		}

		~AAF_SoftShadow_Display_Pass() {
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

	export struct AAFPipeline {
		Core::GUID rtTarget;

		AAFPipeline(RHI::RHILayer* rhiLayer, GFX::ASGroup* asgroup, Core::GUID rtTarget,
			RHI::BindGroupLayout* cameraBindGroupLayout, std::array<RHI::BindGroup*, 2> const& camBindGroup)
			: rtTarget(rtTarget)
		{
			// allocate buffers
			zMinMaxBuffer = Core::ResourceManager::get()->requestRuntimeGUID<GFX::Buffer>();
			loBuffer = Core::ResourceManager::get()->requestRuntimeGUID<GFX::Buffer>();
			colorBuffer = Core::ResourceManager::get()->requestRuntimeGUID<GFX::Buffer>();
			worldPosBuffer = Core::ResourceManager::get()->requestRuntimeGUID<GFX::Buffer>();
			worldNormalBuffer = Core::ResourceManager::get()->requestRuntimeGUID<GFX::Buffer>();
			projDistBuffer = Core::ResourceManager::get()->requestRuntimeGUID<GFX::Buffer>();
			useFilterBuffer = Core::ResourceManager::get()->requestRuntimeGUID<GFX::Buffer>();
			useFilterBlurredBuffer = Core::ResourceManager::get()->requestRuntimeGUID<GFX::Buffer>();
			visBlurredBuffer = Core::ResourceManager::get()->requestRuntimeGUID<GFX::Buffer>();
			slopeBlurredBuffer = Core::ResourceManager::get()->requestRuntimeGUID<GFX::Buffer>();
			sppBuffer = Core::ResourceManager::get()->requestRuntimeGUID<GFX::Buffer>();
			betaBuffer = Core::ResourceManager::get()->requestRuntimeGUID<GFX::Buffer>();

			GFX::GFXManager::get()->registerBufferResource(zMinMaxBuffer, RHI::BufferDescriptor{
				width * height * sizeof(float) * 2, // pixel size * sizeof(vec2)
				(uint32_t)RHI::BufferUsage::STORAGE });
			GFX::GFXManager::get()->registerBufferResource(loBuffer, RHI::BufferDescriptor{
				width * height * sizeof(float) * 4, // pixel size * sizeof(align(vec3))
				(uint32_t)RHI::BufferUsage::STORAGE });
			GFX::GFXManager::get()->registerBufferResource(colorBuffer, RHI::BufferDescriptor{
				width * height * sizeof(float) * 1, // pixel size * sizeof(align(float))
				(uint32_t)RHI::BufferUsage::STORAGE });
			GFX::GFXManager::get()->registerBufferResource(worldPosBuffer, RHI::BufferDescriptor{
				width * height * sizeof(float) * 4, // pixel size * sizeof(align(vec3))
				(uint32_t)RHI::BufferUsage::STORAGE });
			GFX::GFXManager::get()->registerBufferResource(worldNormalBuffer, RHI::BufferDescriptor{
				width * height * sizeof(float) * 4, // pixel size * sizeof(align(vec3))
				(uint32_t)RHI::BufferUsage::STORAGE });
			GFX::GFXManager::get()->registerBufferResource(projDistBuffer, RHI::BufferDescriptor{
				width * height * sizeof(float) * 3, // pixel size * sizeof(float3)
				(uint32_t)RHI::BufferUsage::STORAGE });
			GFX::GFXManager::get()->registerBufferResource(useFilterBuffer, RHI::BufferDescriptor{
				width * height * sizeof(unsigned int) * 1, // pixel size * sizeof(uint)
				(uint32_t)RHI::BufferUsage::STORAGE });
			GFX::GFXManager::get()->registerBufferResource(useFilterBlurredBuffer, RHI::BufferDescriptor{
				width * height * sizeof(unsigned int) * 1, // pixel size * sizeof(uint)
				(uint32_t)RHI::BufferUsage::STORAGE });
			GFX::GFXManager::get()->registerBufferResource(visBlurredBuffer, RHI::BufferDescriptor{
				width * height * sizeof(float) * 1, // pixel size * sizeof(align(float))
				(uint32_t)RHI::BufferUsage::STORAGE });
			GFX::GFXManager::get()->registerBufferResource(slopeBlurredBuffer, RHI::BufferDescriptor{
				width * height * sizeof(float) * 2, // pixel size * sizeof(align(float2))
				(uint32_t)RHI::BufferUsage::STORAGE });
			GFX::GFXManager::get()->registerBufferResource(sppBuffer, RHI::BufferDescriptor{
				width * height * sizeof(unsigned int) * 1, // pixel size * sizeof(align(uint))
				(uint32_t)RHI::BufferUsage::STORAGE });
			GFX::GFXManager::get()->registerBufferResource(betaBuffer, RHI::BufferDescriptor{
				width * height * sizeof(float) * 1, // pixel size * sizeof(align(uint))
				(uint32_t)RHI::BufferUsage::STORAGE });
			GFX::Buffer* pSlopeBuffer = Core::ResourceManager::get()->getResource<GFX::Buffer>(zMinMaxBuffer);
			GFX::Buffer* pVisBuffer = Core::ResourceManager::get()->getResource<GFX::Buffer>(loBuffer);
			GFX::Buffer* pProjDistBuffer = Core::ResourceManager::get()->getResource<GFX::Buffer>(colorBuffer);
			GFX::Buffer* pWorldPosBuffer = Core::ResourceManager::get()->getResource<GFX::Buffer>(worldPosBuffer);
			GFX::Buffer* pWorldNormalBuffer = Core::ResourceManager::get()->getResource<GFX::Buffer>(worldNormalBuffer);
			GFX::Buffer* pVisBlurredBuffer = Core::ResourceManager::get()->getResource<GFX::Buffer>(visBlurredBuffer);
			GFX::Buffer* pBRDFBuffer = Core::ResourceManager::get()->getResource<GFX::Buffer>(projDistBuffer);
			GFX::Buffer* pUseFilterBuffer = Core::ResourceManager::get()->getResource<GFX::Buffer>(useFilterBuffer);
			GFX::Buffer* pUseFilterBlurredBuffer = Core::ResourceManager::get()->getResource<GFX::Buffer>(useFilterBlurredBuffer);
			GFX::Buffer* pSlopeBlurredBuffer = Core::ResourceManager::get()->getResource<GFX::Buffer>(slopeBlurredBuffer);
			GFX::Buffer* pSppBuffer = Core::ResourceManager::get()->getResource<GFX::Buffer>(sppBuffer);
			GFX::Buffer* pBetaBuffer = Core::ResourceManager::get()->getResource<GFX::Buffer>(betaBuffer);
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
					RHI::BindGroupLayoutEntry{7, stages, RHI::StorageTextureBindingLayout{}},							// render target storage image
					RHI::BindGroupLayoutEntry{8, stages, RHI::BufferBindingLayout{RHI::BufferBindingType::STORAGE}},	// use filter tmp buffer
					RHI::BindGroupLayoutEntry{9, stages, RHI::BufferBindingLayout{RHI::BufferBindingType::STORAGE}},	// vis blur tmp buffer
					RHI::BindGroupLayoutEntry{10, stages, RHI::BufferBindingLayout{RHI::BufferBindingType::STORAGE}},	// vis blur tmp buffer
					RHI::BindGroupLayoutEntry{11, stages, RHI::BufferBindingLayout{RHI::BufferBindingType::STORAGE}},	// vis blur tmp buffer
					RHI::BindGroupLayoutEntry{12, stages, RHI::BufferBindingLayout{RHI::BufferBindingType::STORAGE}},	// vis blur tmp buffer
				} });
			// create bind groups
			for (int i = 0; i < 2; ++i) {
				bufferBindGroup[i] = rhiLayer->getDevice()->createBindGroup(RHI::BindGroupDescriptor{
					buffersBindGroupLayout.get(),
					std::vector<RHI::BindGroupEntry>{
						{0,RHI::BindingResource{{pSlopeBuffer->buffer.get(), 0, pSlopeBuffer->buffer->size()}}},
						{1,RHI::BindingResource{{pVisBuffer->buffer.get(), 0, pVisBuffer->buffer->size()}}},
						{2,RHI::BindingResource{{pProjDistBuffer->buffer.get(), 0, pProjDistBuffer->buffer->size()}}},
						{3,RHI::BindingResource{{pWorldPosBuffer->buffer.get(), 0, pWorldPosBuffer->buffer->size()}}},
						{4,RHI::BindingResource{{pWorldNormalBuffer->buffer.get(), 0, pWorldNormalBuffer->buffer->size()}}},
						{5,RHI::BindingResource{{pBRDFBuffer->buffer.get(), 0, pBRDFBuffer->buffer->size()}}},
						{6,RHI::BindingResource{{pUseFilterBuffer->buffer.get(), 0, pUseFilterBuffer->buffer->size()}}},
						{7,RHI::BindingResource{Core::ResourceManager::get()->getResource<GFX::Texture>(rtTarget)->originalView.get()}},
						{8,RHI::BindingResource{{pUseFilterBlurredBuffer->buffer.get(), 0, pUseFilterBlurredBuffer->buffer->size()}}},
						{9,RHI::BindingResource{{pVisBlurredBuffer->buffer.get(), 0, pVisBlurredBuffer->buffer->size()}}},
						{10,RHI::BindingResource{{pSlopeBlurredBuffer->buffer.get(), 0, pSlopeBlurredBuffer->buffer->size()}}},
						{11,RHI::BindingResource{{pSppBuffer->buffer.get(), 0, pSppBuffer->buffer->size()}}},
						{12,RHI::BindingResource{{pBetaBuffer->buffer.get(), 0, pBetaBuffer->buffer->size()}}},
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

			initial_sampling_pass = std::make_unique<AAF_SoftShadow_Sparse_InitialSampling_Pass>(rhiLayer, pipelineLayout.get(),
				std::array<RHI::BindGroup*, 2>{rtBindGroup[0].get(), rtBindGroup[1].get()},
				std::array<RHI::BindGroup*, 2>{camBindGroup[0], camBindGroup[1]},
				std::array<RHI::BindGroup*, 2>{bufferBindGroup[0].get(), bufferBindGroup[1].get()});
			slope_filter_pass = std::make_unique<AAF_SoftShadow_SlopeFilter_Pass>(rhiLayer, displayPipelineLayout.get(),
				std::array<RHI::BindGroup*, 2>{bufferBindGroup[0].get(), bufferBindGroup[1].get()});
			continue_sampling_pass = std::make_unique<AAF_SoftShadow_Sparse_ContinueSampling_Pass>(rhiLayer, pipelineLayout.get(),
				std::array<RHI::BindGroup*, 2>{rtBindGroup[0].get(), rtBindGroup[1].get()},
				std::array<RHI::BindGroup*, 2>{camBindGroup[0], camBindGroup[1]},
				std::array<RHI::BindGroup*, 2>{bufferBindGroup[0].get(), bufferBindGroup[1].get()});
			vis_filter_pass = std::make_unique<AAF_SoftShadow_VisibilityFilter_Pass>(rhiLayer, displayPipelineLayout.get(),
				std::array<RHI::BindGroup*, 2>{bufferBindGroup[0].get(), bufferBindGroup[1].get()});
			display_pass = std::make_unique<AAF_SoftShadow_Display_Pass>(rhiLayer, displayPipelineLayout.get(), 
				std::array<RHI::BindGroup*, 2>{bufferBindGroup[0].get(), bufferBindGroup[1].get()});
		}

		~AAFPipeline() {
			display_pass = nullptr;
			initial_sampling_pass = nullptr;
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

			initial_sampling_pass->composeCommands(encoder, index);
			
			
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

			// slope filtering
			{
				// ready slopeBuffer/useFilterBuffer from RT
				bufferRT2Comp.bufferMemoryBarriers[0].buffer = Core::ResourceManager::get()->getResource<GFX::Buffer>(zMinMaxBuffer)->buffer.get();
				encoder->pipelineBarrier(bufferRT2Comp);
				bufferRT2Comp.bufferMemoryBarriers[0].buffer = Core::ResourceManager::get()->getResource<GFX::Buffer>(useFilterBuffer)->buffer.get();
				encoder->pipelineBarrier(bufferRT2Comp);
				// ready useFilterBlurTmpBuffer/slopeBlurTmpBuffer from previous comp
				bufferComp2Comp_R2W.bufferMemoryBarriers[0].buffer = Core::ResourceManager::get()->getResource<GFX::Buffer>(slopeBlurredBuffer)->buffer.get();
				encoder->pipelineBarrier(bufferComp2Comp_R2W);
				bufferComp2Comp_R2W.bufferMemoryBarriers[0].buffer = Core::ResourceManager::get()->getResource<GFX::Buffer>(useFilterBlurredBuffer)->buffer.get();
				encoder->pipelineBarrier(bufferComp2Comp_R2W);
				// compose commands x
				slope_filter_pass->composeCommands_x(encoder, index);
				// ready useFilterBlurTmpBuffer/slopeBlurTmpBuffer from previous comp
				bufferComp2Comp_W2R.bufferMemoryBarriers[0].buffer = Core::ResourceManager::get()->getResource<GFX::Buffer>(slopeBlurredBuffer)->buffer.get();
				encoder->pipelineBarrier(bufferComp2Comp_W2R);
				bufferComp2Comp_W2R.bufferMemoryBarriers[0].buffer = Core::ResourceManager::get()->getResource<GFX::Buffer>(useFilterBlurredBuffer)->buffer.get();
				encoder->pipelineBarrier(bufferComp2Comp_W2R);
				bufferComp2Comp_R2W.bufferMemoryBarriers[0].buffer = Core::ResourceManager::get()->getResource<GFX::Buffer>(zMinMaxBuffer)->buffer.get();
				encoder->pipelineBarrier(bufferComp2Comp_R2W);
				bufferComp2Comp_R2W.bufferMemoryBarriers[0].buffer = Core::ResourceManager::get()->getResource<GFX::Buffer>(useFilterBuffer)->buffer.get();
				encoder->pipelineBarrier(bufferComp2Comp_R2W);
				// compose commands y
				slope_filter_pass->composeCommands_y(encoder, index);
				bufferComp2Comp_W2R.bufferMemoryBarriers[0].buffer = Core::ResourceManager::get()->getResource<GFX::Buffer>(zMinMaxBuffer)->buffer.get();
				encoder->pipelineBarrier(bufferComp2Comp_W2R);
				bufferComp2Comp_W2R.bufferMemoryBarriers[0].buffer = Core::ResourceManager::get()->getResource<GFX::Buffer>(useFilterBuffer)->buffer.get();
				encoder->pipelineBarrier(bufferComp2Comp_W2R);
			}
			continue_sampling_pass->composeCommands(encoder, index);
			{
				bufferRT2Comp.bufferMemoryBarriers[0].buffer = Core::ResourceManager::get()->getResource<GFX::Buffer>(loBuffer)->buffer.get();
				encoder->pipelineBarrier(bufferRT2Comp);
				bufferRT2Comp.bufferMemoryBarriers[0].buffer = Core::ResourceManager::get()->getResource<GFX::Buffer>(colorBuffer)->buffer.get();
				encoder->pipelineBarrier(bufferRT2Comp);
				bufferRT2Comp.bufferMemoryBarriers[0].buffer = Core::ResourceManager::get()->getResource<GFX::Buffer>(worldPosBuffer)->buffer.get();
				encoder->pipelineBarrier(bufferRT2Comp);
				bufferRT2Comp.bufferMemoryBarriers[0].buffer = Core::ResourceManager::get()->getResource<GFX::Buffer>(worldNormalBuffer)->buffer.get();
				encoder->pipelineBarrier(bufferRT2Comp);
				// get vis blurred ready
				bufferComp2Comp_R2W.bufferMemoryBarriers[0].buffer = Core::ResourceManager::get()->getResource<GFX::Buffer>(visBlurredBuffer)->buffer.get();
				encoder->pipelineBarrier(bufferComp2Comp_R2W);

				vis_filter_pass->composeCommands_x(encoder, index);

				bufferComp2Comp_W2R.bufferMemoryBarriers[0].buffer = Core::ResourceManager::get()->getResource<GFX::Buffer>(visBlurredBuffer)->buffer.get();
				encoder->pipelineBarrier(bufferComp2Comp_W2R);
				bufferComp2Comp_R2W.bufferMemoryBarriers[0].buffer = Core::ResourceManager::get()->getResource<GFX::Buffer>(loBuffer)->buffer.get();
				encoder->pipelineBarrier(bufferComp2Comp_R2W);

				vis_filter_pass->composeCommands_y(encoder, index);
				bufferComp2Comp_W2R.bufferMemoryBarriers[0].buffer = Core::ResourceManager::get()->getResource<GFX::Buffer>(loBuffer)->buffer.get();
				encoder->pipelineBarrier(bufferComp2Comp_W2R);
			}

			bufferRT2Comp.bufferMemoryBarriers[0].buffer = Core::ResourceManager::get()->getResource<GFX::Buffer>(projDistBuffer)->buffer.get();
			encoder->pipelineBarrier(bufferRT2Comp);

			display_pass->composeCommands(encoder, index);

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

		size_t width = 800, height = 600;

		Core::GUID zMinMaxBuffer;
		Core::GUID loBuffer;
		Core::GUID colorBuffer;
		Core::GUID worldPosBuffer;
		Core::GUID worldNormalBuffer;
		Core::GUID projDistBuffer;
		Core::GUID useFilterBuffer;
		Core::GUID useFilterBlurredBuffer;
		Core::GUID visBlurredBuffer;
		Core::GUID slopeBlurredBuffer;
		Core::GUID sppBuffer;
		Core::GUID betaBuffer;

		std::unique_ptr<RHI::BindGroupLayout> buffersBindGroupLayout = nullptr;
		std::unique_ptr<RHI::BindGroup> bufferBindGroup[2];
		std::unique_ptr<RHI::BindGroupLayout> rtBindGroupLayout = nullptr;
		std::unique_ptr<RHI::BindGroup> rtBindGroup[2];
		std::array<RHI::BindGroup*, 2> camBindGroup[2];

		std::unique_ptr<RHI::PipelineLayout> pipelineLayout;
		std::unique_ptr<RHI::PipelineLayout> displayPipelineLayout;

		// Pipelines
		std::unique_ptr<AAF_SoftShadow_Sparse_InitialSampling_Pass> initial_sampling_pass = nullptr;
		std::unique_ptr<AAF_SoftShadow_SlopeFilter_Pass> slope_filter_pass = nullptr;
		std::unique_ptr<AAF_SoftShadow_Sparse_ContinueSampling_Pass> continue_sampling_pass = nullptr;
		std::unique_ptr<AAF_SoftShadow_VisibilityFilter_Pass> vis_filter_pass = nullptr;
		std::unique_ptr<AAF_SoftShadow_Display_Pass> display_pass = nullptr;
	};
}