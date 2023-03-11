module;
#include <array>
#include <memory>
#include <compare>
#include <typeinfo>
#include <filesystem>
#include <functional>
#include "../../../Application/Public/SE.Application.Config.h"
export module SE.SRenderer:Tracer.SMultiCubemap;
import :SRenderer;
import SE.Core.Resource;
import SE.RHI;
import SE.GFX.Core;
import SE.GFX.RDG;

namespace SIByL
{
	export struct MultiCubemapPass :public SRenderer::Pass {

		virtual auto loadShaders() noexcept -> void override {
			rgen = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();
			rmiss = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();
			rchit_trimesh = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();
			rchit_sphere = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();
			rint_sphere = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();

			shadow_ray_rchit = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();
			shadow_ray_rmiss = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();

			trimesh_sampling_rcall = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();
			trimesh_sampling_pdf_rcall = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();
			sphere_sampling_rcall = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();
			sphere_sampling_pdf_rcall = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();

			lambertian_eval = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();
			lambertian_sample = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();
			lambertian_pdf = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();
			principled_eval = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();
			principled_sample = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();
			principled_pdf = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();

			GFX::GFXManager::get()->registerShaderModuleResource(rgen, "../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/path_tracer/unipt_mcm_tracer_rgen.spv", { nullptr, RHI::ShaderStages::RAYGEN });
			GFX::GFXManager::get()->registerShaderModuleResource(rmiss, "../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/path_tracer/stracer_rmiss.spv", { nullptr, RHI::ShaderStages::MISS });
			GFX::GFXManager::get()->registerShaderModuleResource(rchit_trimesh,
				"../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/path_tracer/spt_primary_ray_trimesh_rchit.spv",
				{ nullptr, RHI::ShaderStages::CLOSEST_HIT });
			GFX::GFXManager::get()->registerShaderModuleResource(rchit_sphere,
				"../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/path_tracer/spt_primary_ray_sphere_rchit.spv",
				{ nullptr, RHI::ShaderStages::CLOSEST_HIT });

			GFX::GFXManager::get()->registerShaderModuleResource(shadow_ray_rchit,
				"../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/path_tracer/spt_shadow_ray_rchit.spv",
				{ nullptr, RHI::ShaderStages::CLOSEST_HIT });
			GFX::GFXManager::get()->registerShaderModuleResource(shadow_ray_rmiss,
				"../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/path_tracer/spt_shadow_ray_rmiss.spv",
				{ nullptr, RHI::ShaderStages::MISS });

			// Plugins: primitives
			// - sphere
			GFX::GFXManager::get()->registerShaderModuleResource(rint_sphere,
				"../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/plugins/primitive/sphere_hint_rint.spv",
				{ nullptr, RHI::ShaderStages::INTERSECTION });
			GFX::GFXManager::get()->registerShaderModuleResource(sphere_sampling_rcall,
				"../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/plugins/primitive/sphere_sample_rcall.spv",
				{ nullptr, RHI::ShaderStages::CALLABLE });
			GFX::GFXManager::get()->registerShaderModuleResource(sphere_sampling_pdf_rcall,
				"../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/plugins/primitive/sphere_sample_pdf_rcall.spv",
				{ nullptr, RHI::ShaderStages::CALLABLE });
			// - trimesh
			GFX::GFXManager::get()->registerShaderModuleResource(trimesh_sampling_rcall,
				"../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/plugins/primitive/trimesh_sample_rcall.spv",
				{ nullptr, RHI::ShaderStages::CALLABLE });
			GFX::GFXManager::get()->registerShaderModuleResource(trimesh_sampling_pdf_rcall,
				"../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/plugins/primitive/trimesh_sample_pdf_rcall.spv",
				{ nullptr, RHI::ShaderStages::CALLABLE });

			// Plugins: bsdfs
			// - lambertian
			GFX::GFXManager::get()->registerShaderModuleResource(lambertian_eval,
				"../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/plugins/material/lambertian_eval_rcall.spv",
				{ nullptr, RHI::ShaderStages::CALLABLE });
			GFX::GFXManager::get()->registerShaderModuleResource(lambertian_sample,
				"../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/plugins/material/lambertian_sample_rcall.spv",
				{ nullptr, RHI::ShaderStages::CALLABLE });
			GFX::GFXManager::get()->registerShaderModuleResource(lambertian_pdf,
				"../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/plugins/material/lambertian_pdf_rcall.spv",
				{ nullptr, RHI::ShaderStages::CALLABLE });
			// - principled
			GFX::GFXManager::get()->registerShaderModuleResource(principled_eval,
				"../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/plugins/material/principled_eval_rcall.spv",
				{ nullptr, RHI::ShaderStages::CALLABLE });
			GFX::GFXManager::get()->registerShaderModuleResource(principled_sample,
				"../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/plugins/material/principled_sample_rcall.spv",
				{ nullptr, RHI::ShaderStages::CALLABLE });
			GFX::GFXManager::get()->registerShaderModuleResource(principled_pdf,
				"../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/plugins/material/principled_pdf_rcall.spv",
				{ nullptr, RHI::ShaderStages::CALLABLE });
		}

		Core::GUID rgen;
		Core::GUID rchit_trimesh;
		Core::GUID rchit_sphere;
		Core::GUID rmiss;

		Core::GUID shadow_ray_rchit;
		Core::GUID shadow_ray_rmiss;

		// Plugins: primitives
		// - sphere
		Core::GUID rint_sphere;
		Core::GUID sphere_sampling_rcall;
		Core::GUID sphere_sampling_pdf_rcall;
		// - trimesh
		Core::GUID trimesh_sampling_rcall;
		Core::GUID trimesh_sampling_pdf_rcall;

		// Plugins: bsdfs
		// - lambertian
		Core::GUID lambertian_eval;
		Core::GUID lambertian_sample;
		Core::GUID lambertian_pdf;
		// - principled
		Core::GUID principled_eval;
		Core::GUID principled_sample;
		Core::GUID principled_pdf;

		std::unique_ptr<RHI::BindGroupLayout> set2_layout = 0;
		virtual auto registerPass(SRenderer* renderer) noexcept -> void override {
			GFX::RDGraph* rdg = renderer->rdgraph;
			RHI::Device* device = GFX::GFXManager::get()->rhiLayer->getDevice();
			RHI::ShaderStagesFlags stages =
				(uint32_t)RHI::ShaderStages::VERTEX
				| (uint32_t)RHI::ShaderStages::FRAGMENT
				| (uint32_t)RHI::ShaderStages::RAYGEN
				| (uint32_t)RHI::ShaderStages::CLOSEST_HIT
				| (uint32_t)RHI::ShaderStages::INTERSECTION
				| (uint32_t)RHI::ShaderStages::MISS
				| (uint32_t)RHI::ShaderStages::CALLABLE
				| (uint32_t)RHI::ShaderStages::ANY_HIT
				| (uint32_t)RHI::ShaderStages::COMPUTE;
			if (set2_layout == nullptr) {
				set2_layout = device->createBindGroupLayout(
					RHI::BindGroupLayoutDescriptor{ {
						RHI::BindGroupLayoutEntry{ 0, stages, RHI::StorageTextureBindingLayout{}},
						} }
				);
			}

			GFX::RDGPassNode* pass = rdg->addPass("PathTracerPass", GFX::RDGPassFlag::RAY_TRACING,
				[rdg = rdg]()->void {
					// consume
					rdg->getTexture("TracerTarget_Color")->consume(GFX::ConsumeType::READ_WRITE, RHI::TextureLayout::GENERAL, RHI::TextureUsage::STORAGE_BINDING);
					rdg->getTexture("MultiCubemap_0")->consume(GFX::ConsumeType::READ_WRITE, RHI::TextureLayout::GENERAL, RHI::TextureUsage::STORAGE_BINDING);
				},
				[&, renderer = renderer, rdg = rdg]()->GFX::CustomPassExecuteFn {
					// bindgroup layout
					RHI::Device* device = GFX::GFXManager::get()->rhiLayer->getDevice();
				struct PushConstant {
					uint32_t width;
					uint32_t height;
					uint32_t sample_batch;
					uint32_t all_batch = 0;
				};
				std::shared_ptr<RHI::PipelineLayout> pipelineLayout = device->createPipelineLayout(RHI::PipelineLayoutDescriptor{
					{ {(uint32_t)RHI::ShaderStages::RAYGEN
					 | (uint32_t)RHI::ShaderStages::COMPUTE
					 | (uint32_t)RHI::ShaderStages::CLOSEST_HIT
					 | (uint32_t)RHI::ShaderStages::INTERSECTION
					 | (uint32_t)RHI::ShaderStages::MISS
					 | (uint32_t)RHI::ShaderStages::CALLABLE
					 | (uint32_t)RHI::ShaderStages::ANY_HIT, 0, sizeof(PushConstant)}},
					{ renderer->commonDescData.set0_layout.get(),
					  renderer->commonDescData.set1_layout_rt.get(),
					  set2_layout.get() }, });
				std::shared_ptr<RHI::RayTracingPipeline> tracingPipeline[MULTIFRAME_FLIGHTS_COUNT];
				std::shared_ptr<RHI::BindGroup> set2_flights[MULTIFRAME_FLIGHTS_COUNT];
				for (int i = 0; i < MULTIFRAME_FLIGHTS_COUNT; ++i) {
					set2_flights[i] = device->createBindGroup(RHI::BindGroupDescriptor{
						set2_layout.get(),
						std::vector<RHI::BindGroupEntry>{
							{0,RHI::BindingResource{rdg->getTexture("MultiCubemap_0")->texture->originalView.get()}},
					} });
				}
				for (int i = 0; i < MULTIFRAME_FLIGHTS_COUNT; ++i) {
					tracingPipeline[i] = device->createRayTracingPipeline(RHI::RayTracingPipelineDescriptor{
						pipelineLayout.get(), 3, RHI::SBTsDescriptor{
							RHI::SBTsDescriptor::RayGenerationSBT{{ Core::ResourceManager::get()->getResource<GFX::ShaderModule>(rgen)->shaderModule.get() }},
							RHI::SBTsDescriptor::MissSBT{{
								{Core::ResourceManager::get()->getResource<GFX::ShaderModule>(rmiss)->shaderModule.get()},
								{Core::ResourceManager::get()->getResource<GFX::ShaderModule>(shadow_ray_rmiss)->shaderModule.get()}, }},
							RHI::SBTsDescriptor::HitGroupSBT{{
								{Core::ResourceManager::get()->getResource<GFX::ShaderModule>(rchit_trimesh)->shaderModule.get()},
								{Core::ResourceManager::get()->getResource<GFX::ShaderModule>(rchit_sphere)->shaderModule.get(), nullptr,
								 Core::ResourceManager::get()->getResource<GFX::ShaderModule>(rint_sphere)->shaderModule.get(),},
								{Core::ResourceManager::get()->getResource<GFX::ShaderModule>(shadow_ray_rchit)->shaderModule.get()},
								{Core::ResourceManager::get()->getResource<GFX::ShaderModule>(shadow_ray_rchit)->shaderModule.get(), nullptr,
								 Core::ResourceManager::get()->getResource<GFX::ShaderModule>(rint_sphere)->shaderModule.get(),}, }},
							RHI::SBTsDescriptor::CallableSBT{{
								{Core::ResourceManager::get()->getResource<GFX::ShaderModule>(sphere_sampling_rcall)->shaderModule.get()},
								{Core::ResourceManager::get()->getResource<GFX::ShaderModule>(sphere_sampling_pdf_rcall)->shaderModule.get()},
								{Core::ResourceManager::get()->getResource<GFX::ShaderModule>(trimesh_sampling_rcall)->shaderModule.get()},
								{Core::ResourceManager::get()->getResource<GFX::ShaderModule>(trimesh_sampling_pdf_rcall)->shaderModule.get()},
								// lambertian
								{Core::ResourceManager::get()->getResource<GFX::ShaderModule>(lambertian_eval)->shaderModule.get()},
								{Core::ResourceManager::get()->getResource<GFX::ShaderModule>(lambertian_sample)->shaderModule.get()},
								{Core::ResourceManager::get()->getResource<GFX::ShaderModule>(lambertian_pdf)->shaderModule.get()},
								// principled
								{Core::ResourceManager::get()->getResource<GFX::ShaderModule>(principled_eval)->shaderModule.get()},
								{Core::ResourceManager::get()->getResource<GFX::ShaderModule>(principled_sample)->shaderModule.get()},
								{Core::ResourceManager::get()->getResource<GFX::ShaderModule>(principled_pdf)->shaderModule.get()}, }},
						} });
				}
				std::shared_ptr<RHI::RayTracingPassEncoder> passEncoder[2] = {};
				RHI::MultiFrameFlights* multiFrameFlights = GFX::GFXManager::get()->rhiLayer->getMultiFrameFlights();
				// execute callback
				GFX::CustomPassExecuteFn fn = [
					multiFrameFlights, &geometries = renderer->sceneDataPack.geometry_buffer_cpu, pipelineLayout = std::move(pipelineLayout),
						pipelines = std::array<std::shared_ptr<RHI::RayTracingPipeline>, 2>{ std::move(tracingPipeline[0]), std::move(tracingPipeline[1]) },
						rtBindGroup_set0 = renderer->commonDescData.set0_flights_array,
						rtBindGroup_set1 = renderer->commonDescData.set1_flights_rt_array,
						rtBindGroup_set2 = std::array<std::shared_ptr<RHI::BindGroup>, 2>{ std::move(set2_flights[0]), std::move(set2_flights[1]) },
						renderer = renderer,
						passEncoders = std::array<std::shared_ptr<RHI::RayTracingPassEncoder>, 2>{ std::move(passEncoder[0]), std::move(passEncoder[1]) }
				](GFX::RDGRegistry const& registry, RHI::CommandEncoder* cmdEncoder) mutable ->void {
					uint32_t index = multiFrameFlights->getFlightIndex();
					passEncoders[index] = cmdEncoder->beginRayTracingPass(RHI::RayTracingPassDescriptor{});
					if (renderer->sceneDataPack.geometry_buffer_cpu.size() > 0) {
						passEncoders[index]->setPipeline(pipelines[index].get());
						passEncoders[index]->setBindGroup(0, rtBindGroup_set0[index], 0, 0);
						passEncoders[index]->setBindGroup(1, rtBindGroup_set1[index], 0, 0);
						passEncoders[index]->setBindGroup(2, rtBindGroup_set2[index].get(), 0, 0);
						PushConstant pConst = {
							renderer->state.width,
							renderer->state.height,
							renderer->state.batchIdx,
							renderer->state.allBatch
						};
						passEncoders[index]->pushConstants(&pConst,
							(uint32_t)RHI::ShaderStages::RAYGEN
							| (uint32_t)RHI::ShaderStages::COMPUTE
							| (uint32_t)RHI::ShaderStages::CLOSEST_HIT
							| (uint32_t)RHI::ShaderStages::INTERSECTION
							| (uint32_t)RHI::ShaderStages::MISS
							| (uint32_t)RHI::ShaderStages::CALLABLE
							| (uint32_t)RHI::ShaderStages::ANY_HIT,
							0, sizeof(PushConstant));
						passEncoders[index]->traceRays(renderer->state.width, renderer->state.height, 1);
						passEncoders[index]->end();
						++renderer->state.batchIdx;
						++renderer->state.allBatch;
					}
				};
				return fn;
				});
		}
	};
}