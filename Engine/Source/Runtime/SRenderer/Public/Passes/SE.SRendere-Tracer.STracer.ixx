module;
#include <array>
#include <memory>
#include <compare>
#include <typeinfo>
#include <filesystem>
#include <functional>
#include "../../../Application/Public/SE.Application.Config.h"
export module SE.SRenderer:Tracer.STracer;
import :SRenderer;
import SE.Core.Resource;
import SE.RHI;
import SE.GFX.Core;
import SE.GFX.RDG;

namespace SIByL
{
	export struct PathTracerPass :public SRenderer::Pass {

		virtual auto loadShaders() noexcept -> void override {
			rgen = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();
			rmiss = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();
			rchit_trimesh = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();
			rchit_sphere = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();
			rint_sphere = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();

			shadow_ray_rchit = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();
			shadow_ray_rmiss = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();

			sphere_sampling_rcall = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();
			trimesh_sampling_rcall = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();
			sphere_sampling_pdf_rcall = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();
			trimesh_sampling_pdf_rcall = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();

			GFX::GFXManager::get()->registerShaderModuleResource(rgen, "../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/path_tracer/stracer_rgen.spv", { nullptr, RHI::ShaderStages::RAYGEN });
			GFX::GFXManager::get()->registerShaderModuleResource(rmiss, "../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/path_tracer/stracer_rmiss.spv", { nullptr, RHI::ShaderStages::MISS });
			GFX::GFXManager::get()->registerShaderModuleResource(rchit_trimesh, 
				"../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/path_tracer/spt_primary_ray_trimesh_rchit.spv", 
				{ nullptr, RHI::ShaderStages::CLOSEST_HIT });
			GFX::GFXManager::get()->registerShaderModuleResource(rchit_sphere,
				"../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/path_tracer/spt_primary_ray_sphere_rchit.spv", 
				{ nullptr, RHI::ShaderStages::CLOSEST_HIT });
			GFX::GFXManager::get()->registerShaderModuleResource(rint_sphere,
				"../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/custom_primitive/sphere_rint.spv", 
				{ nullptr, RHI::ShaderStages::INTERSECTION });

			GFX::GFXManager::get()->registerShaderModuleResource(shadow_ray_rchit,
				"../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/path_tracer/spt_shadow_ray_rchit.spv", 
				{ nullptr, RHI::ShaderStages::CLOSEST_HIT });
			GFX::GFXManager::get()->registerShaderModuleResource(shadow_ray_rmiss,
				"../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/path_tracer/spt_shadow_ray_rmiss.spv", 
				{ nullptr, RHI::ShaderStages::MISS });

			GFX::GFXManager::get()->registerShaderModuleResource(sphere_sampling_rcall,
				"../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/custom_primitive/sphere_sample_rcall.spv", 
				{ nullptr, RHI::ShaderStages::CALLABLE });
			GFX::GFXManager::get()->registerShaderModuleResource(trimesh_sampling_rcall,
				"../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/custom_primitive/trimesh_sample_rcall.spv", 
				{ nullptr, RHI::ShaderStages::CALLABLE });

			GFX::GFXManager::get()->registerShaderModuleResource(sphere_sampling_pdf_rcall,
				"../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/custom_primitive/sphere_sample_pdf_rcall.spv", 
				{ nullptr, RHI::ShaderStages::CALLABLE });
		}

		Core::GUID rgen;
		Core::GUID rchit_trimesh;
		Core::GUID rchit_sphere;
		Core::GUID rmiss;
		Core::GUID rint_sphere;

		Core::GUID shadow_ray_rchit;
		Core::GUID shadow_ray_rmiss;

		Core::GUID sphere_sampling_rcall;
		Core::GUID trimesh_sampling_rcall;
		Core::GUID sphere_sampling_pdf_rcall;
		Core::GUID trimesh_sampling_pdf_rcall;

		virtual auto registerPass(SRenderer* renderer) noexcept -> void override {
			GFX::RDGraph* rdg = renderer->rdgraph;
			GFX::RDGPassNode* pass = rdg->addPass("PathTracerPass", GFX::RDGPassFlag::RAY_TRACING,
				[rdg = rdg]()->void {
					// consume
					rdg->getTexture("TracerTarget_Color")->consume(GFX::ConsumeType::READ_WRITE, RHI::TextureLayout::GENERAL, RHI::TextureUsage::STORAGE_BINDING);
				},
				[&, renderer = renderer, rdg = rdg]()->GFX::CustomPassExecuteFn {
					// bindgroup layout
					RHI::Device* device = GFX::GFXManager::get()->rhiLayer->getDevice();
					struct PushConstant {
						uint32_t width;
						uint32_t height;
						uint32_t sample_batch;
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
						  renderer->commonDescData.set1_layout_rt.get() }, });
					std::shared_ptr<RHI::RayTracingPipeline> tracingPipeline[MULTIFRAME_FLIGHTS_COUNT];
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
									{Core::ResourceManager::get()->getResource<GFX::ShaderModule>(trimesh_sampling_rcall)->shaderModule.get()},
									{Core::ResourceManager::get()->getResource<GFX::ShaderModule>(sphere_sampling_pdf_rcall)->shaderModule.get()}, }},
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
						renderer = renderer,
						passEncoders = std::array<std::shared_ptr<RHI::RayTracingPassEncoder>, 2>{ std::move(passEncoder[0]), std::move(passEncoder[1]) }
				](GFX::RDGRegistry const& registry, RHI::CommandEncoder* cmdEncoder) mutable ->void {
					uint32_t index = multiFrameFlights->getFlightIndex();
					passEncoders[index] = cmdEncoder->beginRayTracingPass(RHI::RayTracingPassDescriptor{});
					if (renderer->sceneDataPack.geometry_buffer_cpu.size() > 0) {
						passEncoders[index]->setPipeline(pipelines[index].get());
						passEncoders[index]->setBindGroup(0, rtBindGroup_set0[index], 0, 0);
						passEncoders[index]->setBindGroup(1, rtBindGroup_set1[index], 0, 0);
						PushConstant pConst = {
							renderer->state.width,
							renderer->state.height,
							renderer->state.batchIdx
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
					}
				};
				return fn;
				});
		}
	};
}