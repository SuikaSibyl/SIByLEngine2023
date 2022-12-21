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
			rchit = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();
			rmiss = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();
			GFX::GFXManager::get()->registerShaderModuleResource(rgen, "../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/path_tracer/stracer_rgen.spv", { nullptr, RHI::ShaderStages::RAYGEN });
			GFX::GFXManager::get()->registerShaderModuleResource(rchit, "../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/path_tracer/stracer_rchit.spv", { nullptr, RHI::ShaderStages::CLOSEST_HIT });
			GFX::GFXManager::get()->registerShaderModuleResource(rmiss, "../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/path_tracer/stracer_rmiss.spv", { nullptr, RHI::ShaderStages::MISS });
		}

		Core::GUID rgen;
		Core::GUID rchit;
		Core::GUID rmiss;

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
					std::shared_ptr<RHI::PipelineLayout> pipelineLayout = device->createPipelineLayout(RHI::PipelineLayoutDescriptor{
						{ {(uint32_t)RHI::ShaderStages::RAYGEN
						 | (uint32_t)RHI::ShaderStages::COMPUTE
						 | (uint32_t)RHI::ShaderStages::CLOSEST_HIT
						 | (uint32_t)RHI::ShaderStages::MISS
						 | (uint32_t)RHI::ShaderStages::ANY_HIT, 0, sizeof(uint32_t)}},
						{ renderer->commonDescData.set0_layout.get(),
						  renderer->commonDescData.set1_layout_rt.get() }, });
					std::shared_ptr<RHI::RayTracingPipeline> tracingPipeline[MULTIFRAME_FLIGHTS_COUNT];
					for (int i = 0; i < MULTIFRAME_FLIGHTS_COUNT; ++i) {
						tracingPipeline[i] = device->createRayTracingPipeline(RHI::RayTracingPipelineDescriptor{
							pipelineLayout.get(), 3, RHI::SBTsDescriptor{
								RHI::SBTsDescriptor::RayGenerationSBT{{ Core::ResourceManager::get()->getResource<GFX::ShaderModule>(rgen)->shaderModule.get() }},
								RHI::SBTsDescriptor::MissSBT{{
									{Core::ResourceManager::get()->getResource<GFX::ShaderModule>(rmiss)->shaderModule.get()}, }},
								RHI::SBTsDescriptor::HitGroupSBT{{ 
									{Core::ResourceManager::get()->getResource<GFX::ShaderModule>(rchit)->shaderModule.get()}, }}
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
					passEncoders[index]->setPipeline(pipelines[index].get());
					passEncoders[index]->setBindGroup(0, rtBindGroup_set0[index], 0, 0);
					passEncoders[index]->setBindGroup(1, rtBindGroup_set1[index], 0, 0);
					passEncoders[index]->pushConstants(&renderer->state.batchIdx,
						(uint32_t)RHI::ShaderStages::RAYGEN
						| (uint32_t)RHI::ShaderStages::COMPUTE
						| (uint32_t)RHI::ShaderStages::CLOSEST_HIT
						| (uint32_t)RHI::ShaderStages::MISS
						| (uint32_t)RHI::ShaderStages::ANY_HIT,
						0, sizeof(uint32_t));
					passEncoders[index]->traceRays(800, 600, 1);
					passEncoders[index]->end();
				};
				return fn;
				});
		}
	};
}