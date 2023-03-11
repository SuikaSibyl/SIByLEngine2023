module;
#include <array>
#include <memory>
#include <compare>
#include <typeinfo>
#include <filesystem>
#include <functional>
#include "../../../Application/Public/SE.Application.Config.h"
export module SE.SRenderer:Tracer.SBDPT;
import :SRenderer;
import SE.Core.Resource;
import SE.RHI;
import SE.GFX.Core;
import SE.GFX.RDG;

namespace SIByL
{
	export struct BDPathTracerPass :public SRenderer::Pass {

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
			computetest = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();

			GFX::GFXManager::get()->registerShaderModuleResource(rgen, "../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/bdpt/bdpt_rgen.spv", { nullptr, RHI::ShaderStages::RAYGEN });
			GFX::GFXManager::get()->registerShaderModuleResource(rmiss, "../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/path_tracer/stracer_rmiss.spv", { nullptr, RHI::ShaderStages::MISS });
			GFX::GFXManager::get()->registerShaderModuleResource(computetest, "../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/bdpt/combine_atomic_rt_comp.spv", { nullptr, RHI::ShaderStages::COMPUTE });
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

			vert_fullscreen_pass = GFX::GFXManager::get()->registerShaderModuleResource("../Engine/Binaries/Runtime/spirv/SRenderer/rasterizer/fullscreen_pass/fullscreen_pass_vert.spv", { nullptr, RHI::ShaderStages::VERTEX });
			frag_clear_atomic_image = GFX::GFXManager::get()->registerShaderModuleResource("../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/bdpt/clear_atomic_rt_frag.spv", { nullptr, RHI::ShaderStages::FRAGMENT });
			frag_combine_atomic_image = GFX::GFXManager::get()->registerShaderModuleResource("../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/bdpt/combine_atomic_rt_frag.spv", { nullptr, RHI::ShaderStages::FRAGMENT });
		}

		Core::GUID rgen;
		Core::GUID rchit_trimesh;
		Core::GUID rchit_sphere;
		Core::GUID rmiss;

		Core::GUID shadow_ray_rchit;
		Core::GUID shadow_ray_rmiss;

		Core::GUID computetest;

		Core::GUID vert_fullscreen_pass, frag_clear_atomic_image, frag_combine_atomic_image;

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

		std::unique_ptr<RHI::BindGroupLayout> set2_layout_rt = 0;
		std::array<std::unique_ptr<RHI::BindGroup>, MULTIFRAME_FLIGHTS_COUNT> set2_flights_rt = {};
		std::array<RHI::BindGroup*, MULTIFRAME_FLIGHTS_COUNT> set2_flights_rt_array = {};

		virtual auto registerPass(SRenderer* renderer) noexcept -> void override {
			GFX::RDGraph* rdg = renderer->rdgraph;
			GFX::RDGPassNode* pass = rdg->addPass("BDPathTracerPass", GFX::RDGPassFlag::RAY_TRACING,
				[rdg = rdg]()->void {
					// consume
					rdg->getTexture("TracerTarget_Color")->consume(GFX::ConsumeType::READ_WRITE, RHI::TextureLayout::GENERAL, RHI::TextureUsage::STORAGE_BINDING);
					rdg->getTexture("TracerTarget_Color")->consume(GFX::ConsumeType::WRITE, RHI::TextureLayout::COLOR_ATTACHMENT_OPTIMAL, RHI::TextureUsage::COLOR_ATTACHMENT);
					rdg->getTexture("AtomicMutex")->consume(GFX::ConsumeType::READ_WRITE, RHI::TextureLayout::GENERAL, RHI::TextureUsage::STORAGE_BINDING);

					rdg->getTexture("AtomicRGB32")->consume(GFX::ConsumeType::WRITE, RHI::TextureLayout::COLOR_ATTACHMENT_OPTIMAL, RHI::TextureUsage::COLOR_ATTACHMENT);
					rdg->getTexture("AtomicRGB32")->consume(GFX::ConsumeType::READ_WRITE, RHI::TextureLayout::GENERAL, RHI::TextureUsage::STORAGE_BINDING);
				},
				[&, renderer = renderer, rdg = rdg]()->GFX::CustomPassExecuteFn {
					// bindgroup layout
					RHI::Device* device = GFX::GFXManager::get()->rhiLayer->getDevice();

					std::shared_ptr<RHI::PipelineLayout> clear_pipelineLayout = device->createPipelineLayout(RHI::PipelineLayoutDescriptor{
						{ {(uint32_t)RHI::ShaderStages::VERTEX, 0, sizeof(uint32_t)}},
						{ renderer->commonDescData.set0_layout.get()} });
					std::shared_ptr<RHI::RenderPipeline> clear_renderPipeline[MULTIFRAME_FLIGHTS_COUNT];
					for (int i = 0; i < MULTIFRAME_FLIGHTS_COUNT; ++i) {
						clear_renderPipeline[i] = device->createRenderPipeline(RHI::RenderPipelineDescriptor{
							clear_pipelineLayout.get(),
							RHI::VertexState{
								// vertex shader
								Core::ResourceManager::get()->getResource<GFX::ShaderModule>(vert_fullscreen_pass)->shaderModule.get(), "main",
								// vertex attribute layout
								{ }},
							RHI::PrimitiveState{ RHI::PrimitiveTopology::TRIANGLE_LIST, RHI::IndexFormat::UINT16_t },
							RHI::DepthStencilState{ },
							RHI::MultisampleState{},
							RHI::FragmentState{
								// fragment shader
								Core::ResourceManager::get()->getResource<GFX::ShaderModule>(frag_clear_atomic_image)->shaderModule.get(), "main",
								{{RHI::TextureFormat::R32_SINT}, {RHI::TextureFormat::R32_SINT}, {RHI::TextureFormat::R32_SINT}, {RHI::TextureFormat::R32_SINT}}}
							});
					}

					set2_layout_rt = device->createBindGroupLayout(
						RHI::BindGroupLayoutDescriptor{ {
							RHI::BindGroupLayoutEntry{ 0, (uint32_t)RHI::ShaderStages::RAYGEN | (uint32_t)RHI::ShaderStages::COMPUTE | (uint32_t)RHI::ShaderStages::FRAGMENT, RHI::StorageTextureBindingLayout{}},
							RHI::BindGroupLayoutEntry{ 1, (uint32_t)RHI::ShaderStages::RAYGEN | (uint32_t)RHI::ShaderStages::COMPUTE | (uint32_t)RHI::ShaderStages::FRAGMENT, RHI::StorageTextureBindingLayout{}},
							} });

					for (int i = 0; i < MULTIFRAME_FLIGHTS_COUNT; ++i) {
						set2_flights_rt[i] = device->createBindGroup(RHI::BindGroupDescriptor{
							set2_layout_rt.get(),
							std::vector<RHI::BindGroupEntry>{
								{0,RHI::BindingResource{rdg->getTexture("AtomicMutex")->texture->originalView.get()}},
								{1,RHI::BindingResource{rdg->getTexture("AtomicRGB32")->texture->originalView.get()}},
						} });
						set2_flights_rt_array[i] = set2_flights_rt[i].get();
					}


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
						  renderer->commonDescData.set1_layout_rt.get(),
						  set2_layout_rt.get() }, });
					std::shared_ptr<RHI::RayTracingPipeline> tracingPipeline[MULTIFRAME_FLIGHTS_COUNT];
					std::shared_ptr<RHI::ComputePipeline>	 computePipeline[MULTIFRAME_FLIGHTS_COUNT];
					for (int i = 0; i < MULTIFRAME_FLIGHTS_COUNT; ++i) {
						computePipeline[i] = device->createComputePipeline(RHI::ComputePipelineDescriptor{
								pipelineLayout.get(),
								{Core::ResourceManager::get()->getResource<GFX::ShaderModule>(computetest)->shaderModule.get(), "main"}
							});
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
					RHI::RenderPassDescriptor renderPassDescriptor = {
						{ RHI::RenderPassColorAttachment{ rdg->getTexture("AtomicRGB32")->texture->viewArrays[0].get(), nullptr, {0,0,0,1}, RHI::LoadOp::CLEAR, RHI::StoreOp::STORE},
						  RHI::RenderPassColorAttachment{ rdg->getTexture("AtomicRGB32")->texture->viewArrays[1].get(), nullptr, {0,0,0,1}, RHI::LoadOp::CLEAR, RHI::StoreOp::STORE },
						  RHI::RenderPassColorAttachment{ rdg->getTexture("AtomicRGB32")->texture->viewArrays[2].get(), nullptr, {0,0,0,1}, RHI::LoadOp::CLEAR, RHI::StoreOp::STORE },
						  RHI::RenderPassColorAttachment{ rdg->getTexture("AtomicRGB32")->texture->viewArrays[3].get(), nullptr, {0,0,0,1}, RHI::LoadOp::CLEAR, RHI::StoreOp::STORE }},
						RHI::RenderPassDepthStencilAttachment{},
					};


					// Combine atomic pass
					std::shared_ptr<RHI::PipelineLayout> combine_pipelineLayout = device->createPipelineLayout(RHI::PipelineLayoutDescriptor{
						{ {(uint32_t)RHI::ShaderStages::FRAGMENT, 0, sizeof(PushConstant)}},
						{ renderer->commonDescData.set0_layout.get(),
						  renderer->commonDescData.set1_layout_rt.get(),
						  set2_layout_rt.get()} });
					std::shared_ptr<RHI::RenderPipeline> combine_renderPipeline[MULTIFRAME_FLIGHTS_COUNT];
					for (int i = 0; i < MULTIFRAME_FLIGHTS_COUNT; ++i) {
						combine_renderPipeline[i] = device->createRenderPipeline(RHI::RenderPipelineDescriptor{
							combine_pipelineLayout.get(),
							RHI::VertexState{
								// vertex shader
								Core::ResourceManager::get()->getResource<GFX::ShaderModule>(vert_fullscreen_pass)->shaderModule.get(), "main",
								// vertex attribute layout
								{ }},
							RHI::PrimitiveState{ RHI::PrimitiveTopology::TRIANGLE_LIST, RHI::IndexFormat::UINT16_t },
							RHI::DepthStencilState{ },
							RHI::MultisampleState{},
							RHI::FragmentState{
								// fragment shader
								Core::ResourceManager::get()->getResource<GFX::ShaderModule>(frag_combine_atomic_image)->shaderModule.get(), "main",
								{{RHI::TextureFormat::RGBA32_FLOAT}}}
							});
					}
					RHI::RenderPassDescriptor combineRenderPassDescriptor = {
						{ RHI::RenderPassColorAttachment{ rdg->getTexture("TracerTarget_Color")->texture->originalView.get(), nullptr, {0,0,0,1}, RHI::LoadOp::CLEAR, RHI::StoreOp::STORE} },
						RHI::RenderPassDepthStencilAttachment{},
					};

					std::shared_ptr<RHI::RenderPassEncoder> clearPassEncoder[2] = {};
					std::shared_ptr<RHI::RenderPassEncoder> combinePassEncoder[2] = {};
					std::shared_ptr<RHI::RayTracingPassEncoder> passEncoder[2] = {};
					std::shared_ptr<RHI::ComputePassEncoder> computePassEncoder[2] = {};
					RHI::MultiFrameFlights* multiFrameFlights = GFX::GFXManager::get()->rhiLayer->getMultiFrameFlights();
					// execute callback
					GFX::CustomPassExecuteFn fn = [
						multiFrameFlights, &geometries = renderer->sceneDataPack.geometry_buffer_cpu, pipelineLayout = std::move(pipelineLayout),
							rdg = rdg,
							renderPassDescriptor = renderPassDescriptor,
							clear_pipelineLayout = std::move(clear_pipelineLayout),
							clear_renderPipeline = std::array<std::shared_ptr<RHI::RenderPipeline>, 2>{ std::move(clear_renderPipeline[0]), std::move(clear_renderPipeline[1]) },
							pipelines = std::array<std::shared_ptr<RHI::RayTracingPipeline>, 2>{ std::move(tracingPipeline[0]), std::move(tracingPipeline[1]) },
							cpipelines = std::array<std::shared_ptr<RHI::ComputePipeline>, 2>{ std::move(computePipeline[0]), std::move(computePipeline[1]) },
							rtBindGroup_set0 = renderer->commonDescData.set0_flights_array,
							rtBindGroup_set1 = renderer->commonDescData.set1_flights_rt_array,
							rtBindGroup_set2 = set2_flights_rt_array,
							renderer = renderer,
							clearPassEncoders = std::array<std::shared_ptr<RHI::RenderPassEncoder>, 2>{ std::move(clearPassEncoder[0]), std::move(clearPassEncoder[1]) },
							passEncoders = std::array<std::shared_ptr<RHI::RayTracingPassEncoder>, 2>{ std::move(passEncoder[0]), std::move(passEncoder[1]) },
							computePassEncoders = std::array<std::shared_ptr<RHI::ComputePassEncoder>, 2>{ std::move(computePassEncoder[0]), std::move(computePassEncoder[1])},
							// atomic combine pass
							combine_pipelineLayout = std::move(combine_pipelineLayout),
							combinePassEncoders = std::array<std::shared_ptr<RHI::RenderPassEncoder>, 2>{ std::move(combinePassEncoder[0]), std::move(combinePassEncoder[1])},
							combine_renderPipelines = std::array<std::shared_ptr<RHI::RenderPipeline>, 2>{ std::move(combine_renderPipeline[0]), std::move(combine_renderPipeline[1]) },
							combineRenderPassDescriptor = combineRenderPassDescriptor
					](GFX::RDGRegistry const& registry, RHI::CommandEncoder* cmdEncoder) mutable ->void {
						uint32_t index = multiFrameFlights->getFlightIndex();

						{	// clear atomic storage image
							clearPassEncoders[index] = cmdEncoder->beginRenderPass(renderPassDescriptor);
							clearPassEncoders[index]->setPipeline(clear_renderPipeline[index].get());
							int width = 800, height = 600;
							clearPassEncoders[index]->setViewport(0, 0, width, height, 0, 1);
							clearPassEncoders[index]->setScissorRect(0, 0, width, height);

							clearPassEncoders[index]->draw(3, 1, 0, 0);
							clearPassEncoders[index]->end();
						}

						cmdEncoder->pipelineBarrier(RHI::BarrierDescriptor{
							(uint32_t)RHI::PipelineStages::COLOR_ATTACHMENT_OUTPUT_BIT,
							(uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR,
							(uint32_t)RHI::DependencyType::NONE,
							{}, {},
							{ RHI::TextureMemoryBarrierDescriptor{
								rdg->getTexture("AtomicRGB32")->texture->texture.get(),
								RHI::ImageSubresourceRange{(uint32_t)RHI::TextureAspect::COLOR_BIT, 0,1,0,4},
								(uint32_t)RHI::AccessFlagBits::COLOR_ATTACHMENT_WRITE_BIT,
								(uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT | (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT,
								RHI::TextureLayout::SHADER_READ_ONLY_OPTIMAL,
								RHI::TextureLayout::GENERAL
							}}
							});

						PushConstant pConst = {
							renderer->state.width,
							renderer->state.height,
							renderer->state.batchIdx
						};

						//computePassEncoders[index] = cmdEncoder->beginComputePass(RHI::ComputePassDescriptor{});
						//computePassEncoders[index]->setPipeline(cpipelines[index].get());
						//computePassEncoders[index]->setBindGroup(0, rtBindGroup_set0[index], 0, 0);
						//computePassEncoders[index]->setBindGroup(1, rtBindGroup_set1[index], 0, 0);
						//computePassEncoders[index]->setBindGroup(2, rtBindGroup_set2[index], 0, 0);
						//computePassEncoders[index]->pushConstants(&pConst,
						//	(uint32_t)RHI::ShaderStages::RAYGEN
						//	| (uint32_t)RHI::ShaderStages::COMPUTE
						//	| (uint32_t)RHI::ShaderStages::CLOSEST_HIT
						//	| (uint32_t)RHI::ShaderStages::INTERSECTION
						//	| (uint32_t)RHI::ShaderStages::MISS
						//	| (uint32_t)RHI::ShaderStages::CALLABLE
						//	| (uint32_t)RHI::ShaderStages::ANY_HIT,
						//	0, sizeof(PushConstant));

						//computePassEncoders[index]->dispatchWorkgroups((renderer->state.width + 7) / 8, (renderer->state.height + 3) / 4, 1);
						//computePassEncoders[index]->end();
						//virtual auto setPipeline(ComputePipeline* pipeline) noexcept -> void override;
						///** Dispatch work to be performed with the current GPUComputePipeline.*/
						//virtual auto dispatchWorkgroups(uint32_t workgroupCountX, uint32_t workgroupCountY = 1, uint32_t workgroupCountZ = 1) noexcept -> void override;

						passEncoders[index] = cmdEncoder->beginRayTracingPass(RHI::RayTracingPassDescriptor{});
						if (renderer->sceneDataPack.geometry_buffer_cpu.size() > 0) {
							passEncoders[index]->setPipeline(pipelines[index].get());
							passEncoders[index]->setBindGroup(0, rtBindGroup_set0[index], 0, 0);
							passEncoders[index]->setBindGroup(1, rtBindGroup_set1[index], 0, 0);
							passEncoders[index]->setBindGroup(2, rtBindGroup_set2[index], 0, 0);
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

						cmdEncoder->pipelineBarrier(RHI::BarrierDescriptor{
							(uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR,
							(uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT,
							(uint32_t)RHI::DependencyType::NONE,
							{}, {},
							{ RHI::TextureMemoryBarrierDescriptor{
								rdg->getTexture("AtomicRGB32")->texture->texture.get(),
								RHI::ImageSubresourceRange{(uint32_t)RHI::TextureAspect::COLOR_BIT, 0,1,0,4},
								(uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT | (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT,
								(uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT,
								RHI::TextureLayout::GENERAL,
								RHI::TextureLayout::GENERAL
							}}
							});

						cmdEncoder->pipelineBarrier(RHI::BarrierDescriptor{
							(uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR,
							(uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT,
							(uint32_t)RHI::DependencyType::NONE,
							{}, {},
							{ RHI::TextureMemoryBarrierDescriptor{
								rdg->getTexture("TracerTarget_Color")->texture->texture.get(),
								RHI::ImageSubresourceRange{(uint32_t)RHI::TextureAspect::COLOR_BIT, 0,1,0,1},
								(uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT | (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT,
								(uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT | (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT,
								RHI::TextureLayout::GENERAL,
								RHI::TextureLayout::GENERAL
							}}
							});

						//{	// combine atomic storage image
						//	combinePassEncoders[index] = cmdEncoder->beginRenderPass(combineRenderPassDescriptor);
						//	combinePassEncoders[index]->setPipeline(combine_renderPipelines[index].get());
						//	int width = 800, height = 600;
						//	combinePassEncoders[index]->setViewport(0, 0, width, height, 0, 1);
						//	combinePassEncoders[index]->setScissorRect(0, 0, width, height);

						//	combinePassEncoders[index]->setBindGroup(0, rtBindGroup_set0[index], 0, 0);
						//	combinePassEncoders[index]->setBindGroup(1, rtBindGroup_set1[index], 0, 0);
						//	combinePassEncoders[index]->setBindGroup(2, rtBindGroup_set2[index], 0, 0);
						//	combinePassEncoders[index]->pushConstants(&pConst,
						//		(uint32_t)RHI::ShaderStages::FRAGMENT,
						//		0, sizeof(PushConstant));

						//	combinePassEncoders[index]->draw(3, 1, 0, 0);
						//	combinePassEncoders[index]->end();
						//}
						{	// combine atomic storage image
							computePassEncoders[index] = cmdEncoder->beginComputePass(RHI::ComputePassDescriptor{});
							computePassEncoders[index]->setPipeline(cpipelines[index].get());
							computePassEncoders[index]->setBindGroup(0, rtBindGroup_set0[index], 0, 0);
							computePassEncoders[index]->setBindGroup(1, rtBindGroup_set1[index], 0, 0);
							computePassEncoders[index]->setBindGroup(2, rtBindGroup_set2[index], 0, 0);
							computePassEncoders[index]->pushConstants(&pConst,
								(uint32_t)RHI::ShaderStages::RAYGEN
								| (uint32_t)RHI::ShaderStages::COMPUTE
								| (uint32_t)RHI::ShaderStages::CLOSEST_HIT
								| (uint32_t)RHI::ShaderStages::INTERSECTION
								| (uint32_t)RHI::ShaderStages::MISS
								| (uint32_t)RHI::ShaderStages::CALLABLE
								| (uint32_t)RHI::ShaderStages::ANY_HIT,
								0, sizeof(PushConstant));

							computePassEncoders[index]->dispatchWorkgroups((renderer->state.width + 15) / 16, (renderer->state.height + 15) / 16, 1);
							computePassEncoders[index]->end();
						}


						cmdEncoder->pipelineBarrier(RHI::BarrierDescriptor{
							(uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT,
							(uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT,
							(uint32_t)RHI::DependencyType::NONE,
							{}, {},
							{ RHI::TextureMemoryBarrierDescriptor{
								rdg->getTexture("TracerTarget_Color")->texture->texture.get(),
								RHI::ImageSubresourceRange{(uint32_t)RHI::TextureAspect::COLOR_BIT, 0,1,0,1},
								(uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT | (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT,
								(uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT,
								RHI::TextureLayout::GENERAL,
								RHI::TextureLayout::SHADER_READ_ONLY_OPTIMAL
							}}
							});
					};
					return fn;
				});
		}
	};
}