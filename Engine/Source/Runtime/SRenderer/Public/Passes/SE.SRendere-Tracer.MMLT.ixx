module;
#include <array>
#include <memory>
#include <compare>
#include <typeinfo>
#include <filesystem>
#include <functional>
#include "../../../Application/Public/SE.Application.Config.h"
#include "../../../../../Shaders/SRenderer/raytracer/mmlt/mmlt_config.h"
export module SE.SRenderer:Tracer.MMLTPass;
import :SRenderer;
import :FullScreen.CreateSumUpMip;
import SE.Core.Resource;
import SE.RHI;
import SE.GFX.Core;
import SE.GFX.RDG;

namespace SIByL
{
	export struct MMLTPass :public SRenderer::Pass {

		CreateSumUpMipPass sumUpMipCreateSubpass;

		virtual auto loadShaders() noexcept -> void override {

			sumUpMipCreateSubpass.loadShaders();

			vert_fullscreen_pass = GFX::GFXManager::get()->registerShaderModuleResource("../Engine/Binaries/Runtime/spirv/SRenderer/rasterizer/fullscreen_pass/fullscreen_pass_vert.spv", { nullptr, RHI::ShaderStages::VERTEX });
			frag_clear_atomic_image = GFX::GFXManager::get()->registerShaderModuleResource("../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/bdpt/clear_atomic_rt_frag.spv", { nullptr, RHI::ShaderStages::FRAGMENT });
			combine_atomic_image_comp = GFX::GFXManager::get()->registerShaderModuleResource("../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/mmlt/mmlt_combine_atomic_comp.spv", { nullptr, RHI::ShaderStages::COMPUTE });
			
			mmlt_boostrap_rgen = GFX::GFXManager::get()->registerShaderModuleResource("../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/mmlt/mmlt_boostrap_pass_rgen.spv", { nullptr, RHI::ShaderStages::RAYGEN });
			mmlt_metroplis_rgen = GFX::GFXManager::get()->registerShaderModuleResource("../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/mmlt/mmlt_metroplis_pass_rgen.spv", { nullptr, RHI::ShaderStages::RAYGEN });
		}

		Core::GUID vert_fullscreen_pass, frag_clear_atomic_image, combine_atomic_image_comp;

		Core::GUID mmlt_boostrap_rgen;
		Core::GUID mmlt_metroplis_rgen;
		Core::GUID texture_id;
		GFX::Texture* texture;

		std::unique_ptr<RHI::Buffer> pss_sample_stream_buffer;
		std::unique_ptr<RHI::Buffer> pss_sample_info_buffer;

		std::unique_ptr<RHI::BindGroupLayout> set2_layout_rt = 0;
		std::array<std::unique_ptr<RHI::BindGroup>, MULTIFRAME_FLIGHTS_COUNT> set2_flights_rt = {};
		std::array<RHI::BindGroup*, MULTIFRAME_FLIGHTS_COUNT> set2_flights_rt_array = {};

		std::unique_ptr<RHI::Sampler> mipLodSampler;

		virtual auto registerPass(SRenderer* renderer) noexcept -> void override {

			texture_id = GFX::GFXManager::get()->registerTextureResource("./content/texture.jpg");
			texture = Core::ResourceManager::get()->getResource<GFX::Texture>(texture_id);

			RHI::Device* device = GFX::GFXManager::get()->rhiLayer->getDevice();
			pss_sample_stream_buffer = device->createBuffer(RHI::BufferDescriptor{
				metroplis_buffer_width* metroplis_buffer_height * sizeof(float) * 4 * num_states_vec4,
				(RHI::BufferUsagesFlags)RHI::BufferUsage::STORAGE
				});
			pss_sample_info_buffer = device->createBuffer(RHI::BufferDescriptor{
				metroplis_buffer_width* metroplis_buffer_height * sizeof(float) * 8,
				(RHI::BufferUsagesFlags)RHI::BufferUsage::STORAGE
				});
			RHI::SamplerDescriptor samplerDesc = {

			};
			samplerDesc.maxLod = 9;
			mipLodSampler = device->createSampler(samplerDesc);

			GFX::RDGraph* rdg = renderer->rdgraph;
			GFX::RDGPassNode* pass = rdg->addPass("MMLTPass", GFX::RDGPassFlag::RAY_TRACING,
				[rdg = rdg]()->void {
					// consume
					rdg->getTexture("TracerTarget_Color")->consume(GFX::ConsumeType::READ_WRITE, RHI::TextureLayout::GENERAL, RHI::TextureUsage::STORAGE_BINDING);
					rdg->getTexture("TracerTarget_Color")->consume(GFX::ConsumeType::WRITE, RHI::TextureLayout::COLOR_ATTACHMENT_OPTIMAL, RHI::TextureUsage::COLOR_ATTACHMENT);
					rdg->getTexture("AtomicMutex")->consume(GFX::ConsumeType::READ_WRITE, RHI::TextureLayout::GENERAL, RHI::TextureUsage::STORAGE_BINDING);

					rdg->getTexture("AtomicRGB32")->consume(GFX::ConsumeType::WRITE, RHI::TextureLayout::COLOR_ATTACHMENT_OPTIMAL, RHI::TextureUsage::COLOR_ATTACHMENT);
					rdg->getTexture("AtomicRGB32")->consume(GFX::ConsumeType::READ_WRITE, RHI::TextureLayout::GENERAL, RHI::TextureUsage::STORAGE_BINDING);
					rdg->getTexture("boostrapLuminance")->consume(GFX::ConsumeType::READ_WRITE, RHI::TextureLayout::GENERAL, RHI::TextureUsage::STORAGE_BINDING);
					rdg->getTexture("boostrapLuminance")->consume(GFX::ConsumeType::READ, RHI::TextureLayout::SHADER_READ_ONLY_OPTIMAL, RHI::TextureUsage::TEXTURE_BINDING);
					rdg->getTexture("boostrapLuminance")->consume(GFX::ConsumeType::WRITE, RHI::TextureLayout::COLOR_ATTACHMENT_OPTIMAL, RHI::TextureUsage::COLOR_ATTACHMENT);
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
							RHI::BindGroupLayoutEntry{ 2, (uint32_t)RHI::ShaderStages::RAYGEN | (uint32_t)RHI::ShaderStages::COMPUTE | (uint32_t)RHI::ShaderStages::FRAGMENT, RHI::StorageTextureBindingLayout{}},
							RHI::BindGroupLayoutEntry{ 3, (uint32_t)RHI::ShaderStages::RAYGEN | (uint32_t)RHI::ShaderStages::COMPUTE | (uint32_t)RHI::ShaderStages::FRAGMENT, RHI::BufferBindingLayout{RHI::BufferBindingType::STORAGE}},
							RHI::BindGroupLayoutEntry{ 4, (uint32_t)RHI::ShaderStages::RAYGEN | (uint32_t)RHI::ShaderStages::COMPUTE | (uint32_t)RHI::ShaderStages::FRAGMENT, RHI::BufferBindingLayout{RHI::BufferBindingType::STORAGE}},
							RHI::BindGroupLayoutEntry{ 5, (uint32_t)RHI::ShaderStages::RAYGEN | (uint32_t)RHI::ShaderStages::COMPUTE | (uint32_t)RHI::ShaderStages::FRAGMENT, RHI::BindlessTexturesBindingLayout{}},
							RHI::BindGroupLayoutEntry{ 6, (uint32_t)RHI::ShaderStages::RAYGEN | (uint32_t)RHI::ShaderStages::COMPUTE | (uint32_t)RHI::ShaderStages::FRAGMENT, RHI::BindlessTexturesBindingLayout{}},
							} });

					for (int i = 0; i < MULTIFRAME_FLIGHTS_COUNT; ++i) {
						set2_flights_rt[i] = device->createBindGroup(RHI::BindGroupDescriptor{
							set2_layout_rt.get(),
							std::vector<RHI::BindGroupEntry>{
								{0,RHI::BindingResource{rdg->getTexture("AtomicMutex")->texture->originalView.get()}},
								{1,RHI::BindingResource{rdg->getTexture("AtomicRGB32")->texture->originalView.get()}},
								{2,RHI::BindingResource{rdg->getTexture("boostrapLuminance")->texture->originalView.get()}},
								{3,RHI::BindingResource{RHI::BufferBinding{pss_sample_stream_buffer.get(), 0, pss_sample_stream_buffer->size()}}},
								{4,RHI::BindingResource{RHI::BufferBinding{pss_sample_info_buffer.get(), 0, pss_sample_info_buffer->size()}}},
								{5,RHI::BindingResource(
									std::vector<RHI::TextureView*>{rdg->getTexture("boostrapLuminance")->texture->getSRV(0,10,0,1)},
									mipLodSampler.get())
									},
								{6,RHI::BindingResource(
									std::vector<RHI::TextureView*>{texture->getSRV(0,1,0,1)},
									Core::ResourceManager::get()->getResource<GFX::Sampler>(GFX::GFXManager::get()->commonSampler.defaultSampler)->sampler.get())
									},
						} });
						set2_flights_rt_array[i] = set2_flights_rt[i].get();
					}

					sumUpMipCreateSubpass.setInput(rdg->getTexture("boostrapLuminance")->texture, (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR);


					struct PushConstant {
						uint32_t width;
						uint32_t height;
						uint32_t sample_batch;
						uint32_t time_stamp;
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
					std::shared_ptr<RHI::RayTracingPipeline> boostrapPipeline[MULTIFRAME_FLIGHTS_COUNT];
					std::shared_ptr<RHI::RayTracingPipeline> metroplisPipeline[MULTIFRAME_FLIGHTS_COUNT];
					std::shared_ptr<RHI::ComputePipeline>	 computePipeline[MULTIFRAME_FLIGHTS_COUNT];

					RHI::RayTracingPipelineDescriptor boostrap_pass_descriptor = renderer->rtCommon.getPipelineDescriptor();
					boostrap_pass_descriptor.layout = pipelineLayout.get();
					boostrap_pass_descriptor.sbtsDescriptor.rgenSBT = RHI::SBTsDescriptor::RayGenerationSBT{ { Core::ResourceManager::get()->getResource<GFX::ShaderModule>(mmlt_boostrap_rgen)->shaderModule.get() } };

					RHI::RayTracingPipelineDescriptor metroplis_pass_descriptor = renderer->rtCommon.getPipelineDescriptor();
					metroplis_pass_descriptor.layout = pipelineLayout.get();
					metroplis_pass_descriptor.sbtsDescriptor.rgenSBT = RHI::SBTsDescriptor::RayGenerationSBT{ { Core::ResourceManager::get()->getResource<GFX::ShaderModule>(mmlt_metroplis_rgen)->shaderModule.get() } };

					for (int i = 0; i < MULTIFRAME_FLIGHTS_COUNT; ++i) {
						computePipeline[i] = device->createComputePipeline(RHI::ComputePipelineDescriptor{
								pipelineLayout.get(),
								{Core::ResourceManager::get()->getResource<GFX::ShaderModule>(combine_atomic_image_comp)->shaderModule.get(), "main"}
							});

						boostrapPipeline[i] = device->createRayTracingPipeline(boostrap_pass_descriptor);
						metroplisPipeline[i] = device->createRayTracingPipeline(metroplis_pass_descriptor);
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
						pipelines = std::array<std::shared_ptr<RHI::RayTracingPipeline>, 2>{ std::move(boostrapPipeline[0]), std::move(boostrapPipeline[1]) },
						metro_pipelines = std::array<std::shared_ptr<RHI::RayTracingPipeline>, 2>{ std::move(metroplisPipeline[0]), std::move(metroplisPipeline[1]) },
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
						combineRenderPassDescriptor = combineRenderPassDescriptor,
						pSumUpMipCreateSubpass = &sumUpMipCreateSubpass,
						ppss_sample_stream_buffer = pss_sample_stream_buffer.get(),
						ppss_sample_info_buffer = pss_sample_info_buffer.get()
				](GFX::RDGRegistry const& registry, RHI::CommandEncoder* cmdEncoder) mutable ->void {
						uint32_t index = multiFrameFlights->getFlightIndex();

						static uint32_t timestamp = 0;

						if (renderer->state.batchIdx == 0)
						{	
							cmdEncoder->pipelineBarrier(RHI::BarrierDescriptor{
								(uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT,
								(uint32_t)RHI::PipelineStages::COLOR_ATTACHMENT_OUTPUT_BIT,
								(uint32_t)RHI::DependencyType::NONE,
								{}, {},
								{ RHI::TextureMemoryBarrierDescriptor{
									rdg->getTexture("AtomicRGB32")->texture->texture.get(),
									RHI::ImageSubresourceRange{(uint32_t)RHI::TextureAspect::COLOR_BIT, 0,1,0,4},
									(uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT,
									(uint32_t)RHI::AccessFlagBits::COLOR_ATTACHMENT_WRITE_BIT,
									RHI::TextureLayout::SHADER_READ_ONLY_OPTIMAL,
									RHI::TextureLayout::COLOR_ATTACHMENT_OPTIMAL
								}}
							});
							
							// clear atomic storage image
							clearPassEncoders[index] = cmdEncoder->beginRenderPass(renderPassDescriptor);
							clearPassEncoders[index]->setPipeline(clear_renderPipeline[index].get());
							int width = 800, height = 600;
							clearPassEncoders[index]->setViewport(0, 0, width, height, 0, 1);
							clearPassEncoders[index]->setScissorRect(0, 0, width, height);

							clearPassEncoders[index]->draw(3, 1, 0, 0);
							clearPassEncoders[index]->end();

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
						}
						else {
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
						}

						PushConstant pConst = {
							renderer->state.width,
							renderer->state.height,
							renderer->state.batchIdx,
							timestamp++
						};

						cmdEncoder->pipelineBarrier(RHI::BarrierDescriptor{
							(uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR,
							(uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR,
							(uint32_t)RHI::DependencyType::NONE,
							{}, {},
							{ RHI::TextureMemoryBarrierDescriptor{
								rdg->getTexture("boostrapLuminance")->texture->texture.get(),
								RHI::ImageSubresourceRange{(uint32_t)RHI::TextureAspect::COLOR_BIT, 0,1,0,1},
								(uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT | (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT,
								(uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT | (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT,
								RHI::TextureLayout::SHADER_READ_ONLY_OPTIMAL,
								RHI::TextureLayout::GENERAL
							}}
							});

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
							passEncoders[index]->traceRays(512 / 4, 512, 1);
							passEncoders[index]->end();
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

						// execute sum up mip creation
						pSumUpMipCreateSubpass->execute(registry, cmdEncoder);

						{	// barriers mip creation pass <---> metro-pass 
							cmdEncoder->pipelineBarrier(RHI::BarrierDescriptor{
								(uint32_t)RHI::PipelineStages::COLOR_ATTACHMENT_OUTPUT_BIT,
								(uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR | (uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT,
								(uint32_t)RHI::DependencyType::NONE,
								{}, {},
								{ RHI::TextureMemoryBarrierDescriptor{
									rdg->getTexture("boostrapLuminance")->texture->texture.get(),
									RHI::ImageSubresourceRange{(uint32_t)RHI::TextureAspect::COLOR_BIT, 0,10,0,1},
									(uint32_t)RHI::AccessFlagBits::COLOR_ATTACHMENT_WRITE_BIT,
									(uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT,
									RHI::TextureLayout::SHADER_READ_ONLY_OPTIMAL,
									RHI::TextureLayout::SHADER_READ_ONLY_OPTIMAL
								}}
								});
							// barriers boostrap-pass <---> metro-pass 
							cmdEncoder->pipelineBarrier(RHI::BarrierDescriptor{
								(uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR,
								(uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR,
								(uint32_t)RHI::DependencyType::NONE,
								{}, { RHI::BufferMemoryBarrierDescriptor{
									ppss_sample_stream_buffer,
									(uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT | (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT,
									(uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT | (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT,
								},	RHI::BufferMemoryBarrierDescriptor{
									ppss_sample_info_buffer,
									(uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT | (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT,
									(uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT | (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT,
								} },
								{}
								});
						}
						{	// metro-pass
							passEncoders[index] = cmdEncoder->beginRayTracingPass(RHI::RayTracingPassDescriptor{});
							if (renderer->sceneDataPack.geometry_buffer_cpu.size() > 0) {
								passEncoders[index]->setPipeline(metro_pipelines[index].get());
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
								passEncoders[index]->traceRays(metroplis_buffer_width, metroplis_buffer_height, 1);
								passEncoders[index]->end();
							}
						}
						{	// barriers metro-pass <---> atomic combine pass 
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
						}
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


						++renderer->state.batchIdx;
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

						cmdEncoder->pipelineBarrier(RHI::BarrierDescriptor{
							(uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR,
							(uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT,
							(uint32_t)RHI::DependencyType::NONE,
							{}, {},
							{ RHI::TextureMemoryBarrierDescriptor{
								rdg->getTexture("AtomicRGB32")->texture->texture.get(),
								RHI::ImageSubresourceRange{(uint32_t)RHI::TextureAspect::COLOR_BIT, 0,1,0,4},
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