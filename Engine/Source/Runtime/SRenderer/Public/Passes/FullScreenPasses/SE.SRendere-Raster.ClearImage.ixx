module;
#include <array>
#include <memory>
#include <compare>
#include <typeinfo>
#include <filesystem>
#include <functional>
#include "../../../../Application/Public/SE.Application.Config.h"
export module SE.SRenderer:Raster.ClearImagePass;
import :SRenderer;
import SE.Core.Resource;
import SE.RHI;
import SE.GFX.Core;
import SE.GFX.RDG;

namespace SIByL
{
	export struct ClearImagePass :public SRenderer::Pass {

		virtual auto loadShaders() noexcept -> void override {
			vert = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();
			frag = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();
			GFX::GFXManager::get()->registerShaderModuleResource(vert, "../Engine/Binaries/Runtime/spirv/SRenderer/rasterizer/fullscreen_pass/fullscreen_pass_vert.spv", { nullptr, RHI::ShaderStages::VERTEX });
			GFX::GFXManager::get()->registerShaderModuleResource(frag, "../Engine/Binaries/Runtime/spirv/SRenderer/rasterizer/fullscreen_pass/clear_image_frag.spv", { nullptr, RHI::ShaderStages::FRAGMENT });
		}

		Core::GUID vert, frag;

		virtual auto registerPass(SRenderer* renderer) noexcept -> void override {
			GFX::RDGraph* rdg = renderer->rdgraph;
			GFX::RDGPassNode* pass = rdg->addPass("ClearImagePass", GFX::RDGPassFlag::RASTER,
				[rdg = rdg]()->void {
					// consume
					rdg->getTexture("AtomicMutex")->consume(GFX::ConsumeType::RENDER_TARGET, RHI::TextureLayout::COLOR_ATTACHMENT_OPTIMAL, RHI::TextureUsage::COLOR_ATTACHMENT);
				},
				[&, renderer = renderer, rdg = rdg]()->GFX::CustomPassExecuteFn {
					// bindgroup layout
					RHI::Device* device = GFX::GFXManager::get()->rhiLayer->getDevice();
				std::shared_ptr<RHI::PipelineLayout> pipelineLayout = device->createPipelineLayout(RHI::PipelineLayoutDescriptor{
					{ {(uint32_t)RHI::ShaderStages::VERTEX, 0, sizeof(uint32_t)}},
					{ renderer->commonDescData.set0_layout.get()} });
				std::shared_ptr<RHI::RenderPipeline> renderPipeline[MULTIFRAME_FLIGHTS_COUNT];
				for (int i = 0; i < MULTIFRAME_FLIGHTS_COUNT; ++i) {
					renderPipeline[i] = device->createRenderPipeline(RHI::RenderPipelineDescriptor{
						pipelineLayout.get(),
						RHI::VertexState{
							// vertex shader
							Core::ResourceManager::get()->getResource<GFX::ShaderModule>(vert)->shaderModule.get(), "main",
							// vertex attribute layout
							{ }},
						RHI::PrimitiveState{ RHI::PrimitiveTopology::TRIANGLE_LIST, RHI::IndexFormat::UINT16_t },
						RHI::DepthStencilState{ },
						RHI::MultisampleState{},
						RHI::FragmentState{
							// fragment shader
							Core::ResourceManager::get()->getResource<GFX::ShaderModule>(frag)->shaderModule.get(), "main",
							{{RHI::TextureFormat::R32_SINT}}}
						});
				}
				std::shared_ptr<RHI::RenderPassEncoder> passEncoder[2] = {};
				RHI::MultiFrameFlights* multiFrameFlights = GFX::GFXManager::get()->rhiLayer->getMultiFrameFlights();
				RHI::RenderPassDescriptor renderPassDescriptor = {
					{ RHI::RenderPassColorAttachment{
						rdg->getTexture("AtomicMutex")->texture->originalView.get(),
					nullptr, {0,0,0,1}, RHI::LoadOp::CLEAR, RHI::StoreOp::STORE }},
					RHI::RenderPassDepthStencilAttachment{},
				};
				// execute callback
				GFX::CustomPassExecuteFn fn = [
					multiFrameFlights, &geometries = renderer->sceneDataPack.geometry_buffer_cpu, renderPassDescriptor, pipelineLayout = std::move(pipelineLayout),
						irdg = rdg,
						pipelines = std::array<std::shared_ptr<RHI::RenderPipeline>, 2>{ std::move(renderPipeline[0]), std::move(renderPipeline[1]) },
						rtBindGroup = renderer->commonDescData.set0_flights_array, renderer = renderer,
						passEncoders = std::array<std::shared_ptr<RHI::RenderPassEncoder>, 2>{ std::move(passEncoder[0]), std::move(passEncoder[1]) }
				](GFX::RDGRegistry const& registry, RHI::CommandEncoder* cmdEncoder) mutable ->void {
					uint32_t index = multiFrameFlights->getFlightIndex();


					cmdEncoder->pipelineBarrier(RHI::BarrierDescriptor{
						(uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT,
						(uint32_t)RHI::PipelineStages::COLOR_ATTACHMENT_OUTPUT_BIT,
						(uint32_t)RHI::DependencyType::NONE,
						{}, {},
						{ RHI::TextureMemoryBarrierDescriptor{
							irdg->getTexture("AtomicMutex")->texture->texture.get(),
							RHI::ImageSubresourceRange{(uint32_t)RHI::TextureAspect::COLOR_BIT, 0,1,0,1},
							(uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT,
							(uint32_t)RHI::AccessFlagBits::COLOR_ATTACHMENT_WRITE_BIT,
							RHI::TextureLayout::SHADER_READ_ONLY_OPTIMAL,
							RHI::TextureLayout::COLOR_ATTACHMENT_OPTIMAL
						}}
						});

					passEncoders[index] = cmdEncoder->beginRenderPass(renderPassDescriptor);
					passEncoders[index]->setPipeline(pipelines[index].get());
					int width = 800, height = 600;
					passEncoders[index]->setViewport(0, 0, width, height, 0, 1);
					passEncoders[index]->setScissorRect(0, 0, width, height);

					passEncoders[index]->draw(3, 1, 0, 0);
					passEncoders[index]->end();

					cmdEncoder->pipelineBarrier(RHI::BarrierDescriptor{
						(uint32_t)RHI::PipelineStages::COLOR_ATTACHMENT_OUTPUT_BIT,
						(uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR,
						(uint32_t)RHI::DependencyType::NONE,
						{}, {},
						{ RHI::TextureMemoryBarrierDescriptor{
							irdg->getTexture("AtomicMutex")->texture->texture.get(),
							RHI::ImageSubresourceRange{(uint32_t)RHI::TextureAspect::COLOR_BIT, 0,1,0,1},
							(uint32_t)RHI::AccessFlagBits::COLOR_ATTACHMENT_WRITE_BIT,
							(uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT | (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT,
							RHI::TextureLayout::SHADER_READ_ONLY_OPTIMAL,
							RHI::TextureLayout::GENERAL
						}}
						});
				};
				return fn;
				});
		}
	};
}