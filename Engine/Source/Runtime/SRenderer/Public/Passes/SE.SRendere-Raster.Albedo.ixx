module;
#include <array>
#include <memory>
#include <compare>
#include <typeinfo>
#include <filesystem>
#include <functional>
#include "../../../Application/Public/SE.Application.Config.h"
export module SE.SRenderer:Raster.Albedo;
import :SRenderer;
import SE.Core.Resource;
import SE.RHI;
import SE.GFX.Core;
import SE.GFX.RDG;

namespace SIByL 
{
	export struct AlbedoOnlyPass :public SRenderer::Pass {

		virtual auto loadShaders() noexcept -> void override {
			vert = GFX::GFXManager::get()->registerShaderModuleResource("../Engine/Binaries/Runtime/spirv/SRenderer/rasterizer/basecolor_only_pass/basecolor_only_vert.spv", { nullptr, RHI::ShaderStages::VERTEX });
			frag = GFX::GFXManager::get()->registerShaderModuleResource("../Engine/Binaries/Runtime/spirv/SRenderer/rasterizer/basecolor_only_pass/basecolor_only_frag.spv", { nullptr, RHI::ShaderStages::FRAGMENT });
		}

		Core::GUID vert, frag;

		virtual auto registerPass(SRenderer* renderer) noexcept -> void override {
			GFX::RDGraph* rdg = renderer->rdgraph;
			GFX::RDGPassNode* pass = rdg->addPass("AlbedoOnlyPass", GFX::RDGPassFlag::RASTER,
				[rdg=rdg]()->void {
					// consume
					rdg->getTexture("RasterizerTarget_Color")->consume(GFX::ConsumeType::RENDER_TARGET, RHI::TextureLayout::COLOR_ATTACHMENT_OPTIMAL, RHI::TextureUsage::COLOR_ATTACHMENT);
					rdg->getTexture("RasterizerTarget_Depth")->consume(GFX::ConsumeType::RENDER_TARGET, RHI::TextureLayout::DEPTH_ATTACHMENT_OPTIMAL, RHI::TextureUsage::DEPTH_ATTACHMENT);
				},
				[&, renderer = renderer, rdg = rdg] ()->GFX::CustomPassExecuteFn {
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
							RHI::DepthStencilState{ RHI::TextureFormat::DEPTH32_FLOAT, true, RHI::CompareFunction::LESS },
							RHI::MultisampleState{},
							RHI::FragmentState{
								// fragment shader
								Core::ResourceManager::get()->getResource<GFX::ShaderModule>(frag)->shaderModule.get(), "main",
								{{RHI::TextureFormat::RGBA32_FLOAT}}}
							});
					}
					std::shared_ptr<RHI::RenderPassEncoder> passEncoder[2] = {};
					RHI::MultiFrameFlights* multiFrameFlights = GFX::GFXManager::get()->rhiLayer->getMultiFrameFlights();
					RHI::RenderPassDescriptor renderPassDescriptor = {
						{ RHI::RenderPassColorAttachment{
							rdg->getTexture("RasterizerTarget_Color")->texture->originalView.get(),
						nullptr, {0,0,0,1}, RHI::LoadOp::CLEAR, RHI::StoreOp::STORE }},
						RHI::RenderPassDepthStencilAttachment{
							rdg->getTexture("RasterizerTarget_Depth")->texture->originalView.get(),
							1, RHI::LoadOp::CLEAR, RHI::StoreOp::DONT_CARE, false,
							0, RHI::LoadOp::CLEAR, RHI::StoreOp::DONT_CARE, false
					},
					};
					// execute callback
					GFX::CustomPassExecuteFn fn = [
						multiFrameFlights, &geometries = renderer->sceneDataPack.geometry_buffer_cpu, renderPassDescriptor, pipelineLayout = std::move(pipelineLayout),
							pipelines = std::array<std::shared_ptr<RHI::RenderPipeline>, 2>{ std::move(renderPipeline[0]), std::move(renderPipeline[1]) },
							rtBindGroup = renderer->commonDescData.set0_flights_array, renderer = renderer,
							passEncoders = std::array<std::shared_ptr<RHI::RenderPassEncoder>, 2>{ std::move(passEncoder[0]), std::move(passEncoder[1]) }
					](GFX::RDGRegistry const& registry, RHI::CommandEncoder* cmdEncoder) mutable ->void {
						uint32_t index = multiFrameFlights->getFlightIndex();
						passEncoders[index] = cmdEncoder->beginRenderPass(renderPassDescriptor);
						passEncoders[index]->setPipeline(pipelines[index].get());
						int width = 800, height = 600;
						passEncoders[index]->setViewport(0, 0, width, height, 0, 1);
						passEncoders[index]->setScissorRect(0, 0, width, height);
						if (renderer->sceneDataPack.geometry_buffer_cpu.size() > 0) {
							passEncoders[index]->setIndexBuffer(renderer->sceneDataPack.index_buffer.get(),
								RHI::IndexFormat::UINT32_T, 0, renderer->sceneDataPack.index_buffer->size());
							passEncoders[index]->setBindGroup(0, rtBindGroup[index], 0, 0);
							uint32_t geometry_idx = 0;
							for (auto& geometry : geometries) {
								passEncoders[index]->pushConstants(&geometry_idx, (uint32_t)RHI::ShaderStages::VERTEX, 0, sizeof(uint32_t));
								passEncoders[index]->drawIndexed(geometry.indexSize, 1, geometry.indexOffset, geometry.vertexOffset, 0);
								geometry_idx++;
							}
						}
						passEncoders[index]->end();
					};
					return fn;
				});
		}
	};
}