module;
#include <array>
#include <vector>
#include <memory>
#include <compare>
#include <typeinfo>
#include <filesystem>
#include <functional>
#include "../../../../Application/Public/SE.Application.Config.h"

export module SE.SRenderer:FullScreen.CreateSumUpMip;
import :SRenderer;
import SE.Core.Resource;
import SE.RHI;
import SE.GFX;

namespace SIByL
{
	//export struct CreateSumUpMipPass {

	//	auto setInput(GFX::Texture* texture, RHI::PipelineStageFlags previous_stage) noexcept -> void {
	//		target = texture;
	//		this->previous_stage = previous_stage;

	//		prepare();
	//	}

	//	auto loadShaders() noexcept -> void {
	//		vert = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();
	//		frag = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();
	//		GFX::GFXManager::get()->registerShaderModuleResource(vert, "../Engine/Binaries/Runtime/spirv/SRenderer/rasterizer/fullscreen_pass/fullscreen_pass_vert.spv", { nullptr, RHI::ShaderStages::VERTEX });
	//		GFX::GFXManager::get()->registerShaderModuleResource(frag, "../Engine/Binaries/Runtime/spirv/SRenderer/rasterizer/fullscreen_pass/sumup_mip_float32_frag.spv", { nullptr, RHI::ShaderStages::FRAGMENT });
	//	}

	//	Core::GUID vert, frag;

	//	GFX::Texture* target;
	//	RHI::PipelineStageFlags previous_stage;

	//	std::vector<std::array<std::unique_ptr<RHI::RenderPassEncoder>, 2>> mipPassEncoders;
	//	std::vector<std::array<std::unique_ptr<RHI::RenderPipeline>, 2>>	mipPassPipeline;
	//	std::vector<std::unique_ptr<RHI::BindGroup>> bindgroup;

	//	std::unique_ptr<RHI::BindGroupLayout> bindgroup_layout = 0;
	//	std::unique_ptr<RHI::PipelineLayout> pipelineLayout;

	//	void prepare() noexcept {
	//		mipPassEncoders.resize(target->texture->mipLevelCount() - 1);
	//		mipPassPipeline.resize(target->texture->mipLevelCount() - 1);
	//		bindgroup.resize(target->texture->mipLevelCount() - 1);

	//		RHI::Device* device = GFX::GFXManager::get()->rhiLayer->getDevice();
	//		// create bindgroup layout
	//		bindgroup_layout = device->createBindGroupLayout(
	//			RHI::BindGroupLayoutDescriptor{ {
	//					// only bind the source image
	//					RHI::BindGroupLayoutEntry{ 0, (uint32_t)RHI::ShaderStages::FRAGMENT, RHI::BindlessTexturesBindingLayout{}},
	//				} });

	//		// create bind group
	//		for (int i = 0; i < target->texture->mipLevelCount() - 1; ++i) {
	//			bindgroup[i] = device->createBindGroup(RHI::BindGroupDescriptor{
	//				bindgroup_layout.get(),
	//				std::vector<RHI::BindGroupEntry>{
	//					{0,RHI::BindingResource(
	//						std::vector<RHI::TextureView*>{target->getSRV(i,1,0,1)},
	//						Core::ResourceManager::get()->getResource<GFX::Sampler>(GFX::GFXManager::get()->commonSampler.clamp_nearest)->sampler.get())
	//						},
	//			} });
	//		}

	//		// create pipelines
	//		pipelineLayout = device->createPipelineLayout(RHI::PipelineLayoutDescriptor{
	//			{ {(uint32_t)RHI::ShaderStages::FRAGMENT, 0, sizeof(uint32_t)}},
	//			{ bindgroup_layout.get() }, });

	//		for (int i = 0; i < MULTIFRAME_FLIGHTS_COUNT; ++i) {
	//			for (int j = 0; j < target->texture->mipLevelCount() - 1; ++j) {
	//				mipPassPipeline[j][i] = device->createRenderPipeline(RHI::RenderPipelineDescriptor{
	//					pipelineLayout.get(),
	//					RHI::VertexState{
	//						// vertex shader
	//						Core::ResourceManager::get()->getResource<GFX::ShaderModule>(vert)->shaderModule.get(), "main",
	//						// vertex attribute layout
	//						{ }},
	//					RHI::PrimitiveState{ RHI::PrimitiveTopology::TRIANGLE_LIST, RHI::IndexFormat::UINT16_t },
	//					RHI::DepthStencilState{ RHI::TextureFormat::DEPTH32_FLOAT, true, RHI::CompareFunction::LESS },
	//					RHI::MultisampleState{},
	//					RHI::FragmentState{
	//						// fragment shader
	//						Core::ResourceManager::get()->getResource<GFX::ShaderModule>(frag)->shaderModule.get(), "main",
	//						{{RHI::TextureFormat::RGBA32_FLOAT}}}
	//					});

	//			}
	//		}
	//	}

	//	auto execute(GFX::RDGRegistry const& registry, RHI::CommandEncoder* cmdEncoder) noexcept -> void {

	//		uint32_t index = 1; // TODo:: = multiFrameFlights->getFlightIndex();


	//		cmdEncoder->pipelineBarrier(RHI::BarrierDescriptor{
	//			previous_stage,
	//			(uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT,
	//			(uint32_t)RHI::DependencyType::NONE,
	//			{}, {},
	//			{ RHI::TextureMemoryBarrierDescriptor{
	//				target->texture.get(),
	//				RHI::ImageSubresourceRange{(uint32_t)RHI::TextureAspect::COLOR_BIT, 0,1,0,1},
	//				(uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT | (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT,
	//				(uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT,
	//				RHI::TextureLayout::GENERAL,
	//				RHI::TextureLayout::SHADER_READ_ONLY_OPTIMAL
	//			}}
	//		});

	//		cmdEncoder->pipelineBarrier(RHI::BarrierDescriptor{
	//			(uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR,	// TODO :: change this
	//			(uint32_t)RHI::PipelineStages::COLOR_ATTACHMENT_OUTPUT_BIT,
	//			(uint32_t)RHI::DependencyType::NONE,
	//			{}, {},
	//			{ RHI::TextureMemoryBarrierDescriptor{
	//				target->texture.get(),
	//				RHI::ImageSubresourceRange{(uint32_t)RHI::TextureAspect::COLOR_BIT, 1,1,0,1},
	//				(uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT,
	//				(uint32_t)RHI::AccessFlagBits::COLOR_ATTACHMENT_WRITE_BIT,
	//				RHI::TextureLayout::SHADER_READ_ONLY_OPTIMAL,
	//				RHI::TextureLayout::COLOR_ATTACHMENT_OPTIMAL
	//			}}
	//			});

	//		uint32_t src_size = target->texture->width();

	//		for (int i = 0; i < target->texture->mipLevelCount() - 1; ++i) {

	//			if (i != 0) {
	//				cmdEncoder->pipelineBarrier(RHI::BarrierDescriptor{
	//					(uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT,	// TODO :: change this
	//					(uint32_t)RHI::PipelineStages::COLOR_ATTACHMENT_OUTPUT_BIT,
	//					(uint32_t)RHI::DependencyType::NONE,
	//					{}, {},
	//					{ RHI::TextureMemoryBarrierDescriptor{
	//						target->texture.get(),
	//						RHI::ImageSubresourceRange{(uint32_t)RHI::TextureAspect::COLOR_BIT, uint32_t(i+1),1,0,1},
	//						(uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT,
	//						(uint32_t)RHI::AccessFlagBits::COLOR_ATTACHMENT_WRITE_BIT,
	//						RHI::TextureLayout::SHADER_READ_ONLY_OPTIMAL,
	//						RHI::TextureLayout::COLOR_ATTACHMENT_OPTIMAL
	//					}}
	//					});
	//			}
	//			RHI::RenderPassDescriptor renderPassDescriptor = {
	//				{ RHI::RenderPassColorAttachment{
	//					target->getRTV(i + 1, 0, 1),
	//					nullptr, {0,0,0,1}, RHI::LoadOp::CLEAR, RHI::StoreOp::STORE }},
	//				RHI::RenderPassDepthStencilAttachment{},
	//			};

	//			mipPassEncoders[i][index] = cmdEncoder->beginRenderPass(renderPassDescriptor);

	//			mipPassEncoders[i][index]->setPipeline(mipPassPipeline[i][index].get());

	//			mipPassEncoders[i][index]->setBindGroup(0, bindgroup[i].get());
	//			mipPassEncoders[i][index]->pushConstants(&src_size,
	//				(uint32_t)RHI::ShaderStages::FRAGMENT,
	//				0, sizeof(uint32_t));

	//			src_size /= 2;
	//			mipPassEncoders[i][index]->setViewport(0, 0, src_size, src_size, 0, 1);
	//			mipPassEncoders[i][index]->setScissorRect(0, 0, src_size, src_size);

	//			mipPassEncoders[i][index]->draw(3, 1, 0, 0);
	//			mipPassEncoders[i][index]->end();
	//		}
	//	}
	//};
}