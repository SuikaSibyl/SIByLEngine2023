#include <cstdint>
#include <typeinfo>
#include "../../../../../../Shaders/SRenderer/raytracer/mmlt/mmlt_config.h"
#include <SE.RHI.hpp>
#include <SE.GFX.hpp>
#include <SE.RDG.hpp>
#include <SE.SRenderer.hpp>
#include <Resource/SE.Core.Resource.hpp>

namespace SIByL
{
	SE_EXPORT struct MMLTBoostrapPass :public RDG::RayTracingPass {

		struct PushConstant {
			uint32_t width;
			uint32_t height;
			uint32_t sample_batch;
		};

		MMLTBoostrapPass() {
			boostrap_rgen = GFX::GFXManager::get()->registerShaderModuleResource("../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/mmlt/mmlt_boostrap_pass_rgen.spv", { nullptr, RHI::ShaderStages::RAYGEN });

			GFX::SBTsDescriptor sbt = RTCommon::get()->getSBTDescriptor();
			sbt.rgenSBT = GFX::SBTsDescriptor::RayGenerationSBT{ { Core::ResourceManager::get()->getResource<GFX::ShaderModule>(boostrap_rgen) } };

			RayTracingPass::init(sbt, 1);
		}

		virtual auto reflect() noexcept -> RDG::PassReflection override {
			RDG::PassReflection reflector;

			reflector.addOutput("BoostrapLuminance")
				.isTexture()
				.withFormat(RHI::TextureFormat::R32_FLOAT)
				.withSize(Math::ivec3(512, 512, 1))
				.withLevels(10)
				.withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING | (uint32_t)RHI::TextureUsage::TEXTURE_BINDING | (uint32_t)RHI::TextureUsage::COLOR_ATTACHMENT)
				.consume(RDG::TextureInfo::ConsumeEntry{ RDG::TextureInfo::ConsumeType::StorageBinding }
					.setSubresource(0, 1, 0, 1)
					.addStage((uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));

			return reflector;
		}

		virtual auto execute(RDG::RenderContext* context, RDG::RenderData const& renderData) noexcept -> void override {

			GFX::Texture* boostrapTex = renderData.getTexture("BoostrapLuminance");

			std::vector<RHI::BindGroupEntry>* set_0_entries = renderData.getBindGroupEntries("CommonScene");
			getBindGroup(context, 0)->updateBinding(*set_0_entries);
			std::vector<RHI::BindGroupEntry>* set_1_entries = renderData.getBindGroupEntries("CommonRT");
			getBindGroup(context, 1)->updateBinding(*set_1_entries);
			getBindGroup(context, 2)->updateBinding({ RHI::BindGroupEntry{ 0, RHI::BindingResource{ boostrapTex->getUAV(0,0,1)} } });

			RHI::RayTracingPassEncoder* encoder = beginPass(context);

			uint32_t batchIdx = renderData.getUInt("AccumIdx");
			Math::uvec2 tSize = renderData.getUVec2("TargetSize");
			PushConstant pConst = {
				tSize.x,
				tSize.y,
				batchIdx
			};
			encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::RAYGEN, 0, sizeof(PushConstant));
			encoder->traceRays(512 / 4, 512, 1);

			encoder->end();
		}

		Core::GUID boostrap_rgen;
	};

	SE_EXPORT struct MMLTMutationPass : public RDG::RayTracingPass {

		struct PushConstant {
			uint32_t width;
			uint32_t height;
			uint32_t sample_batch;
		};

		MMLTMutationPass() {
			mutation_rgen = GFX::GFXManager::get()->registerShaderModuleResource("../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/mmlt/mmlt_metroplis_pass_rgen.spv", { nullptr, RHI::ShaderStages::RAYGEN });

			GFX::SBTsDescriptor sbt = RTCommon::get()->getSBTDescriptor();
			sbt.rgenSBT = GFX::SBTsDescriptor::RayGenerationSBT{ { Core::ResourceManager::get()->getResource<GFX::ShaderModule>(mutation_rgen) } };

			RayTracingPass::init(sbt, 1);

			RHI::Device* device = GFX::GFXManager::get()->rhiLayer->getDevice();
			RHI::SamplerDescriptor samplerDesc = {};
			samplerDesc.maxLod = 9;
			mipLodSampler = device->createSampler(samplerDesc);
		}

		virtual auto reflect() noexcept -> RDG::PassReflection override {
			RDG::PassReflection reflector;

			reflector.addInput("boostrapMIP")
				.isTexture()
				.withFormat(RHI::TextureFormat::R32_FLOAT)
				.withUsages((uint32_t)RHI::TextureUsage::TEXTURE_BINDING)
				.consume(RDG::TextureInfo::ConsumeEntry{ RDG::TextureInfo::ConsumeType::StorageBinding }
					.setSubresource(0, 1, 0, 10)
					.addStage((uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT));

			reflector.addOutput("atomicRGBA")
				.isTexture()
				.withSize(Math::vec3(1, 1, 1))
				.withFormat(RHI::TextureFormat::R32_FLOAT)
				.withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
				.withLayers(4)
				.consume(RDG::TextureInfo::ConsumeEntry{ RDG::TextureInfo::ConsumeType::StorageBinding }
					.setSubresource(0, 1, 0, 4)
					.addStage((uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));

			reflector.addInternal("SampleStreamBuffer")
				.isBuffer()
				.withSize(sizeof(Math::vec4) * metroplis_buffer_width * metroplis_buffer_height * num_states_vec4)
				.withUsages((uint32_t)RHI::BufferUsage::STORAGE)
				.consume(RDG::BufferInfo::ConsumeEntry{}
					.setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT | (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
					.addStage((uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));

			reflector.addInternal("SampleInfoBuffer")
				.isBuffer()
				.withSize(sizeof(Math::vec4) * 2 * metroplis_buffer_width * metroplis_buffer_height)
				.withUsages((uint32_t)RHI::BufferUsage::STORAGE)
				.consume(RDG::BufferInfo::ConsumeEntry{}
					.setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT | (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
					.addStage((uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));

			return reflector;
		}

		virtual auto execute(RDG::RenderContext* context, RDG::RenderData const& renderData) noexcept -> void override {

			GFX::Texture* atomicRGBA = renderData.getTexture("atomicRGBA");
			GFX::Texture* boostrapMIP = renderData.getTexture("boostrapMIP");
			GFX::Buffer*  streamBuffer = renderData.getBuffer("SampleStreamBuffer");
			GFX::Buffer*  infoBuffer = renderData.getBuffer("SampleInfoBuffer");

			std::vector<RHI::BindGroupEntry>* set_0_entries = renderData.getBindGroupEntries("CommonScene");
			getBindGroup(context, 0)->updateBinding(*set_0_entries);
			std::vector<RHI::BindGroupEntry>* set_1_entries = renderData.getBindGroupEntries("CommonRT");
			getBindGroup(context, 1)->updateBinding(*set_1_entries);
			getBindGroup(context, 2)->updateBinding({ RHI::BindGroupEntry{ 0, RHI::BindingResource{ atomicRGBA->getUAV(0,0,4)} } });
			getBindGroup(context, 2)->updateBinding({ RHI::BindGroupEntry{ 1, RHI::BindingResource{RHI::BufferBinding{streamBuffer->buffer.get(), 0, streamBuffer->buffer->size()}}} });
			getBindGroup(context, 2)->updateBinding({ RHI::BindGroupEntry{ 2, RHI::BindingResource{RHI::BufferBinding{infoBuffer->buffer.get(), 0, infoBuffer->buffer->size()}}} });
			getBindGroup(context, 2)->updateBinding({ RHI::BindGroupEntry{ 3, RHI::BindingResource{RHI::BindingResource(std::vector<RHI::TextureView*>{boostrapMIP->getSRV(0,10,0,1)}, mipLodSampler.get())}} });

			RHI::RayTracingPassEncoder* encoder = beginPass(context);

			uint32_t batchIdx = renderData.getUInt("AccumIdx");
			PushConstant pConst = {
				atomicRGBA->texture->width(),
				atomicRGBA->texture->height(),
				batchIdx
			};
			encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::RAYGEN, 0, sizeof(PushConstant));
			encoder->traceRays(256, 256, 1);

			encoder->end();
		}

		std::unique_ptr<RHI::Sampler> mipLodSampler;
		Core::GUID mutation_rgen;
	};

	SE_EXPORT struct MMLTCombinePass : public RDG::ComputePass {

		MMLTCombinePass() {
			RHI::Device* device = GFX::GFXManager::get()->rhiLayer->getDevice();
			RHI::SamplerDescriptor samplerDesc = {};
			samplerDesc.maxLod = 9;
			mipLodSampler = device->createSampler(samplerDesc);

			comp = GFX::GFXManager::get()->registerShaderModuleResource(
				"../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/mmlt/mmlt_combine_atomic_comp.spv",
				{ nullptr, RHI::ShaderStages::COMPUTE });
			RDG::ComputePass::init(Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp));
		}

		virtual auto reflect() noexcept -> RDG::PassReflection {
			RDG::PassReflection reflector;

			reflector.addOutput("HDRAccum")
				.isTexture()
				.withSize(Math::vec3(1, 1, 1))
				.withFormat(RHI::TextureFormat::RGBA32_FLOAT)
				.withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
				.consume(RDG::TextureInfo::ConsumeEntry{ RDG::TextureInfo::ConsumeType::StorageBinding }
					.addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));

			reflector.addInput("I32RGBA")
				.isTexture()
				.withSize(Math::vec3(1, 1, 1))
				.withFormat(RHI::TextureFormat::R32_FLOAT)
				.withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
				.withLayers(4)
				.consume(RDG::TextureInfo::ConsumeEntry{ RDG::TextureInfo::ConsumeType::StorageBinding }
					.setSubresource(0, 1, 0, 4)
					.addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));

			reflector.addInput("BoostrapLuminance")
				.isTexture()
				.withFormat(RHI::TextureFormat::R32_FLOAT)
				.withUsages((uint32_t)RHI::TextureUsage::TEXTURE_BINDING)
				.consume(RDG::TextureInfo::ConsumeEntry{ RDG::TextureInfo::ConsumeType::StorageBinding }
					.setSubresource(0, 1, 0, 10)
					.addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));

			return reflector;
		}

		virtual auto execute(RDG::RenderContext* context, RDG::RenderData const& renderData) noexcept -> void {

			GFX::Texture* hdr = renderData.getTexture("HDRAccum");
			GFX::Texture* rgba = renderData.getTexture("I32RGBA");
			GFX::Texture* MIP = renderData.getTexture("BoostrapLuminance");

			getBindGroup(context, 0)->updateBinding(std::vector<RHI::BindGroupEntry>{
				RHI::BindGroupEntry{ 0, RHI::BindingResource{ hdr->getUAV(0,0,1)} },
				RHI::BindGroupEntry{ 1, RHI::BindingResource{ rgba->getUAV(0,0,4)} },
				RHI::BindGroupEntry{ 2, RHI::BindingResource{std::vector<RHI::TextureView*>{MIP->getSRV(0,10,0,1)}, mipLodSampler.get()} }
			});

			RHI::ComputePassEncoder* encoder = beginPass(context);

			uint32_t width = hdr->texture->width();
			uint32_t height = hdr->texture->height();
			uint32_t batchIdx = renderData.getUInt("AccumIdx");

			prepareDispatch(context);

			struct PushConstant {
				Math::uvec2 resolution;
				uint32_t sample_batch;
			};
			PushConstant pconst = {
				Math::uvec2{width,height},
				batchIdx
			};
			encoder->pushConstants(&pconst,
				(uint32_t)RHI::ShaderStages::COMPUTE,
				0, sizeof(PushConstant));
			encoder->dispatchWorkgroups((width + 15) / 16, (height + 15) / 16, 1);

			encoder->end();
		}

		std::unique_ptr<RHI::Sampler> mipLodSampler;
		Core::GUID comp;
	};
}