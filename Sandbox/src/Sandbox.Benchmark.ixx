module;
#include <array>
#include <filesystem>
#include <memory>
#include <utility>
export module Sandbox.Benchmark;
import SE.Core.Resource;
import SE.Math.Geometric;
import SE.RHI;
import SE.GFX.Core;

using namespace SIByL;

namespace Sandbox
{
	export struct Benchmark_Tracing_Pass {

		Benchmark_Tracing_Pass(RHI::RHILayer* rhiLayer, RHI::PipelineLayout* rtPipelineLayout,
			std::array<RHI::BindGroup*, 2> const& rtBindGroups,
			std::array<RHI::BindGroup*, 2> const& camBindGroups,
			std::array<RHI::BindGroup*, 2> const& bufferBindGroups)
			: rtBindGroups(rtBindGroups)
			, camBindGroups(camBindGroups)
			, bufferBindGroups(bufferBindGroups) {
			// require GUID
			lwb_primary_ray_rgen = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();
			lwb_primary_ray_rmiss = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();
			lwb_primary_ray_rchit = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();
			aaf_secondary_rchit = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();
			aaf_secondary_rmiss = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();
			lwb_shadow_ray_rchit = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();
			lwb_shadow_ray_rmiss = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();
			// load Shaders
			GFX::GFXManager::get()->registerShaderModuleResource(lwb_primary_ray_rgen, "../Engine/Binaries/Runtime/spirv/RayTracing/RayTrace/src/lightweight_benchmark/lwb_primary_ray_rgen.spv", { nullptr, RHI::ShaderStages::RAYGEN });
			GFX::GFXManager::get()->registerShaderModuleResource(lwb_primary_ray_rmiss, "../Engine/Binaries/Runtime/spirv/RayTracing/RayTrace/src/lightweight_benchmark/lwb_primary_ray_rmiss.spv", { nullptr, RHI::ShaderStages::MISS });
			GFX::GFXManager::get()->registerShaderModuleResource(lwb_primary_ray_rchit, "../Engine/Binaries/Runtime/spirv/RayTracing/RayTrace/src/lightweight_benchmark/lwb_primary_ray_rchit.spv", { nullptr, RHI::ShaderStages::CLOSEST_HIT });
			GFX::GFXManager::get()->registerShaderModuleResource(aaf_secondary_rchit, "../Engine/Binaries/Runtime/spirv/RayTracing/RayTrace/src/aaf_gi/aaf_gi_secondary_rchit.spv", { nullptr, RHI::ShaderStages::CLOSEST_HIT });
			GFX::GFXManager::get()->registerShaderModuleResource(aaf_secondary_rmiss, "../Engine/Binaries/Runtime/spirv/RayTracing/RayTrace/src/aaf_gi/aaf_gi_secondary_rmiss.spv", { nullptr, RHI::ShaderStages::MISS });
			GFX::GFXManager::get()->registerShaderModuleResource(lwb_shadow_ray_rchit, "../Engine/Binaries/Runtime/spirv/RayTracing/RayTrace/src/lightweight_benchmark/lwb_shadow_ray_rchit.spv", { nullptr, RHI::ShaderStages::CLOSEST_HIT });
			GFX::GFXManager::get()->registerShaderModuleResource(lwb_shadow_ray_rmiss, "../Engine/Binaries/Runtime/spirv/RayTracing/RayTrace/src/lightweight_benchmark/lwb_shadow_ray_rmiss.spv", { nullptr, RHI::ShaderStages::MISS });
			// Create rt pipeline
			for (int i = 0; i < 2; ++i) {
				raytracingPipeline[i] = rhiLayer->getDevice()->createRayTracingPipeline(RHI::RayTracingPipelineDescriptor{
					rtPipelineLayout, 3, RHI::SBTsDescriptor{
						RHI::SBTsDescriptor::RayGenerationSBT{{ Core::ResourceManager::get()->getResource<GFX::ShaderModule>(lwb_primary_ray_rgen)->shaderModule.get() }},
						RHI::SBTsDescriptor::MissSBT{{
							{Core::ResourceManager::get()->getResource<GFX::ShaderModule>(lwb_primary_ray_rmiss)->shaderModule.get()},
							{Core::ResourceManager::get()->getResource<GFX::ShaderModule>(lwb_shadow_ray_rmiss)->shaderModule.get()},
							{Core::ResourceManager::get()->getResource<GFX::ShaderModule>(aaf_secondary_rmiss)->shaderModule.get()} }},
						RHI::SBTsDescriptor::HitGroupSBT{{
							{{Core::ResourceManager::get()->getResource<GFX::ShaderModule>(lwb_primary_ray_rchit)->shaderModule.get()}, nullptr, nullptr},
							{{Core::ResourceManager::get()->getResource<GFX::ShaderModule>(lwb_shadow_ray_rchit)->shaderModule.get()}, nullptr, nullptr},
							{{Core::ResourceManager::get()->getResource<GFX::ShaderModule>(aaf_secondary_rchit)->shaderModule.get()}, nullptr, nullptr} }}
					} });
			}
		}

		~Benchmark_Tracing_Pass() {
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
		Core::GUID lwb_shadow_ray_rchit;
		Core::GUID lwb_shadow_ray_rmiss;

		std::unique_ptr<RHI::RayTracingPipeline> raytracingPipeline[2];
		std::unique_ptr<RHI::RayTracingPassEncoder> rtEncoder[2] = {};
		std::array<RHI::BindGroup*, 2> rtBindGroups = {};
		std::array<RHI::BindGroup*, 2> camBindGroups = {};
		std::array<RHI::BindGroup*, 2> bufferBindGroups = {};
	};

	struct PushConstantRay {
		Math::vec4 clearColor;
		Math::vec3 lightPosition;
		float lightIntensity;
		int   lightType;
	};

	export struct Benchmark_Pipeline {

		Benchmark_Pipeline(RHI::RHILayer* rhiLayer, GFX::ASGroup* asgroup, Core::GUID rtTarget,
			RHI::BindGroupLayout* cameraBindGroupLayout, std::array<RHI::BindGroup*, 2> const& camBindGroup)
			: rtTarget(rtTarget)
		{
			// allocate buffers
			zMinMaxBuffer = Core::ResourceManager::get()->requestRuntimeGUID<GFX::Buffer>();
			indirectBuffer = Core::ResourceManager::get()->requestRuntimeGUID<GFX::Buffer>();
			albedoBuffer = Core::ResourceManager::get()->requestRuntimeGUID<GFX::Buffer>();
			worldPosBuffer = Core::ResourceManager::get()->requestRuntimeGUID<GFX::Buffer>();
			worldNormalBuffer = Core::ResourceManager::get()->requestRuntimeGUID<GFX::Buffer>();
			projDistBuffer = Core::ResourceManager::get()->requestRuntimeGUID<GFX::Buffer>();
			useFilterBuffer = Core::ResourceManager::get()->requestRuntimeGUID<GFX::Buffer>();
			useFilterBlurredBuffer = Core::ResourceManager::get()->requestRuntimeGUID<GFX::Buffer>();
			visBlurredBuffer = Core::ResourceManager::get()->requestRuntimeGUID<GFX::Buffer>();
			slopeBlurredBuffer = Core::ResourceManager::get()->requestRuntimeGUID<GFX::Buffer>();

			GFX::GFXManager::get()->registerBufferResource(zMinMaxBuffer, RHI::BufferDescriptor{
				width * height * sizeof(float) * 2, // pixel size * sizeof(vec2)
				(uint32_t)RHI::BufferUsage::STORAGE });
			GFX::GFXManager::get()->registerBufferResource(indirectBuffer, RHI::BufferDescriptor{
				width * height * sizeof(float) * 4, // pixel size * sizeof(align(vec3))
				(uint32_t)RHI::BufferUsage::STORAGE });
			GFX::GFXManager::get()->registerBufferResource(albedoBuffer, RHI::BufferDescriptor{
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
			GFX::Buffer* pSlopeBuffer = Core::ResourceManager::get()->getResource<GFX::Buffer>(zMinMaxBuffer);
			GFX::Buffer* pVisBuffer = Core::ResourceManager::get()->getResource<GFX::Buffer>(indirectBuffer);
			GFX::Buffer* pProjDistBuffer = Core::ResourceManager::get()->getResource<GFX::Buffer>(albedoBuffer);
			GFX::Buffer* pWorldPosBuffer = Core::ResourceManager::get()->getResource<GFX::Buffer>(worldPosBuffer);
			GFX::Buffer* pWorldNormalBuffer = Core::ResourceManager::get()->getResource<GFX::Buffer>(worldNormalBuffer);
			GFX::Buffer* pVisBlurredBuffer = Core::ResourceManager::get()->getResource<GFX::Buffer>(visBlurredBuffer);
			GFX::Buffer* pBRDFBuffer = Core::ResourceManager::get()->getResource<GFX::Buffer>(projDistBuffer);
			GFX::Buffer* pUseFilterBuffer = Core::ResourceManager::get()->getResource<GFX::Buffer>(useFilterBuffer);
			GFX::Buffer* pUseFilterBlurredBuffer = Core::ResourceManager::get()->getResource<GFX::Buffer>(useFilterBlurredBuffer);
			GFX::Buffer* pSlopeBlurredBuffer = Core::ResourceManager::get()->getResource<GFX::Buffer>(slopeBlurredBuffer);
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

			//displayPipelineLayout = rhiLayer->getDevice()->createPipelineLayout(RHI::PipelineLayoutDescriptor{
			//	{}, { buffersBindGroupLayout.get() } });

			benchmark_tracing_pass = std::make_unique<Benchmark_Tracing_Pass>(rhiLayer, pipelineLayout.get(),
				std::array<RHI::BindGroup*, 2>{rtBindGroup[0].get(), rtBindGroup[1].get()},
				std::array<RHI::BindGroup*, 2>{camBindGroup[0], camBindGroup[1]},
				std::array<RHI::BindGroup*, 2>{bufferBindGroup[0].get(), bufferBindGroup[1].get()});
		}

		~Benchmark_Pipeline() {
			benchmark_tracing_pass = nullptr;
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
				(uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR,
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

			benchmark_tracing_pass->composeCommands(encoder, index);

			// ready output image for display
			encoder->pipelineBarrier(RHI::BarrierDescriptor{
				(uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR,
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

		Core::GUID rtTarget;

		size_t width = 800, height = 600;

		Core::GUID zMinMaxBuffer;
		Core::GUID indirectBuffer;
		Core::GUID albedoBuffer;
		Core::GUID worldPosBuffer;
		Core::GUID worldNormalBuffer;
		Core::GUID projDistBuffer;
		Core::GUID useFilterBuffer;
		Core::GUID useFilterBlurredBuffer;
		Core::GUID visBlurredBuffer;
		Core::GUID slopeBlurredBuffer;

		std::unique_ptr<RHI::BindGroupLayout> buffersBindGroupLayout = nullptr;
		std::unique_ptr<RHI::BindGroup> bufferBindGroup[2];
		std::unique_ptr<RHI::BindGroupLayout> rtBindGroupLayout = nullptr;
		std::unique_ptr<RHI::BindGroup> rtBindGroup[2];
		std::array<RHI::BindGroup*, 2> camBindGroup[2];

		std::unique_ptr<RHI::PipelineLayout> pipelineLayout;
		std::unique_ptr<RHI::PipelineLayout> displayPipelineLayout;

		std::unique_ptr<Benchmark_Tracing_Pass> benchmark_tracing_pass = nullptr;
	};
}