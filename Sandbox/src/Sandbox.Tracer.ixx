module;
#include <array>
#include <filesystem>
export module Sandbox.Tracer;
import Core.Resource.RuntimeManage;
import RHI;
import RHI.RHILayer;
import GFX.Resource;
import GFX.GFXManager;

using namespace SIByL;

namespace Sandbox
{
	export struct DirectTracer {
		DirectTracer(RHI::RHILayer* rhiLayer, std::array<RHI::PipelineLayout*, 2> const& layout) {
			// require GUID
			rgen = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();
			diffuse_rchit = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();
			shadow_rmiss = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();
			sky_rmiss = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();
			// load Shaders
			GFX::GFXManager::get()->registerShaderModuleResource(rgen, "../Engine/Binaries/Runtime/spirv/RayTracing/RayTrace/src/direct_light_tracer/direct_light_integrator_rgen.spv", { nullptr, RHI::ShaderStages::RAYGEN });
			GFX::GFXManager::get()->registerShaderModuleResource(diffuse_rchit, "../Engine/Binaries/Runtime/spirv/RayTracing/RayTrace/src/direct_light_tracer/direct_light_diffuse_mat_rchit.spv", { nullptr, RHI::ShaderStages::CLOSEST_HIT });
			GFX::GFXManager::get()->registerShaderModuleResource(shadow_rmiss, "../Engine/Binaries/Runtime/spirv/RayTracing/RayTrace/src/direct_light_tracer/direct_light_shadow_rmiss.spv", { nullptr, RHI::ShaderStages::MISS });
			GFX::GFXManager::get()->registerShaderModuleResource(sky_rmiss, "../Engine/Binaries/Runtime/spirv/RayTracing/RayTrace/src/direct_light_tracer/direct_light_simple_sky_rmiss.spv", { nullptr, RHI::ShaderStages::MISS });
		
			for (int i = 0; i < 2; ++i) {
				raytracingPipeline[i] = rhiLayer->getDevice()->createRayTracingPipeline(RHI::RayTracingPipelineDescriptor{ 
					layout[i], 2, RHI::SBTsDescriptor{
						RHI::SBTsDescriptor::RayGenerationSBT{{ Core::ResourceManager::get()->getResource<GFX::ShaderModule>(rgen)->shaderModule.get() }},
						RHI::SBTsDescriptor::MissSBT{{
							{Core::ResourceManager::get()->getResource<GFX::ShaderModule>(sky_rmiss)->shaderModule.get()},
							{Core::ResourceManager::get()->getResource<GFX::ShaderModule>(shadow_rmiss)->shaderModule.get()} }},
						RHI::SBTsDescriptor::HitGroupSBT{{
								{{Core::ResourceManager::get()->getResource<GFX::ShaderModule>(diffuse_rchit)->shaderModule.get()}, nullptr, nullptr},
							}}
					} });
			}
		}

		~DirectTracer() {
			raytracingPipeline[0] = nullptr;
			raytracingPipeline[1] = nullptr;
		}

		Core::GUID rgen;
		Core::GUID diffuse_rchit;
		Core::GUID shadow_rmiss;
		Core::GUID sky_rmiss;

		std::unique_ptr<RHI::RayTracingPipeline> raytracingPipeline[2];
	};

	export struct AAFTracer {

	};
}