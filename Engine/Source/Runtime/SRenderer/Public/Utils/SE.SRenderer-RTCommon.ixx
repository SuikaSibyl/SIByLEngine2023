module;
#include <array>
#include <memory>
#include <compare>
#include <typeinfo>
#include <filesystem>
#include <functional>
#include "../../../Application/Public/SE.Application.Config.h"
export module SE.SRenderer:RTCommon;
import SE.Core.Resource;
import SE.RHI;
import SE.GFX.Core;
import SE.GFX.RDG;

namespace SIByL
{
	export struct RTCommon {

		Core::GUID rmiss;
		Core::GUID shadow_ray_rchit;
		Core::GUID shadow_ray_rmiss;

		// Plugins: primitives
		// - sphere
		Core::GUID rchit_sphere;
		Core::GUID rint_sphere;
		Core::GUID sphere_sampling_rcall;
		Core::GUID sphere_sampling_pdf_rcall;
		// - trimesh
		Core::GUID rchit_trimesh;
		Core::GUID trimesh_sampling_rcall;
		Core::GUID trimesh_sampling_pdf_rcall;

		// Plugins: bsdfs
		// - lambertian
		Core::GUID lambertian_eval;
		Core::GUID lambertian_sample;
		Core::GUID lambertian_pdf;
		// - roughdielectric
		Core::GUID roughdielectric_eval;
		Core::GUID roughdielectric_sample;
		Core::GUID roughdielectric_pdf;
		// - principled
		Core::GUID principled_eval;
		Core::GUID principled_sample;
		Core::GUID principled_pdf;

		RHI::RayTracingPipelineDescriptor rt_pipeline;

		auto getPipelineDescriptor() noexcept -> RHI::RayTracingPipelineDescriptor {
			if (!initialized)
				init();
			return rt_pipeline;
		}

	private:
		bool initialized = false;
		auto init() noexcept -> void {
			loadShaders();
			rt_pipeline = RHI::RayTracingPipelineDescriptor{
							nullptr, 3, RHI::SBTsDescriptor{
							RHI::SBTsDescriptor::RayGenerationSBT{{ nullptr }},
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
								{Core::ResourceManager::get()->getResource<GFX::ShaderModule>(principled_pdf)->shaderModule.get()}, 
							}},
						} };
		}

		auto loadShaders() noexcept -> void {

			rmiss = GFX::GFXManager::get()->registerShaderModuleResource("../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/path_tracer/stracer_rmiss.spv", { nullptr, RHI::ShaderStages::MISS });
			shadow_ray_rchit = GFX::GFXManager::get()->registerShaderModuleResource(
				"../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/path_tracer/spt_shadow_ray_rchit.spv",
				{ nullptr, RHI::ShaderStages::CLOSEST_HIT });
			shadow_ray_rmiss = GFX::GFXManager::get()->registerShaderModuleResource(
				"../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/path_tracer/spt_shadow_ray_rmiss.spv",
				{ nullptr, RHI::ShaderStages::MISS });


			// Plugins: primitives
			{	// - sphere
				rchit_sphere = GFX::GFXManager::get()->registerShaderModuleResource(
					"../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/path_tracer/spt_primary_ray_sphere_rchit.spv",
					{ nullptr, RHI::ShaderStages::CLOSEST_HIT });
				rint_sphere = GFX::GFXManager::get()->registerShaderModuleResource(
					"../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/plugins/primitive/sphere_hint_rint.spv",
					{ nullptr, RHI::ShaderStages::INTERSECTION });
				sphere_sampling_rcall = GFX::GFXManager::get()->registerShaderModuleResource(
					"../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/plugins/primitive/sphere_sample_rcall.spv",
					{ nullptr, RHI::ShaderStages::CALLABLE });
				sphere_sampling_pdf_rcall = GFX::GFXManager::get()->registerShaderModuleResource(
					"../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/plugins/primitive/sphere_sample_pdf_rcall.spv",
					{ nullptr, RHI::ShaderStages::CALLABLE });
			}
			{	// - trimesh
				rchit_trimesh = GFX::GFXManager::get()->registerShaderModuleResource(
					"../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/path_tracer/spt_primary_ray_trimesh_rchit.spv",
					{ nullptr, RHI::ShaderStages::CLOSEST_HIT });
				trimesh_sampling_rcall = GFX::GFXManager::get()->registerShaderModuleResource(
					"../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/plugins/primitive/trimesh_sample_rcall.spv",
					{ nullptr, RHI::ShaderStages::CALLABLE });
				trimesh_sampling_pdf_rcall = GFX::GFXManager::get()->registerShaderModuleResource(
					"../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/plugins/primitive/trimesh_sample_pdf_rcall.spv",
					{ nullptr, RHI::ShaderStages::CALLABLE });
			}

			// Plugins: bsdfs
			{	// - lambertian
				lambertian_eval = GFX::GFXManager::get()->registerShaderModuleResource(
					"../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/plugins/material/lambertian_eval_rcall.spv",
					{ nullptr, RHI::ShaderStages::CALLABLE });
				lambertian_sample = GFX::GFXManager::get()->registerShaderModuleResource(
					"../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/plugins/material/lambertian_sample_rcall.spv",
					{ nullptr, RHI::ShaderStages::CALLABLE });
				lambertian_pdf = GFX::GFXManager::get()->registerShaderModuleResource(
					"../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/plugins/material/lambertian_pdf_rcall.spv",
					{ nullptr, RHI::ShaderStages::CALLABLE });
			}
			{	// - roughdielectric
				roughdielectric_eval = GFX::GFXManager::get()->registerShaderModuleResource(
					"../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/plugins/material/roughdielectric_eval_rcall.spv",
					{ nullptr, RHI::ShaderStages::CALLABLE });
				roughdielectric_sample = GFX::GFXManager::get()->registerShaderModuleResource(
					"../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/plugins/material/roughdielectric_sample_rcall.spv",
					{ nullptr, RHI::ShaderStages::CALLABLE });
				roughdielectric_pdf = GFX::GFXManager::get()->registerShaderModuleResource(
					"../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/plugins/material/roughdielectric_pdf_rcall.spv",
					{ nullptr, RHI::ShaderStages::CALLABLE });
			}
			{	// - principled
				principled_eval = GFX::GFXManager::get()->registerShaderModuleResource(
					"../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/plugins/material/principled_eval_rcall.spv",
					{ nullptr, RHI::ShaderStages::CALLABLE });
				principled_sample = GFX::GFXManager::get()->registerShaderModuleResource(
					"../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/plugins/material/principled_sample_rcall.spv",
					{ nullptr, RHI::ShaderStages::CALLABLE });
				principled_pdf = GFX::GFXManager::get()->registerShaderModuleResource(
					"../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/plugins/material/principled_pdf_rcall.spv",
					{ nullptr, RHI::ShaderStages::CALLABLE });
			}
		}
	};
}