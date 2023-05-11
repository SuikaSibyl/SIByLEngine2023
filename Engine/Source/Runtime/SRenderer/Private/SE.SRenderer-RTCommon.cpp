#include <Core/SE.SRenderer-RTCommon.hpp>

namespace SIByL {
auto RTCommon::getSBTDescriptor() noexcept -> GFX::SBTsDescriptor {
  if (!initialized) init();
  return sbtDesc;
}

auto RTCommon::init() noexcept -> void {
  loadShaders();
  sbtDesc = GFX::SBTsDescriptor{
      GFX::SBTsDescriptor::RayGenerationSBT{{nullptr}},
      GFX::SBTsDescriptor::MissSBT{{
          {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(rmiss)},
          {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(
              shadow_ray_rmiss)},
      }},
      GFX::SBTsDescriptor::HitGroupSBT{{
          {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(
              rchit_trimesh)},
          {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(
               rchit_trimesh),
           Core::ResourceManager::get()->getResource<GFX::ShaderModule>(
               rahit_trimesh)},
          {
              Core::ResourceManager::get()->getResource<GFX::ShaderModule>(
                  rchit_sphere),
              nullptr,
              Core::ResourceManager::get()->getResource<GFX::ShaderModule>(
                  rint_sphere),
          },
          {
              Core::ResourceManager::get()->getResource<GFX::ShaderModule>(
                  rchit_sphere),
              Core::ResourceManager::get()->getResource<GFX::ShaderModule>(
                  rahit_sphere),
              Core::ResourceManager::get()->getResource<GFX::ShaderModule>(
                  rint_sphere),
          },
          {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(
              shadow_ray_rchit)},
          {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(
               shadow_ray_rchit),
           Core::ResourceManager::get()->getResource<GFX::ShaderModule>(
               rahit_trimesh_shadow)},
          {
              Core::ResourceManager::get()->getResource<GFX::ShaderModule>(
                  shadow_ray_rchit),
              Core::ResourceManager::get()->getResource<GFX::ShaderModule>(
                  rahit_sphere_shadow),
              Core::ResourceManager::get()->getResource<GFX::ShaderModule>(
                  rint_sphere),
          },
      }},
      GFX::SBTsDescriptor::CallableSBT{{
          {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(
              sphere_sampling_rcall)},
          {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(
              sphere_sampling_pdf_rcall)},
          {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(
              trimesh_sampling_rcall)},
          {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(
              trimesh_sampling_pdf_rcall)},
          // lambertian
          {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(
              lambertian_eval)},
          {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(
              lambertian_sample)},
          {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(
              lambertian_pdf)},
          // principled
          {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(
              principled_eval)},
          {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(
              principled_sample)},
          {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(
              principled_pdf)},
          // roughdielectric
          {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(
              roughdielectric_eval)},
          {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(
              roughdielectric_sample)},
          {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(
              roughdielectric_pdf)},
      }},
  };
}

auto RTCommon::loadShaders() noexcept -> void {
  rmiss = GFX::GFXManager::get()->registerShaderModuleResource(
      "../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/path_tracer/"
      "spt_primary_ray_rmiss.spv",
      {nullptr, RHI::ShaderStages::MISS});
  shadow_ray_rchit = GFX::GFXManager::get()->registerShaderModuleResource(
      "../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/path_tracer/"
      "spt_shadow_ray_rchit.spv",
      {nullptr, RHI::ShaderStages::CLOSEST_HIT});
  shadow_ray_rmiss = GFX::GFXManager::get()->registerShaderModuleResource(
      "../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/path_tracer/"
      "spt_shadow_ray_rmiss.spv",
      {nullptr, RHI::ShaderStages::MISS});

  // Plugins: primitives
  {  // - sphere
    rchit_sphere = GFX::GFXManager::get()->registerShaderModuleResource(
        "../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/path_tracer/"
        "spt_primary_ray_sphere_rchit.spv",
        {nullptr, RHI::ShaderStages::CLOSEST_HIT});
    rint_sphere = GFX::GFXManager::get()->registerShaderModuleResource(
        "../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/plugins/"
        "primitive/sphere_hint_rint.spv",
        {nullptr, RHI::ShaderStages::INTERSECTION});
    rahit_sphere = GFX::GFXManager::get()->registerShaderModuleResource(
        "../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/path_tracer/"
        "spt_primary_ray_sphere_rahit.spv",
        {nullptr, RHI::ShaderStages::ANY_HIT});
    sphere_sampling_rcall =
        GFX::GFXManager::get()->registerShaderModuleResource(
            "../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/plugins/"
            "primitive/sphere_sample_rcall.spv",
            {nullptr, RHI::ShaderStages::CALLABLE});
    sphere_sampling_pdf_rcall =
        GFX::GFXManager::get()->registerShaderModuleResource(
            "../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/plugins/"
            "primitive/sphere_sample_pdf_rcall.spv",
            {nullptr, RHI::ShaderStages::CALLABLE});
    rahit_sphere_shadow = GFX::GFXManager::get()->registerShaderModuleResource(
        "../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/"
        "path_tracer/spt_shadow_ray_sphere_rahit.spv",
        {nullptr, RHI::ShaderStages::ANY_HIT});
  }
  {  // - trimesh
    rchit_trimesh = GFX::GFXManager::get()->registerShaderModuleResource(
        "../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/path_tracer/"
        "spt_primary_ray_trimesh_rchit.spv",
        {nullptr, RHI::ShaderStages::CLOSEST_HIT});
    rahit_trimesh = GFX::GFXManager::get()->registerShaderModuleResource(
        "../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/path_tracer/"
        "spt_primary_ray_trimesh_rahit.spv",
        {nullptr, RHI::ShaderStages::ANY_HIT});
    trimesh_sampling_rcall =
        GFX::GFXManager::get()->registerShaderModuleResource(
            "../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/plugins/"
            "primitive/trimesh_sample_rcall.spv",
            {nullptr, RHI::ShaderStages::CALLABLE});
    trimesh_sampling_pdf_rcall =
        GFX::GFXManager::get()->registerShaderModuleResource(
            "../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/plugins/"
            "primitive/trimesh_sample_pdf_rcall.spv",
            {nullptr, RHI::ShaderStages::CALLABLE});
    rahit_trimesh_shadow = GFX::GFXManager::get()->registerShaderModuleResource(
        "../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/"
        "path_tracer/spt_shadow_ray_trimesh_rahit.spv",
        {nullptr, RHI::ShaderStages::ANY_HIT});
  }

  // Plugins: bsdfs
  {  // - lambertian
    lambertian_eval = GFX::GFXManager::get()->registerShaderModuleResource(
        "../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/plugins/"
        "material/lambertian_eval_rcall.spv",
        {nullptr, RHI::ShaderStages::CALLABLE});
    lambertian_sample = GFX::GFXManager::get()->registerShaderModuleResource(
        "../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/plugins/"
        "material/lambertian_sample_rcall.spv",
        {nullptr, RHI::ShaderStages::CALLABLE});
    lambertian_pdf = GFX::GFXManager::get()->registerShaderModuleResource(
        "../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/plugins/"
        "material/lambertian_pdf_rcall.spv",
        {nullptr, RHI::ShaderStages::CALLABLE});
  }
  {  // - principled
    principled_eval = GFX::GFXManager::get()->registerShaderModuleResource(
        "../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/plugins/"
        "material/principled_eval_rcall.spv",
        {nullptr, RHI::ShaderStages::CALLABLE});
    principled_sample = GFX::GFXManager::get()->registerShaderModuleResource(
        "../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/plugins/"
        "material/principled_sample_rcall.spv",
        {nullptr, RHI::ShaderStages::CALLABLE});
    principled_pdf = GFX::GFXManager::get()->registerShaderModuleResource(
        "../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/plugins/"
        "material/principled_pdf_rcall.spv",
        {nullptr, RHI::ShaderStages::CALLABLE});
  }
  {  // - roughdielectric
    roughdielectric_eval = GFX::GFXManager::get()->registerShaderModuleResource(
        "../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/plugins/"
        "material/roughdielectric_eval_rcall.spv",
        {nullptr, RHI::ShaderStages::CALLABLE});
    roughdielectric_sample =
        GFX::GFXManager::get()->registerShaderModuleResource(
            "../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/plugins/"
            "material/roughdielectric_sample_rcall.spv",
            {nullptr, RHI::ShaderStages::CALLABLE});
    roughdielectric_pdf = GFX::GFXManager::get()->registerShaderModuleResource(
        "../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/plugins/"
        "material/roughdielectric_pdf_rcall.spv",
        {nullptr, RHI::ShaderStages::CALLABLE});
  }
}

RTCommon* RTCommon::singleton = nullptr;
}  // namespace SIByL