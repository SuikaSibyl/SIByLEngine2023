#include <Core/SE.SRenderer-RTCommon.hpp>

namespace SIByL {
auto RTCommon::getSBTDescriptor() noexcept -> GFX::SBTsDescriptor {
  if (!initialized) init();
  return sbtDesc;
}

auto RTCommon::init() noexcept -> void {
  auto [trimesh_ray_rchit, trimesh_anyhit, trimesh_shadow_anyhit,
        sphere_intersection, sphere_ray_rchit, sphere_anyhit,
        sphere_shadow_anyhit, common_ray_rmiss, shadow_ray_rchit, shadow_ray_rmiss, 
        eval_lambertian, sample_lambertian, pdf_lambertian, eval_diff_lambertian,
        eval_roughplastic, sample_roughplastic, pdf_roughplastic, eval_diff_roughplastic,
        eval_roughdielectric, sample_roughdielectric, pdf_roughdielectric, eval_diff_roughdielectric
  ] =
      GFX::ShaderLoader_SLANG::load(
          "../Engine/Shaders/SRenderer/raytracer/"
          "spt.slang",
          std::array<std::pair<std::string, RHI::ShaderStages>, 22>{
              std::make_pair("TrimeshClosestHit",
                             RHI::ShaderStages::CLOSEST_HIT),
              std::make_pair("TrimeshAnyHit", RHI::ShaderStages::ANY_HIT),
              std::make_pair("TrimeshShadowRayAnyHit",
                             RHI::ShaderStages::ANY_HIT),
              std::make_pair("SphereIntersection",
                             RHI::ShaderStages::INTERSECTION),
              std::make_pair("SphereClosestHit",
                             RHI::ShaderStages::CLOSEST_HIT),
              std::make_pair("SphereAnyHit", RHI::ShaderStages::ANY_HIT),
              std::make_pair("SphereShadowRayAnyHit",
                             RHI::ShaderStages::ANY_HIT),
              std::make_pair("CommonRayMiss", RHI::ShaderStages::MISS),
              std::make_pair("ShadowRayClosestHit",
                             RHI::ShaderStages::CLOSEST_HIT),
              std::make_pair("ShadowRayMiss", RHI::ShaderStages::MISS),
              // Material shaders
              std::make_pair("EvalLambertian", RHI::ShaderStages::CALLABLE),
              std::make_pair("SampleLambertian", RHI::ShaderStages::CALLABLE),
              std::make_pair("PdfLambertian", RHI::ShaderStages::CALLABLE),
              std::make_pair("EvalDiffLambertian", RHI::ShaderStages::CALLABLE),
              // Material shaders - rough plastic
              std::make_pair("EvalRoughPlastic", RHI::ShaderStages::CALLABLE),
              std::make_pair("SampleRoughPlastic", RHI::ShaderStages::CALLABLE),
              std::make_pair("PdfRoughPlastic", RHI::ShaderStages::CALLABLE),
              std::make_pair("EvalDiffRoughPlastic", RHI::ShaderStages::CALLABLE),
              // Material shaders - rough dielectric
              std::make_pair("EvalRoughDielectric", RHI::ShaderStages::CALLABLE),
              std::make_pair("SampleRoughDielectric", RHI::ShaderStages::CALLABLE),
              std::make_pair("PdfRoughDielectric", RHI::ShaderStages::CALLABLE),
              std::make_pair("EvalDiffRoughDielectric", RHI::ShaderStages::CALLABLE),
          });

   sbtDesc = GFX::SBTsDescriptor{
      GFX::SBTsDescriptor::RayGenerationSBT{{nullptr}},
      GFX::SBTsDescriptor::MissSBT{{
          {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(
              common_ray_rmiss)},
          {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(
              shadow_ray_rmiss)},
      }},
      GFX::SBTsDescriptor::HitGroupSBT{{
          // 0: triangle mesh opaque
          {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(
              trimesh_ray_rchit)},
          // 1: triangle mesh with alpha cut
          {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(
               trimesh_ray_rchit),
           Core::ResourceManager::get()->getResource<GFX::ShaderModule>(
               trimesh_anyhit)},
          // 2: sphere primitive opaque
          {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(
                sphere_ray_rchit), nullptr,
           Core::ResourceManager::get()->getResource<GFX::ShaderModule>(
                sphere_intersection),},
          // 3: sphere primitive with alpha cut
          {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(
                sphere_ray_rchit),
           Core::ResourceManager::get()->getResource<GFX::ShaderModule>(
                sphere_anyhit),
           Core::ResourceManager::get()->getResource<GFX::ShaderModule>(
                sphere_intersection),},
          // 4: triangle mesh opaque - shadowray
          {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(
              shadow_ray_rchit)},
          // 5: triangle mesh with alpha cut - shadowray
          {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(
               shadow_ray_rchit),
           Core::ResourceManager::get()->getResource<GFX::ShaderModule>(
               trimesh_shadow_anyhit)},
          // 6: sphere primitive opaque - shadowray
          {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(
                shadow_ray_rchit), nullptr,
           Core::ResourceManager::get()->getResource<GFX::ShaderModule>(
                sphere_intersection),},
          // 7: sphere primitive with alpha cut - shadowray
          {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(
                shadow_ray_rchit),
           Core::ResourceManager::get()->getResource<GFX::ShaderModule>(
                sphere_shadow_anyhit),
           Core::ResourceManager::get()->getResource<GFX::ShaderModule>(
                sphere_intersection),},
      }},
      GFX::SBTsDescriptor::CallableSBT{{
          // lambertian
          {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(eval_lambertian)},
          {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(sample_lambertian)},
          {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(pdf_lambertian)},
          // roughplastic
          {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(eval_roughplastic)},
          {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(sample_roughplastic)},
          {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(pdf_roughplastic)},
          // roughdielectric
          {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(eval_roughdielectric)},
          {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(sample_roughdielectric)},
          {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(pdf_roughdielectric)},
          //{Core::ResourceManager::get()
          //     ->getResource<GFX::ShaderModule>(
          //         sphere_sampling_rcall)},
          //{Core::ResourceManager::get()
          //     ->getResource<GFX::ShaderModule>(
          //         sphere_sampling_pdf_rcall)},
          //{Core::ResourceManager::get()
          //     ->getResource<GFX::ShaderModule>(
          //         trimesh_sampling_rcall)},
          //{Core::ResourceManager::get()
          //     ->getResource<GFX::ShaderModule>(
          //         trimesh_sampling_pdf_rcall)},
      }},
  };

  diffCallables = {
    {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(eval_diff_lambertian)},
    {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(eval_diff_roughplastic)},
    {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(eval_diff_roughdielectric)},
  };
  //loadShaders();
  //sbtDesc = GFX::SBTsDescriptor{
  //    GFX::SBTsDescriptor::RayGenerationSBT{{nullptr}},
  //    GFX::SBTsDescriptor::MissSBT{{
  //        {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(rmiss)},
  //        {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(
  //            shadow_ray_rmiss)},
  //    }},
  //    GFX::SBTsDescriptor::HitGroupSBT{{
  //        {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(
  //            rchit_trimesh)},
  //        {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(
  //             rchit_trimesh),
  //         Core::ResourceManager::get()->getResource<GFX::ShaderModule>(
  //             rahit_trimesh)},
  //        {
  //            Core::ResourceManager::get()->getResource<GFX::ShaderModule>(
  //                rchit_sphere),
  //            nullptr,
  //            Core::ResourceManager::get()->getResource<GFX::ShaderModule>(
  //                rint_sphere),
  //        },
  //        {
  //            Core::ResourceManager::get()->getResource<GFX::ShaderModule>(
  //                rchit_sphere),
  //            Core::ResourceManager::get()->getResource<GFX::ShaderModule>(
  //                rahit_sphere),
  //            Core::ResourceManager::get()->getResource<GFX::ShaderModule>(
  //                rint_sphere),
  //        },
  //        {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(
  //            shadow_ray_rchit)},
  //        {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(
  //             shadow_ray_rchit),
  //         Core::ResourceManager::get()->getResource<GFX::ShaderModule>(
  //             rahit_trimesh_shadow)},
  //        {
  //            Core::ResourceManager::get()->getResource<GFX::ShaderModule>(
  //                shadow_ray_rchit),
  //            Core::ResourceManager::get()->getResource<GFX::ShaderModule>(
  //                rahit_sphere_shadow),
  //            Core::ResourceManager::get()->getResource<GFX::ShaderModule>(
  //                rint_sphere),
  //        },
  //    }},
  //    GFX::SBTsDescriptor::CallableSBT{{
  //        {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(
  //            sphere_sampling_rcall)},
  //        {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(
  //            sphere_sampling_pdf_rcall)},
  //        {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(
  //            trimesh_sampling_rcall)},
  //        {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(
  //            trimesh_sampling_pdf_rcall)},
  //        // lambertian
  //        {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(
  //            lambertian_eval)},
  //        {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(
  //            lambertian_sample)},
  //        {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(
  //            lambertian_pdf)},
  //        // principled
  //        {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(
  //            principled_eval)},
  //        {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(
  //            principled_sample)},
  //        {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(
  //            principled_pdf)},
  //        // roughdielectric
  //        {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(
  //            roughdielectric_eval)},
  //        {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(
  //            roughdielectric_sample)},
  //        {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(
  //            roughdielectric_pdf)},
  //    }},
  //};
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