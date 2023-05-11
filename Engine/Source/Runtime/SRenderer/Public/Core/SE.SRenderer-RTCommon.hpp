#pragma once
#include <array>
#include <compare>
#include <filesystem>
#include <functional>
#include <memory>
#include <typeinfo>
#include <Resource/SE.Core.Resource.hpp>
#include <SE.RHI.hpp>
#include <SE.GFX.hpp>

namespace SIByL {
SE_EXPORT struct RTCommon {
  RTCommon() { singleton = this; }

  static auto get() noexcept -> RTCommon* { return singleton; }

  Core::GUID rmiss;
  Core::GUID shadow_ray_rchit;
  Core::GUID shadow_ray_rmiss;

  // Plugins: primitives
  // - sphere
  Core::GUID rchit_sphere;
  Core::GUID rint_sphere;
  Core::GUID rahit_sphere;
  Core::GUID sphere_sampling_rcall;
  Core::GUID sphere_sampling_pdf_rcall;
  Core::GUID rahit_sphere_shadow;
  // - trimesh
  Core::GUID rchit_trimesh;
  Core::GUID rahit_trimesh;
  Core::GUID trimesh_sampling_rcall;
  Core::GUID trimesh_sampling_pdf_rcall;
  Core::GUID rahit_trimesh_shadow;

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

  GFX::SBTsDescriptor sbtDesc;

  uint32_t accumIDX;

  auto getSBTDescriptor() noexcept -> GFX::SBTsDescriptor;

 private:
  static RTCommon* singleton;

  bool initialized = false;
  auto init() noexcept -> void;

  auto loadShaders() noexcept -> void;
};
}  // namespace SIByL