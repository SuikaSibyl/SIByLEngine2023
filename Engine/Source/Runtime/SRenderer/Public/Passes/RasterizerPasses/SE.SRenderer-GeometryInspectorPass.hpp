#pragma once

#include <Resource/SE.Core.Resource.hpp>
#include <SE.Editor.Core.hpp>
#include <SE.GFX.hpp>
#include <SE.Math.Geometric.hpp>
#include <SE.RDG.hpp>
#include <SE.RHI.hpp>
#include <SE.SRenderer.hpp>
#include <array>
#include <compare>
#include <filesystem>
#include <functional>
#include <memory>
#include <typeinfo>

#include "../../../../Application/Public/SE.Application.Config.h"

namespace SIByL {
SE_EXPORT struct GeometryInspectorPass : public RDG::RenderPass {
  GeometryInspectorPass();

  enum struct ShowEnum : uint32_t {
      BaseColor,
      Metalness,
      Anisotropy,
      Roughness,
      FlatNormal,
      GeometryNormal,
      ShadingNormal,
      NormalMap,
      Opacity,
      Emission,
      SpecularF0,
      VertexColor,
      Matcap,
      MatcapSurface,
      VertexNormal,
      UVChecker,
  };

  struct alignas(64) GeoVisUniform {
    ShowEnum showEnum = ShowEnum::BaseColor;
    int uv_checker_resource = -1;
    int matcap_resource = -1;
    int padding = 0;
    // wireframe settings
    Math::vec3 wireframe_color = Math::vec3(1);
    uint32_t use_wireframe = false;
    float wireframe_smoothing = 1.f;
    float wireframe_thickness = 1.f;
    uint32_t mouse_state = 0;
    int selected_object = -1;
  } geo_vis;
  GFX::StructuredUniformBufferView<GeoVisUniform> geo_vis_buffer;

  virtual auto reflect() noexcept -> RDG::PassReflection;
  virtual auto renderUI() noexcept -> void override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept -> void;
  virtual auto readback(RDG::RenderData const& renderData) noexcept -> void override;

 private:
  Core::GUID matcapGuid;
};
}  // namespace SIByL