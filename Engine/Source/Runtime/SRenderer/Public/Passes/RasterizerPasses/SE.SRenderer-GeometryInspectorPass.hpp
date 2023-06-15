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

  struct alignas(64) GeoVisUniform {
    Math::vec3 wireframe_color = Math::vec3(1);
    uint32_t use_wireframe = false;
    float wireframe_smoothing = 1.f;
    float wireframe_thickness = 1.f;
  } geo_vis;
  GFX::StructuredUniformBufferView<GeoVisUniform> geo_vis_buffer;

  virtual auto reflect() noexcept -> RDG::PassReflection;
  virtual auto renderUI() noexcept -> void override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept -> void;

  Core::GUID vert, frag;
};
}  // namespace SIByL