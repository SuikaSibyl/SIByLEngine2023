#include <se.rdg.hpp>
#include <se.gfx.hpp>

namespace se {
struct GeometryInspectorPass : public rdg::RenderPass {
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

  //struct alignas(64) GeoVisUniform {
  //  ShowEnum showEnum = ShowEnum::BaseColor;
  //  int uv_checker_resource = -1;
  //  int matcap_resource = -1;
  //  int padding = 0;
  //  // wireframe settings
  //  Math::vec3 wireframe_color = Math::vec3(1);
  //  uint32_t use_wireframe = false;
  //  float wireframe_smoothing = 1.f;
  //  float wireframe_thickness = 1.f;
  //  float padding_0 = 1.f;
  //  float padding_1 = 1.f;
  //} geo_vis;
  //GFX::StructuredUniformBufferView<GeoVisUniform> geo_vis_buffer;

  GeometryInspectorPass();
  virtual auto reflect() noexcept -> rdg::PassReflection;
  //virtual auto renderUI() noexcept -> void override;
  virtual auto execute(rdg::RenderContext* context,
    rdg::RenderData const& renderData) noexcept -> void;

  se::gfx::SceneHandle scene;
  se::gfx::TextureHandle color;
  se::gfx::TextureHandle depth;
};

struct GeometryInspectorGraph : public rdg::Graph {
  GeometryInspectorGraph();
};

struct GeometryInspectorPipeline : public rdg::SingleGraphPipeline {
  GeometryInspectorPipeline() { pGraph = &graph; }
  GeometryInspectorGraph graph;
};
}