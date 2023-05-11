#pragma once
#include <vector>
#include <SE.Math.Geometric.hpp>
#include <SE.Editor.RDG.hpp>

namespace SIByL::Editor {
struct DebugDrawData {
  uint32_t width, height;
};

struct Data_Line2D {
  struct Point {
    Math::vec2 begin;
    Math::vec2 end;
  };
  struct LineProperty {
    Math::vec3 color;
    float width;
  };
  std::vector<Point> vertices_cpu;
  std::vector<LineProperty> lineProps_cpu;
};

struct Data_Line3D {
  struct Point {
    Math::vec3 begin;
    Math::vec3 end;
  };
  struct LineProperty {
    Math::vec3 color;
    float width;
  };
  std::vector<Point> vertices_cpu;
  std::vector<LineProperty> lineProps_cpu;
};

SE_EXPORT struct DrawLine2DPass : public RDG::RenderPass {
  DrawLine2DPass(Data_Line2D* data);
  virtual auto reflect() noexcept -> RDG::PassReflection;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept -> void;

 private:
  auto updateBuffer(size_t idx) noexcept -> void;
  auto discardGarbage() noexcept -> void;
  auto scanGarbageStation() noexcept -> void;

  Data_Line2D* line2DData;
  Core::GUID vert, frag;

  struct GarbageStation {
    GFX::StructuredArrayMultiStorageBufferView<Data_Line2D::Point> points;
    GFX::StructuredArrayMultiStorageBufferView<Data_Line2D::LineProperty> lines;
    uint32_t count_down = -1;
  } garbageStation;
  GFX::StructuredArrayMultiStorageBufferView<Data_Line2D::Point> curr_points;
  GFX::StructuredArrayMultiStorageBufferView<Data_Line2D::LineProperty>
      curr_lines;
};

SE_EXPORT struct DrawLine3DPass : public RDG::RenderPass {
  DrawLine3DPass(Data_Line3D* data);
  virtual auto reflect() noexcept -> RDG::PassReflection;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept -> void;

 private:
  auto updateBuffer(size_t idx) noexcept -> void;
  auto discardGarbage() noexcept -> void;
  auto scanGarbageStation() noexcept -> void;

  Data_Line3D* line3DData;
  Core::GUID vert, frag;

  struct GarbageStation {
    GFX::StructuredArrayMultiStorageBufferView<Data_Line3D::Point> points;
    GFX::StructuredArrayMultiStorageBufferView<Data_Line3D::LineProperty> lines;
    uint32_t count_down = -1;
  } garbageStation;
  GFX::StructuredArrayMultiStorageBufferView<Data_Line3D::Point> curr_points;
  GFX::StructuredArrayMultiStorageBufferView<Data_Line3D::LineProperty>
      curr_lines;
};

SE_EXPORT struct DebugDrawDummy : public RDG::DummyPass {
  DebugDrawDummy() { RDG::Pass::init(); }
  virtual auto reflect() noexcept -> RDG::PassReflection override;
};

struct DebugDraw;

SE_EXPORT struct DebugDrawGraph : public RDG::Graph {
  DebugDrawGraph();
  auto setSource(GFX::Texture* ref, GFX::Texture* depth) noexcept -> void;
};

SE_EXPORT struct DebugDrawPipeline : public RDG::SingleGraphPipeline {
  DebugDrawPipeline() { pGraph = &graph; }
  DebugDrawGraph graph;
};

SE_EXPORT struct DebugDraw {
  static auto Init(GFX::Texture* ref, GFX::Texture* depth) noexcept -> void;

  static auto Draw(RHI::CommandEncoder* encoder) noexcept -> void;

  static auto Clear() noexcept -> void;

  static auto DrawLine2D(Math::vec2 const& a, Math::vec2 const& b,
                         Math::vec3 color = {1., 0., 0.},
                         float width = 1.f) noexcept -> void;
  static auto DrawLine3D(Math::vec3 const& a, Math::vec3 const& b,
                         Math::vec3 color, float width) noexcept -> void;
  static auto DrawAABB(Math::bounds3 const& aann, Math::vec3 color,
                       float width) noexcept -> void;

  static auto Destroy() noexcept -> void;

  Data_Line2D drawLine2D;
  Data_Line3D drawLine3D;
  std::unique_ptr<DebugDrawPipeline> pipeline = nullptr;

  static auto get() -> DebugDraw* {
    if (singleton == nullptr) singleton = new DebugDraw();
    return singleton;
  }
  static DebugDraw* singleton;
};
}  // namespace SIByL::Editor