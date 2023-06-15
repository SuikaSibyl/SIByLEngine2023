#include "SE.SRendererExt.GeomTab.hpp"

#include <vector>
#include <optional>
#include <algorithm>
#include <SE.Math.Geometric.hpp>

#include "../../../../../Shaders/SRenderer/include/common/compatible_cpp.h"
#include "../../../../../Shaders/SRenderer/include/common/common_space_filling_curve.h"

namespace SIByL {
struct Triangle {
  Math::bounds3 aabb;
  double area;
  uint64_t geomID;
  uint64_t budgetPercent;
  uint64_t budgetCells;
};

struct GeoTreeNode {
  Math::bounds3 aabb;
  Math::vec3 center;
  uint32_t cellLevel;
  Math::vec3 relative_center;
  uint32_t relative_morton;
  std::vector<GeoTreeNode> children;
  std::vector<Triangle> triangle;
};

uint64_t findPrevPowerOf2(uint64_t n) {
  n = n - 1;
  while (n & n - 1) {
    n = n & n - 1;
  }
  return n;
}

bool cmpBudget(Triangle const& a, Triangle const& b) {
  return (a.budgetCells < b.budgetCells);
}

bool cmpMorton(GeoTreeNode const& a, GeoTreeNode const& b) {
  return (a.relative_morton < b.relative_morton);
}

void GeometryTabulator::tabulate(
    uint64_t table_width,
    std::vector<float> const& position_buffer,
    std::vector<uint32_t> const& index_buffer,
    std::vector<SRenderer::GeometryDrawData> const& geometry_buffer) {
  Math::vec3 const* positions =
      reinterpret_cast<Math::vec3 const*>(position_buffer.data());

  return;
  double areaSum = 0;
  std::vector<Triangle> triangles;

  for (size_t g = 0; g < geometry_buffer.size(); ++g) {
    auto const& geom = geometry_buffer[g];

    for (size_t i = 0; i < geom.indexSize; i += 3) {
      size_t i0 = geom.indexOffset + i + 0;
      size_t i1 = geom.indexOffset + i + 1;
      size_t i2 = geom.indexOffset + i + 2;

      Math::vec3 v0 = positions[i0];
      Math::vec3 v1 = positions[i1];
      Math::vec3 v2 = positions[i2];

      Math::mat4 transform = Math::mat4(geom.geometryTransform);
      v0 = (transform * Math::vec4(v0, 1)).xyz();
      v1 = (transform * Math::vec4(v1, 1)).xyz();
      v2 = (transform * Math::vec4(v2, 1)).xyz();

      Triangle t;
      Math::vec3 e1 = v1 - v0;
      Math::vec3 e2 = v2 - v0;
      t.area = static_cast<double>(Math::length(Math::cross(e1, e2)));
      t.aabb = Math::bounds3{};
      t.aabb.pMin = Math::min(Math::min(v0, v1), v2);
      t.aabb.pMax = Math::max(Math::max(v0, v1), v2);
      t.geomID = g;

      areaSum += t.area;
      triangles.emplace_back(t);
    }
  }

  const uint64_t cell_count = table_width * table_width;
  for (auto& triangle : triangles) {
    double percentage_budget = triangle.area / areaSum;
    uint64_t budget_cell = cell_count * percentage_budget;
    budget_cell = findPrevPowerOf2(budget_cell);
    triangle.budgetPercent = percentage_budget;
    triangle.budgetCells = budget_cell;
  }
  //ZCurve3DToMortonCode
  std::sort(triangles.begin(), triangles.end(), cmpBudget);

  std::vector<GeoTreeNode> geometry_now;
  while (triangles.size() != 0) {
    const uint32_t cells = geometry_now.size() == 0
                               ? triangles.front().budgetCells
                               : geometry_now[0].cellLevel;

    for (auto iter = triangles.begin(); iter != triangles.end();) {
      if (iter->budgetCells != cells) break;
      else {
        GeoTreeNode node;
        node.cellLevel = cells;
        node.triangle.push_back(*iter);
        node.aabb = iter->aabb;
        node.center = 0.5 * (node.aabb.pMin + node.aabb.pMax);
        geometry_now.emplace_back(node);
        iter = triangles.erase(iter);
      }
    }
    Math::bounds3 bounds;
    for (auto const& geom : geometry_now) {
      bounds = Math::unionBounds(bounds, geom.aabb);
    }
    for (auto& geom : geometry_now) {
      geom.relative_center =
          (geom.center - bounds.pMin) / (bounds.pMax - bounds.pMin);
      geom.relative_morton = ZCurve3DToMortonCode(geom.relative_center);
    }
    std::sort(geometry_now.begin(), geometry_now.end(), cmpMorton);
    std::vector<GeoTreeNode> geometry_next;
    for (int i = 0; i < geometry_now.size(); i += 4) {
      int num = std::min(static_cast<int>(geometry_now.size()) - i, 4);
      GeoTreeNode parent;
      for (int j = i; j < i + num; ++j) {
        parent.aabb = Math::unionBounds(parent.aabb, geometry_now[j].aabb);
        parent.center = 0.5 * (parent.aabb.pMin + parent.aabb.pMax);
        parent.children.push_back(geometry_now[j]);
        parent.cellLevel = cells * 4;
      }
      geometry_next.push_back(parent);
    }
    geometry_now = geometry_next;
    float a = 1.f;
  }

  float a = 1.f;
}
}  // namespace SIByL