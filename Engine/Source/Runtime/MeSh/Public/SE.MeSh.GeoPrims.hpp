#pragma once
#include <cstdint>
#include <SE.Math.Geometric.hpp>

namespace SIByL::MeSh {
using VertexID= uint32_t;
using FaceID= uint32_t;
constexpr inline uint32_t NilID = 0xffffffffU;

struct Vertex {
  union {
    Math::vec3 position;
    struct { VertexID parent, next, prev; } proxy;
  } as;

  float& operator()(int i) { return as.position[i]; }
  float  operator()(int i) const { return as.position[i]; }
  float& operator[](int i) { return as.position[i]; }
  float  operator[](int i) const { return as.position[i]; }
};

struct Edge {
  VertexID v1, v2;  // two vertices connected to
  Edge();
  Edge(VertexID a, VertexID b);
  auto opposite_vertex(VertexID v) noexcept -> VertexID;
};

struct TriangleFace {
  VertexID v[3];

};
}  // namespace SIByL::MeSh