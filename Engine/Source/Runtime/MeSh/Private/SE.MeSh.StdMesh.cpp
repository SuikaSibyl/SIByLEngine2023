#include "../Public/SE.MeSh.GeoPrims.hpp"
#include <Print/SE.Core.Log.hpp>

namespace SIByL::MeSh {
Edge::Edge() { v1 = v2 = NilID; }
Edge::Edge(VertexID a, VertexID b) : v1(a), v2(b) {}
auto Edge::opposite_vertex(VertexID v) noexcept -> VertexID {
  if (v == v1) return v2;
  else if (v == v2) return v1;
  else {
    Core::LogManager::Error(
        "MeSh::Edge::opposite_vertex::input v is not in edge.");
    return NilID;
  }
}


}