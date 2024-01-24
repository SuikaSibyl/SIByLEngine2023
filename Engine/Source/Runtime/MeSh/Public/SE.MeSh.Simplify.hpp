#pragma once
#include <SE.GFX-Main.hpp>

namespace SIByL::MeSh {
struct MeshSimplifier {
  static auto quadric_simplify(GFX::Mesh& mesh, int target_count,
    float target_ratio = 1.f, double agressiveness = 7) noexcept -> GFX::Mesh;
};
}  // namespace SIByL::MeSh