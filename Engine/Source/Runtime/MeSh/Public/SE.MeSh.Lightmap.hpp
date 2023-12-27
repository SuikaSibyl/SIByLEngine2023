#pragma once
#include <SE.GFX-Main.hpp>

namespace SIByL::MeSh {
struct LightmapBuilder {
  static auto build(GFX::Scene& scene) noexcept -> void;
};
}