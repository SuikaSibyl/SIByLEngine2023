#pragma once
#include "../SE.SRenderer.hpp"

namespace SIByL {
struct GeometryTabulator {
  static void tabulate(
      uint64_t table_width,
      std::vector<float> const& position_buffer,
      std::vector<uint32_t> const& index_buffer,
      std::vector<SRenderer::GeometryDrawData> const& geometry_buffer);
};
}