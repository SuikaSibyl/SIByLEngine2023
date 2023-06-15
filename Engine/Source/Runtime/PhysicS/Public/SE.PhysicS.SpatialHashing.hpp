#pragma once
#include <vector>
#include <cstdint>
#include <variant>
#include <functional>
#include <SE.Math.Geometric.hpp>
#ifdef _OPENMP
#include <omp.h>
#include <atomic>
#endif

namespace SIByL::PhysicS {
struct Grid2D_Handler_Bounded {
  uint32_t numX;
  uint32_t numY;
  float spacing, inv_spacing;
  auto hashing(Math::vec2 const& pos) const noexcept -> uint32_t;
  auto handle_count(std::vector<Math::vec2> const&,
                    std::vector<uint32_t>& count_buffer) const noexcept -> void;
  auto handle_visit_neighbors(std::vector<Math::vec2> const& pos,
                              std::function<void(uint32_t, uint32_t)>& test,
                              std::vector<uint32_t>& partial_sums,
                              std::vector<uint32_t>& dense_ids) const noexcept
      -> void;
};

struct Grid2D_Handler_Unbounded {
  float spacing, inv_spacing;
  auto hashing(Math::vec2 const& pos) const noexcept -> uint32_t;
  auto handle_count(std::vector<Math::vec2> const&,
                    std::vector<uint32_t>& count_buffer) const noexcept -> void;
  auto handle_visit_neighbors(std::vector<Math::vec2> const& pos,
                              std::function<void(uint32_t, uint32_t)>& test,
                              std::vector<uint32_t>& partial_sums,
                              std::vector<uint32_t>& dense_ids) const noexcept
      -> void;
};

using Grid2D_Handler = std::variant<Grid2D_Handler_Bounded, Grid2D_Handler_Unbounded>;


struct Grid2D {
  std::vector<uint32_t> count;
  std::vector<uint32_t> partial_sums;
  std::vector<uint32_t> dense_ids;
  Grid2D_Handler hash_handler;
  auto invalid() noexcept -> void;
  auto accum(std::vector<Math::vec2> const& pos) noexcept -> void;
  auto partial_sum() noexcept -> void;
  auto compact() noexcept -> void;
 private:
  std::vector<Math::vec2> const* pos_used;
};

struct SpatialHashing2D {
  Grid2D grid;

  auto init_bounded(Math::uvec2 num, float spacing, uint32_t pcount) noexcept
      -> void;
  auto prepare(std::vector<Math::vec2> const& pos) noexcept -> void;
  auto visit_neighbors(std::vector<Math::vec2> const& pos,
                       std::function<void(uint32_t, uint32_t)> handle) noexcept
      -> void;
};

struct Grid3D {
  uint32_t numX;
  uint32_t numY;
  float spacing;
};

struct SpatialHashing3D {
};
}