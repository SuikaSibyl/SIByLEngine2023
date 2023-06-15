#include "../Public/SE.PhysicS.SpatialHashing.hpp"
#include <algorithm>

namespace SIByL::PhysicS {
auto Grid2D_Handler_Bounded::hashing(Math::vec2 const& pos) const noexcept
    -> uint32_t {
  int const xi =
      std::clamp(int(std::floor(pos.x * inv_spacing)), 0, int(numX) - 1);
  int const yi =
      std::clamp(int(std::floor(pos.y * inv_spacing)), 0, int(numY) - 1);
  return xi * numY + yi;
}
auto Grid2D_Handler_Bounded::handle_count(
    std::vector<Math::vec2> const& pos_vec,
    std::vector<uint32_t>& count_buffer) const noexcept -> void {
//#ifdef _OPENMP
//#pragma omp parallel for num_threads(8)
//#endif
  for (int i = 0; i < pos_vec.size(); ++i) {
    Math::vec2 const& pos = pos_vec[i];
    uint32_t const cellNr = hashing(pos);
    count_buffer[cellNr]++;
    //std::atomic_ref<uint32_t> { count_buffer[cellNr] }
    //++;
  }
}

auto Grid2D_Handler_Bounded::handle_visit_neighbors(
    std::vector<Math::vec2> const& poses,
    std::function<void(uint32_t, uint32_t)>& handle,
    std::vector<uint32_t>& partial_sums,
    std::vector<uint32_t>& dense_ids) const noexcept->void {
  for (uint32_t i = 0; i < poses.size(); i++) {
    Math::vec2 const& pos = poses[i];
    int const pxi = std::floor(pos.x * inv_spacing);
    int const pyi = std::floor(pos.y * inv_spacing);
    int const x0 = std::max(pxi - 1, 0);
    int const y0 = std::max(pyi - 1, 0);
    int const x1 = std::min(pxi + 1, int(numX) - 1);
    int const y1 = std::min(pyi + 1, int(numY) - 1);
    for (uint32_t xi = x0; xi <= x1; xi++) {
      for (uint32_t yi = y0; yi <= y1; yi++) {
        uint32_t cellNr = xi * numY + yi;
        uint32_t first = partial_sums[cellNr];
        uint32_t last = partial_sums[cellNr + 1];
        for (uint32_t j = first; j < last; j++) {
          uint32_t id = dense_ids[j];
          handle(i, id);
        }
      }
    }
  }
}

auto Grid2D_Handler_Unbounded::hashing(Math::vec2 const& pos) const noexcept
    -> uint32_t {
    // TODO
  return 0;
}
auto Grid2D_Handler_Unbounded::handle_count(
    std::vector<Math::vec2> const& pos_vec,
    std::vector<uint32_t>& count_buffer) const noexcept -> void {
#ifdef _OPENMP
#pragma omp parallel for num_threads(8)
#endif
  for (int i = 0; i < pos_vec.size(); ++i) {
    Math::vec2 const& pos = pos_vec[i];
    uint32_t cellNr = hashing(pos);
    std::atomic_ref<uint32_t> { count_buffer[cellNr] }
    ++;
  }
}

auto Grid2D_Handler_Unbounded::handle_visit_neighbors(
    std::vector<Math::vec2> const& pos,
    std::function<void(uint32_t, uint32_t)>& handle,
    std::vector<uint32_t>& partial_sums,
    std::vector<uint32_t>& dense_ids) const noexcept->void {
    // TODO
}

auto Grid2D::invalid() noexcept -> void {
  std::fill(count.begin(), count.end(), 0);
  std::fill(partial_sums.begin(), partial_sums.end(), 0);
  std::fill(dense_ids.begin(), dense_ids.end(), 0);
}

auto Grid2D::accum(std::vector<Math::vec2> const& pos_vec) noexcept -> void {
  pos_used = &pos_vec;
  std::vector<uint32_t>& count_ref = count;
  std::visit(
      [&pos_vec, &count_ref](auto const& handler) {
        handler.handle_count(pos_vec, count_ref);
      },
      hash_handler);
}
auto Grid2D::partial_sum() noexcept -> void {
  uint32_t first = 0;
  for (uint32_t i = 0; i < count.size(); i++) {
    first += count[i];
    partial_sums[i] = first;
  }
  partial_sums[count.size()] = first; // guard
}
auto Grid2D::compact() noexcept -> void {
  std::vector<Math::vec2> const& pos_vec = *pos_used;
  std::vector<uint32_t>& count_ref = count;
  std::visit(
      [&pos_vec, &count_ref, this](auto const& handler) {
        for (uint32_t i = 0; i < pos_vec.size(); i++) {
          Math::vec2 const& pos = pos_vec[i];
          uint32_t cellNr = handler.hashing(pos);
          partial_sums[cellNr]--;
          dense_ids[partial_sums[cellNr]] = i;
        }
      },
      hash_handler);
}
auto SpatialHashing2D::init_bounded(Math::uvec2 num, float spacing, uint32_t pcount) noexcept -> void {
  grid.hash_handler =
      Grid2D_Handler_Bounded{num.x, num.y, spacing, 1.f / spacing};
  uint32_t num_grids = num.x * num.y;
  grid.count.resize(num_grids);
  grid.partial_sums.resize(num_grids + 1);
  grid.dense_ids.resize(pcount);
}
auto SpatialHashing2D::prepare(std::vector<Math::vec2> const& pos) noexcept
-> void {
  grid.invalid();
  grid.accum(pos);
  grid.partial_sum();
  grid.compact();
}
auto SpatialHashing2D::visit_neighbors(
    std::vector<Math::vec2> const& pos,
    std::function<void(uint32_t, uint32_t)> handle) noexcept -> void {
  std::visit(
      [&pos, &handle, this](auto const& handler) {
        handler.handle_visit_neighbors(pos, handle, grid.partial_sums,
                                       grid.dense_ids);
      },
      grid.hash_handler);
}

}  // namespace SIByL::PhysicS