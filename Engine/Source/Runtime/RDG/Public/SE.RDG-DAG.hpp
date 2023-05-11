#pragma once
#include <set>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <common_config.hpp>

namespace SIByL::RDG {
SE_EXPORT struct DAG {
  auto addEdge(uint32_t src, uint32_t dst) noexcept -> void;
  auto reverse() const noexcept -> DAG;
  std::unordered_map<uint32_t, std::set<uint32_t>> adj;
};

SE_EXPORT auto flatten_bfs(DAG const& g, size_t output) noexcept
    -> std::optional<std::vector<size_t>>;
}  // namespace SIByL::RDG