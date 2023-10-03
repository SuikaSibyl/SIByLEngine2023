#pragma once
#include "SE.RHI-Interface.hpp"
#include <SE.Core.Utility.hpp>
#include <SE.Core.Profiler.hpp>
#include <unordered_map>
#include <string>
#include <array>

namespace SIByL::RHI {
SE_EXPORT struct DeviceProfilerManager {
  SINGLETON(DeviceProfilerManager, {});
  /** initialize the profiler by a device */
  auto initialize(RHI::Device* device,
                  std::string const& name = "device_default") noexcept -> void;
  auto finalize() noexcept -> void;
  auto clear() noexcept -> void;

  auto flushResults() noexcept -> void;
  auto reset(RHI::CommandEncoder* cmdEncoder) noexcept -> void;

  auto beginSegment(RHI::CommandEncoder* cmdEncoder,
                    RHI::PipelineStages stageMask,
                    std::string const& name) noexcept -> void;
  auto endSegment(RHI::CommandEncoder* cmdEncoder,
                  RHI::PipelineStages stageMask,
                  std::string const& name) noexcept -> void;

  /** the device the profiler manager works on */
  RHI::Device* device = nullptr;
  std::unique_ptr<RHI::QuerySet> sets[2];
  uint32_t current_flight = 1;
  using Label2QueryIDMap = std::unordered_map<std::string, std::pair<uint32_t, uint32_t>>;
  Label2QueryIDMap maps[2];
  uint32_t queryCount[2];
  std::array<uint64_t, 256> buffer;
  std::unique_ptr<Core::ProfileSession> device_session = nullptr;
  struct LabelStatistics {
    uint64_t accumulation = 0;
    uint64_t count = 0;
  };
  using Label2Statistics = std::unordered_map<std::string, LabelStatistics>;
  Label2Statistics statistics;
  bool enabled = false;
};
}