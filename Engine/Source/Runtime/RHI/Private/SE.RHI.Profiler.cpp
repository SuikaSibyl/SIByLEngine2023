#include "../Public/SE.RHI.Profiler.hpp"

namespace SIByL::RHI {
auto DeviceProfilerManager::flushResults() noexcept -> void {
  current_flight = (current_flight + 1) % 2;
  if (queryCount[current_flight] != 0) {
    sets[current_flight]->resolveQueryResult(
        0, queryCount[current_flight], sizeof(uint64_t) * 256, buffer.data(),
        sizeof(uint64_t) * 2,
        (uint32_t)RHI::QueryResultBits::RESULT_64 |
        (uint32_t)RHI::QueryResultBits::RESULT_WITH_AVAILABILITY);

    for (auto& pair : maps[current_flight]) {
      uint64_t const timestamp_beg = buffer[pair.second.first * 2 + 0];
      uint64_t const availabil_beg = buffer[pair.second.first * 2 + 1];
      uint64_t const timestamp_end = buffer[pair.second.second * 2 + 0];
      uint64_t const availabil_end = buffer[pair.second.second * 2 + 1];
      if (availabil_beg > 0 && availabil_end > 0) {
        device_session->writeSegment(
            {pair.first, 0, timestamp_beg, timestamp_end});

        if (statistics.find(pair.first) == statistics.end()) {
          statistics[pair.first] = {timestamp_end - timestamp_beg, 0};
        } else {
          LabelStatistics& stat = statistics[pair.first];
          stat.accumulation += timestamp_end - timestamp_beg;
          stat.count++;
        }
      }
    }
  }
}

auto DeviceProfilerManager::initialize(RHI::Device* device,
                                       std::string const& name) noexcept
    -> void {
  this->device = device;
  // create the query sets
  sets[0] = device->createQuerySet({QueryType::TIMESTAMP, 128});
  sets[1] = device->createQuerySet({QueryType::TIMESTAMP, 128});
  queryCount[0] = queryCount[1] = 0;
  device_session = std::make_unique<Core::ProfileSession>(name);
}

auto DeviceProfilerManager::finalize() noexcept -> void {
  sets[0] = nullptr;
  sets[1] = nullptr;
  device_session = nullptr;
}

auto DeviceProfilerManager::clear() noexcept -> void {
  device_session = std::make_unique<Core::ProfileSession>("new_session");
  statistics.clear();
}

auto DeviceProfilerManager::reset(RHI::CommandEncoder* cmdEncoder) noexcept
    -> void {
  cmdEncoder->resetQuerySet(sets[current_flight].get(), 0, 128);
  maps[current_flight].clear();
  queryCount[current_flight] = 0;
}

auto DeviceProfilerManager::beginSegment(
    RHI::CommandEncoder* cmdEncoder,
    RHI::PipelineStages stageMask,
    std::string const& name
) noexcept -> void {
  if (enabled) {
    maps[current_flight][name].first = queryCount[current_flight];
    cmdEncoder->writeTimestamp(sets[current_flight].get(), stageMask,
                               queryCount[current_flight]++);
  }
}

auto DeviceProfilerManager::endSegment(
    RHI::CommandEncoder* cmdEncoder,
    RHI::PipelineStages stageMask,
    std::string const& name) noexcept
    -> void {
  if (enabled) {
    maps[current_flight][name].second = queryCount[current_flight];
    cmdEncoder->writeTimestamp(sets[current_flight].get(), stageMask,
                               queryCount[current_flight]++);
  }
}
}