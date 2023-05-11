#pragma once

#include <common_config.hpp>

namespace SIByL::Core {
SE_EXPORT struct Layer {
  /** virtual destructor */
  virtual ~Layer() = default;
};

/** An interface for singleton manager for engine*/
SE_EXPORT struct Manager {
  Manager() = default;
  virtual ~Manager() = default;
  /* start up the manager */
  virtual auto startUp() noexcept -> void {}
  /* shut down the manager */
  virtual auto shutDown() noexcept -> void {}
};
}  // namespace SIByL::Core