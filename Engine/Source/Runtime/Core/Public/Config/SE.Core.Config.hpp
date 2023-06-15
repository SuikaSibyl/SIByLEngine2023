#pragma once
#include "../../../Prelude/Public/common_config.hpp"
#include <string>
#include <unordered_map>

namespace SIByL::Core {
SE_EXPORT struct RuntimeConfig {
  RuntimeConfig();
  static auto get() -> RuntimeConfig const*;
  auto string_property(std::string const& name) const noexcept -> std::string;

 private:
  std::unordered_map<std::string, std::string> string_properties;
  static RuntimeConfig* singleton;
};
}