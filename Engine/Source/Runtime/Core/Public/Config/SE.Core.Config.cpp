#include <yaml-cpp/yaml.h>
#include "SE.Core.Config.hpp"
#include "../IO/SE.Core.IO.hpp"
#include "../Print/SE.Core.Log.hpp"

namespace SIByL::Core {
RuntimeConfig* RuntimeConfig::singleton = nullptr;

RuntimeConfig::RuntimeConfig() {
  // gameObjects.clear();
  Core::Buffer condigdata;
  Core::syncReadFile("./runtime.config", condigdata);
  YAML::NodeAoS data = YAML::Load(reinterpret_cast<char*>(condigdata.data));
  // check scene name
  string_properties["engine_path"] = data["engine_path"].as<std::string>();
}

auto RuntimeConfig::get() -> RuntimeConfig const* {
  if (singleton == nullptr)
    singleton = new RuntimeConfig();
  return singleton;
}

auto RuntimeConfig::string_property(std::string const& name) const noexcept
-> std::string {
  auto find = string_properties.find(name);
  if (find == string_properties.end()) {
    return "UNKOWN_STRING";
  } 
  else {
    return find->second;
  }
}

}