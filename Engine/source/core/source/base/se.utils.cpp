#define DLIB_EXPORT
#include <se.utils.hpp>
#undef DLIB_EXPORT
#include <se.core.hpp>
#include <yaml-cpp/yaml.h>

namespace se {
  RuntimeConfig* RuntimeConfig::singleton = nullptr;
  std::string RuntimeConfig::config_file_path = "./runtime.config";

  RuntimeConfig::RuntimeConfig() {
    se::buffer condigdata;
    se::syncReadFile("./runtime.config", condigdata);
    YAML::NodeAoS data = YAML::Load(reinterpret_cast<char*>(condigdata.data));
    // check scene name
    string_properties["engine_path"] = data["engine_path"].as<std::string>();
    if (data["shader_path"]) {
      std::vector<std::string> shader_paths;
      for (auto node : data["shader_path"]) {
        shader_paths.push_back(node.as<std::string>());
      }
      string_array_properties["shader_path"] = shader_paths;
    }
  }
  
  auto RuntimeConfig::set_config_file(std::string const& path) noexcept -> void {
    config_file_path = path;
  }

  auto RuntimeConfig::get() -> RuntimeConfig const* {
    if (singleton == nullptr)
      singleton = new RuntimeConfig();
      return singleton;
  }

  auto RuntimeConfig::string_property(std::string const& name) const noexcept -> std::string {
    auto find = string_properties.find(name);
    if (find == string_properties.end()) { return "UNKOWN_STRING"; }
    else { return find->second; }
  }
  auto RuntimeConfig::string_array_property(std::string const& name) const noexcept -> std::vector<std::string> const& {
    auto find = string_array_properties.find(name);
    if (find == string_array_properties.end()) { return std::vector<std::string>{}; }
    else {return find->second; }
  }

}