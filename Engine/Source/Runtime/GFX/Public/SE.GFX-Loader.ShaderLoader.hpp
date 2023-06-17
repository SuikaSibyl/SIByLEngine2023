#pragma once
#include <array>
#include <vector>
#include <Resource/SE.Core.Resource.hpp>
#include <SE.RHI.hpp>

namespace SIByL::GFX {
SE_EXPORT enum struct ShaderLang {
  UNKNOWN,
  SPIRV,
  GLSL,
  HLSL,
  SLANG,
};

SE_EXPORT struct ShaderReflection {
  enum struct ResourceType {
    Undefined,
    UniformBuffer,
    StorageBuffer,
    StorageImages,
    SampledImages,
    AccelerationStructure,
  };
  enum struct ResourceFlag : uint32_t {
    None = 0,
    NotReadable = 1 << 0,
    NotWritable = 1 << 1,
  };
  using ResourceFlags = uint32_t;
  struct ResourceEntry {
    ResourceType type = ResourceType::Undefined;
    ResourceFlags flags = 0;
    RHI::ShaderStagesFlags stages = 0;
  };
  struct PushConstantEntry {
    uint32_t index = -1;
    uint32_t offset = -1;
    uint32_t range = -1;
    RHI::ShaderStagesFlags stages = 0;
  };
  std::vector<PushConstantEntry> pushConstant;
  std::vector<std::vector<ResourceEntry>> bindings;

  struct BindingInfo {
    ResourceType type = ResourceType::Undefined;
    uint32_t set;
    uint32_t binding;
  };
  std::unordered_map<std::string, BindingInfo> bindingInfo;

  static auto toBindGroupLayoutDescriptor(
      std::vector<ResourceEntry> const& bindings) noexcept
      -> RHI::BindGroupLayoutDescriptor;
  auto operator+(ShaderReflection const& reflection) const -> ShaderReflection;
};

auto SPIRV_TO_Reflection(Core::Buffer* code,
                                RHI::ShaderStages stage) noexcept
    -> ShaderReflection;

SE_EXPORT struct ShaderLoader {};

SE_EXPORT struct ShaderLoader_SPIRV {};

SE_EXPORT struct ShaderLoader_GLSL {};

SE_EXPORT struct ShaderLoader_HLSL {};

SE_EXPORT struct ShaderLoader_SLANG {
  static auto load(
      std::string const& filepath,
      std::vector<std::pair<std::string, RHI::ShaderStages>> const&
          entrypoints,
      std::vector<std::pair<char const*, char const*>> const& macros =
          {}) noexcept -> std::vector<Core::GUID>;

  template<size_t N>
  static auto load(std::string const& filepath,
                   std::array<std::pair<std::string, RHI::ShaderStages>,
      N> const& entrypoints,
                   std::vector<std::pair<char const*, char const*>> const&
                       macros = {}) noexcept
      -> std::array<Core::GUID, N> {
    std::vector<std::pair<std::string, RHI::ShaderStages>> vec;
    for (size_t i = 0; i < N; ++i) {
      vec.push_back(
          std::make_pair(entrypoints[i].first, entrypoints[i].second));
    }
    std::vector<Core::GUID> guids = load(filepath, vec, macros);
    std::array<Core::GUID, N> arr;
    for (size_t i = 0; i < N; ++i) arr[i] = guids[i];
    return arr;
  }
};
}  // namespace SIByL::GFX