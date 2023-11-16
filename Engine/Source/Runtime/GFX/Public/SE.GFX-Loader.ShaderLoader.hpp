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
    ReadonlyImage,
    Sampler,
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
    uint32_t arraySize = 1;
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

SE_EXPORT struct ShaderLoader_SPIRV {
  /**
   * Load one or more shader models from a single file.
   * @param buffer: the buffer containing the spir-v code.
   */
  static auto load(Core::Buffer* buffer, RHI::ShaderStages stage) noexcept
      -> Core::GUID;
};

SE_EXPORT struct ShaderLoader_GLSL {
  /**
   * Load one or more shader models from a single file.
   * @param filepath: the filepath to the loaded file
   * @param argv: argv to the compiling cmdline.
   */
  static auto load(std::string const& filepath, RHI::ShaderStages stage,
                   std::vector<std::string> const& argv = {}) noexcept -> Core::GUID;
};

SE_EXPORT struct ShaderLoader_HLSL {};

SE_EXPORT struct ShaderLoader_SLANG {
  /**
   * Load one or more shader models from a single file.
   * @param filepath: the filepath to the loaded file
   * @param entrypoints: entrypoints to the shader models.
   * @param macros: the macros added to compilation.
   * @param glsl_intermediate: use glsl as the first target of SLANG.
   */
  static auto load(
      std::string const& filepath,
      std::vector<std::pair<std::string, RHI::ShaderStages>> const& entrypoints,
      std::vector<std::pair<char const*, char const*>> const& macros = {},
      bool glsl_intermediate = false) noexcept -> std::vector<Core::GUID>;

  /**
   * An templated array version of "load()" above,
   * so that the caller could receive the result in a [a,b,...] grammer sugar.
   * @param filepath: the filepath to the loaded file
   * @param entrypoints: entrypoints to the shader models.
   * @param macros: the macros added to compilation.
   * @param glsl_intermediate: use glsl as the first target of SLANG.
   */
  template<size_t N>
  static auto load(
      std::string const& filepath,
      std::array<std::pair<std::string, RHI::ShaderStages>, N> const& entrypoints,
      std::vector<std::pair<char const*, char const*>> const& macros = {},
      bool glsl_intermediate = false) noexcept -> std::array<Core::GUID, N> 
  {
    std::vector<std::pair<std::string, RHI::ShaderStages>> vec;
    for (size_t i = 0; i < N; ++i)
      vec.push_back(std::make_pair(entrypoints[i].first, entrypoints[i].second));
    std::vector<Core::GUID> guids = load(filepath, vec, macros, glsl_intermediate);
    std::array<Core::GUID, N> arr;
    for (size_t i = 0; i < N; ++i) arr[i] = guids[i];
    return arr;
  }
};
}  // namespace SIByL::GFX