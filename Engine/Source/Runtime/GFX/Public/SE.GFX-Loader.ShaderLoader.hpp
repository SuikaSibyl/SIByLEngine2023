#pragma once
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

SE_EXPORT struct ShaderLoader_SLANG {};
}  // namespace SIByL::GFX