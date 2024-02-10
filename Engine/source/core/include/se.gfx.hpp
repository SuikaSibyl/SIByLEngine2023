#pragma once
#include <se.core.hpp>
#include <se.rhi.hpp>
#include <unordered_map>
#include <type_traits>

#define MULTIFRAME_FLIGHTS_COUNT 2

// ===========================================================================
// Base class definition

namespace se::gfx {
struct SIByL_API ViewIndex {
  rhi::TextureViewType type;
  uint32_t mostDetailedMip;
  uint32_t mipCount;
  uint32_t firstArraySlice;
  uint32_t arraySize;
  bool operator==(ViewIndex const& p) const;
};
}

namespace std {
template <>
struct hash<se::gfx::ViewIndex> {
  size_t operator()(const se::gfx::ViewIndex& s) const noexcept {
    return hash<uint32_t>()((uint32_t)s.type) +
           hash<uint32_t>()(s.mostDetailedMip) + hash<uint32_t>()(s.mipCount) +
           hash<uint32_t>()(s.firstArraySlice) + hash<uint32_t>()(s.arraySize);
  }
};
}  // namespace std

namespace se::gfx {
/** Interface of resources */
struct SIByL_API Resource {
  /** virtual destructor */
  virtual ~Resource() = default;
  /** get name */
  virtual auto getName() const noexcept -> char const* = 0;
};

// Base class definition
// ===========================================================================
// Shader Resource

struct SIByL_API ShaderReflection {
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
    rhi::ShaderStages stages = 0;
    uint32_t arraySize = 1;
  };
  struct PushConstantEntry {
    uint32_t index = -1;
    uint32_t offset = -1;
    uint32_t range = -1;
    rhi::ShaderStages stages = 0;
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
    -> rhi::BindGroupLayoutDescriptor;
  auto operator+(ShaderReflection const& reflection) const -> ShaderReflection;
};

auto SIByL_API SPIRV_TO_Reflection(se::buffer* code,
  rhi::ShaderStageBit stage) noexcept -> ShaderReflection;

struct SIByL_API ShaderModule : public Resource {
  /** rhi shader module */
  std::unique_ptr<rhi::ShaderModule> shaderModule = nullptr;
  /** reflection Information of the shader module */
  ShaderReflection reflection;
  /** get name */
  virtual auto getName() const noexcept -> char const*;
};

// loader
// -------------------------------------------------

using ShaderHandle = ex::resource<se::gfx::ShaderModule>;

struct SIByL_API ShaderModuleLoader {
  using result_type = std::shared_ptr<ShaderModule>;

  struct from_spirv_tag {};
  struct from_glsl_tag {};
  struct from_slang_tag {};

  result_type operator()(from_spirv_tag, se::buffer* buffer, rhi::ShaderStageBit stage);
  result_type operator()(from_glsl_tag, std::string const& filepath, 
    rhi::ShaderStageBit stage, std::vector<std::string> const& argv = {});
};

// Shader Resource
// ===========================================================================
// Base class definition

struct SIByL_API Buffer : public Resource {
  /** ctors & rval copies */
  Buffer() = default;
  Buffer(Buffer&& buffer) = default;
  Buffer(Buffer const& buffer) = delete;
  auto operator=(Buffer&& buffer) -> Buffer& = default;
  auto operator=(Buffer const& buffer) -> Buffer& = delete;
  /** the gpu vertex buffer */
  std::unique_ptr<rhi::Buffer> buffer = nullptr;
  /** get name */
  virtual auto getName() const noexcept -> char const* override;
  /** release the resource */
  auto release() noexcept -> void;
};

using BufferHandle = ex::resource<se::gfx::Buffer>;

struct SIByL_API Texture : public Resource {
  /** ctors & rval copies */
  Texture() = default;
  Texture(Texture&& texture) = default;
  Texture(Texture const& texture) = delete;
  auto operator=(Texture&& texture) -> Texture& = default;
  auto operator=(Texture const& texture) -> Texture& = delete;
  /** texture */
  std::unique_ptr<rhi::Texture> texture = nullptr;
  /** texture display view*/
  std::unique_ptr<rhi::TextureView> originalView = nullptr;
  /** path string */
  std::optional<std::string> resourcePath;
  /** path string */
  std::optional<std::vector<std::string>> resourcePathArray;
  /** differentiable attributes */
  uint32_t differentiable_channels = 0u;
  /** name */
  std::string name;
  /** get name */
  virtual auto getName() const noexcept -> char const*;
  /** Get the UAV of the texture */
  auto getUAV(uint32_t mipLevel, uint32_t firstArraySlice,
              uint32_t arraySize) noexcept -> rhi::TextureView*;
  /** Get the RTV of the texture */
  auto getRTV(uint32_t mipLevel, uint32_t firstArraySlice,
              uint32_t arraySize) noexcept -> rhi::TextureView*;
  /** Get the RTV of the texture */
  auto getDSV(uint32_t mipLevel, uint32_t firstArraySlice,
              uint32_t arraySize) noexcept -> rhi::TextureView*;
  /** Get the SRV of the texture */
  auto getSRV(uint32_t mostDetailedMip, uint32_t mipCount,
              uint32_t firstArraySlice, uint32_t arraySize) noexcept -> rhi::TextureView*;
 private:
  std::unordered_map<ViewIndex, std::unique_ptr<rhi::TextureView>> viewPool;
};

using TextureHandle = ex::resource<se::gfx::Texture>;

// Base class definition
// ===========================================================================
// GFX Context definition

struct SIByL_API GFXContext {
  static auto initialize(rhi::Device* device) noexcept -> void;
  static auto finalize() noexcept -> void;
  static rhi::Device* device;
  static ex::resource_cache<ShaderModule, ShaderModuleLoader> shaders;
  
  static auto load_shader_spirv(
    se::buffer* buffer,
    rhi::ShaderStageBit stage
  ) noexcept -> ShaderHandle;
  static auto load_shader_glsl(
    const char* path, 
    rhi::ShaderStageBit stage
  ) noexcept -> ShaderHandle;
  static auto load_shader_slang(
    std::string const& path,
    std::vector<std::pair<std::string, rhi::ShaderStageBit>> const& entrypoints,
    std::vector<std::pair<char const*, char const*>> const& macros = {},
    bool glsl_intermediate = false
  ) noexcept -> std::vector<ShaderHandle>;
  template<size_t N>
  static auto load_shader_slang(
    std::string const& filepath,
    std::array<std::pair<std::string, rhi::ShaderStageBit>, N> const& entrypoints,
    std::vector<std::pair<char const*, char const*>> const& macros = {},
    bool glsl_intermediate = false) noexcept -> std::array<ShaderHandle, N> {
      std::vector<std::pair<std::string, rhi::ShaderStageBit>> vec;
      for (size_t i = 0; i < N; ++i)
        vec.push_back(std::make_pair(entrypoints[i].first, entrypoints[i].second));
      std::vector<ShaderHandle> handdles = GFXContext::load_shader_slang(filepath, vec, macros, glsl_intermediate);
      std::array<ShaderHandle, N> arr;
      for (size_t i = 0; i < N; ++i) arr[i] = handdles[i];
      return arr;
  }

};
}