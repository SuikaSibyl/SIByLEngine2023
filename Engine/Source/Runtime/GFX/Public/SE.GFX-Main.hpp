#pragma once

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <format>
#include <functional>
#include <future>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <type_traits>
#include <typeinfo>
#include <unordered_map>
#include <utility>
#include <vector>

#include <ECS/SE.Core.ECS.hpp>
#include <SE.Math.Misc.hpp>
#include <SE.Math.Geometric.hpp>
#include <SE.RHI.hpp>
#include <SE.GFX-GFXConfig.hpp>
#include <SE.GFX-Loader.ShaderLoader.hpp>
#include <SE.GFX-SerializeUtils.hpp>
#include <Resource/SE.Core.Resource.hpp>
#include <SE.Image.hpp>
#include "../../Application/Public/SE.Application.Config.h"

namespace SIByL::GFX {
struct ViewIndex {
  RHI::TextureViewType type;
  uint32_t mostDetailedMip;
  uint32_t mipCount;
  uint32_t firstArraySlice;
  uint32_t arraySize;
  bool operator==(ViewIndex const& p) const {
    return type == p.type && mostDetailedMip == p.mostDetailedMip &&
           mipCount == p.mipCount && firstArraySlice == p.firstArraySlice &&
           arraySize == p.arraySize;
  }
};
}  // namespace SIByL::GFX

namespace std {
template <>
struct hash<SIByL::GFX::ViewIndex> {
  size_t operator()(const SIByL::GFX::ViewIndex& s) const noexcept {
    return hash<uint32_t>()((uint32_t)s.type) +
           hash<uint32_t>()(s.mostDetailedMip) + hash<uint32_t>()(s.mipCount) +
           hash<uint32_t>()(s.firstArraySlice) + hash<uint32_t>()(s.arraySize);
  }
};
}  // namespace std

namespace SIByL::GFX {
struct Buffer;
struct ShaderModule;
struct Mesh;
struct Texture;
struct Sampler;

/**
 * GFX Loader is a singleton sub-module of GFX Manager.
 * It manages how to load resources from
 */
SE_EXPORT struct GFXLoader {};

SE_EXPORT struct Buffer : public Core::Resource {
  /** ctors & rval copies */
  Buffer() = default;
  Buffer(Buffer&& buffer) = default;
  Buffer(Buffer const& buffer) = delete;
  auto operator=(Buffer&& buffer) -> Buffer& = default;
  auto operator=(Buffer const& buffer) -> Buffer& = delete;
  /** the gpu vertex buffer */
  std::unique_ptr<RHI::Buffer> buffer = nullptr;
  /** guid of the buffer resource */
  Core::GUID guid;
  /** get name */
  virtual auto getName() const noexcept -> char const* override {
    return buffer->getName().c_str();
  }
  /** release the resource */
  auto release() noexcept -> void;
};

SE_EXPORT struct ShaderModule : public Core::Resource {
  /** rhi shader module */
  std::unique_ptr<RHI::ShaderModule> shaderModule = nullptr;
  /** reflection Information of the shader module */
  ShaderReflection reflection;
  /** get name */
  virtual auto getName() const noexcept -> char const* {
    return shaderModule->getName().c_str();
  }
};

SE_EXPORT struct Mesh : public Core::Resource {
  /** ctors & rval copies */
  Mesh() = default;
  Mesh(Mesh&& mesh) = default;
  Mesh(Mesh const& mesh) = delete;
  auto operator=(Mesh&& mesh) -> Mesh& = default;
  auto operator=(Mesh const& mesh) -> Mesh& = delete;
  /* vertex buffer layout */
  RHI::VertexBufferLayout vertexBufferLayout = {};
  /** primitive state */
  RHI::PrimitiveState primitiveState = {};
  /** the gpu|device vertex buffer */
  std::unique_ptr<RHI::Buffer> vertexBuffer_device = nullptr;
  std::unique_ptr<RHI::Buffer> positionBuffer_device = nullptr;
  std::unique_ptr<RHI::Buffer> indexBuffer_device = nullptr;
  /** the cpu|host vertex/index/position buffers */
  Core::Buffer vertexBuffer_host = {};
  Core::Buffer positionBuffer_host = {};
  Core::Buffer indexBuffer_host = {};
  /** host-device copy */
  struct DeviceHostBufferInfo {
    uint32_t size = 0;
    bool onHost = false;
    bool onDevice = false;
  };
  DeviceHostBufferInfo vertexBufferInfo;
  DeviceHostBufferInfo positionBufferInfo;
  DeviceHostBufferInfo indexBufferInfo;
  /** binded ORID */
  Core::ORID ORID = Core::INVALID_ORID;
  /** submeshes */
  struct Submesh {
    uint32_t offset;
    uint32_t size;
    uint32_t baseVertex;
    uint32_t matID;
  };
  std::vector<Submesh> submeshes;
  /** resource name */
  std::string name = "New Mesh";
  /** AABB bounding box */
  Math::bounds3 aabb;
  /** compute the surface area */
  auto surfaceAreaEveryPrimitive(Math::mat4 const& transform) noexcept
      -> std::tuple<float, std::vector<float>>;
  /** serialize */
  auto serialize() noexcept -> void;
  /** deserialize */
  auto deserialize(RHI::Device* device, Core::ORID orid) noexcept
      -> void;
  /** get name */
  virtual auto getName() const noexcept -> char const* { return name.c_str(); }
};

SE_EXPORT struct Texture : public Core::Resource {
  /** ctors & rval copies */
  Texture() = default;
  Texture(Texture&& texture) = default;
  Texture(Texture const& texture) = delete;
  auto operator=(Texture&& texture) -> Texture& = default;
  auto operator=(Texture const& texture) -> Texture& = delete;
  /** serialize */
  auto serialize() noexcept -> void;
  /** deserialize */
  auto deserialize(RHI::Device* device, Core::ORID orid) noexcept
      -> void;
  /** resrouce GUID */
  Core::GUID guid;
  /** resrouce ORID */
  Core::ORID orid = Core::INVALID_ORID;
  /** texture */
  std::unique_ptr<RHI::Texture> texture = nullptr;
  /** texture display view*/
  std::unique_ptr<RHI::TextureView> originalView = nullptr;
  /** texture display view arrays*/
  std::vector<std::unique_ptr<RHI::TextureView>> viewArrays;
  /** path string */
  std::optional<std::string> resourcePath;
  /** path string */
  std::optional<std::vector<std::string>> resourcePathArray;
  /** differentiable attributes */
  uint32_t differentiable_channels = 0u;
  /** name */
  std::string name;
  /** get name */
  virtual auto getName() const noexcept -> char const* {
    return texture->getName().c_str();
  }
  /** Get the UAV of the texture */
  auto getUAV(uint32_t mipLevel, uint32_t firstArraySlice,
              uint32_t arraySize) noexcept -> RHI::TextureView*;
  /** Get the RTV of the texture */
  auto getRTV(uint32_t mipLevel, uint32_t firstArraySlice,
              uint32_t arraySize) noexcept -> RHI::TextureView*;
  /** Get the RTV of the texture */
  auto getDSV(uint32_t mipLevel, uint32_t firstArraySlice,
              uint32_t arraySize) noexcept -> RHI::TextureView*;
  /** Get the SRV of the texture */
  auto getSRV(uint32_t mostDetailedMip, uint32_t mipCount,
              uint32_t firstArraySlice, uint32_t arraySize) noexcept
      -> RHI::TextureView*;
 private:
  std::unordered_map<ViewIndex, std::unique_ptr<RHI::TextureView>> viewPool;
};

SE_EXPORT struct Sampler : public Core::Resource {
  /** ctors & rval copies */
  Sampler() = default;
  Sampler(Sampler&& sampler) = default;
  Sampler(Sampler const& sampler) = delete;
  auto operator=(Sampler&& sampler) -> Sampler& = default;
  auto operator=(Sampler const& sampler) -> Sampler& = delete;
  /* rhi sampler */
  std::unique_ptr<RHI::Sampler> sampler = nullptr;
  /** get name */
  virtual auto getName() const noexcept -> char const* {
    return sampler->getName().c_str();
  }
};

/** Material Template */
SE_EXPORT struct MaterialTemplate {
  /** add a const data entry to the template */
  auto addConstantData(std::string const& name,
                              RHI::DataFormat format) noexcept
      -> MaterialTemplate&;
  /** add a texture entry to the template */
  auto addTexture(std::string const& name) noexcept -> MaterialTemplate&;
  /** const data entries */
  std::unordered_map<std::string, RHI::DataFormat> constDataEntries;
  /** texture entries */
  std::vector<std::string> textureEntries;
  /** bsdf id */
  uint32_t bsdf_id;
  /** material name */
  std::string materialName;
};

/** Material */
SE_EXPORT struct Material : public Core::Resource {
  /** texture resource entry */
  enum struct TexFlag {
    Cubemap = 1 << 1,
    NormalMap = 1 << 2,
    VideoClip = 1 << 3,
  };
  struct TextureEntry {
    Core::GUID guid;
    uint32_t flags = 0;
    RHI::SamplerDescriptor sampler;
  };
  /** add a const data entry to the template */
  auto addConstantData(std::string const& name, RHI::DataFormat format) noexcept
      -> Material&;
  /** add a texture entry to the template */
  auto addTexture(std::string const& name, TextureEntry const& entry) noexcept
      -> Material&;
  /** register from a template */
  auto registerFromTemplate(MaterialTemplate const& mat_template) noexcept
      -> void;
  /** get name */
  virtual auto getName() const noexcept -> char const* override {
    return name.c_str();
  }
  /** serialize */
  auto serialize() noexcept -> void;
  /** deserialize */
  auto deserialize(RHI::Device* device, Core::ORID orid) noexcept
      -> void;
  /** load the material from path */
  auto loadPath() noexcept -> void;
  /** all textures in material */
  std::unordered_map<std::string, TextureEntry> textures;
  /** ORID of the material */
  Core::ORID ORID = Core::INVALID_ORID;
  /** BxDF ID */
  uint32_t BxDF;
  /** resource name */
  std::string name = "New Material";
  /** emission */
  bool isEmissive = false;
  /** alpha test info */
  enum struct AlphaState {
    Opaque,
    DitherDiscard,
    AlphaCutoff,
  } alphaState = AlphaState::Opaque;
  float alphaThreshold = 1.f;
  /** Data */
  Math::vec3 baseOrDiffuseColor = Math::vec3(1.f);
  Math::vec3 specularColor = Math::vec3(1.f);
  Math::vec3 emissiveColor = Math::vec3(0.f);
  float roughness = 1.f;
  float metalness = 0.f;
  float eta = 1.2f;
  /** resource path */
  std::string path;
  /** is dirty */
  bool isDirty = false;
};

SE_EXPORT struct TagComponent {
  /** constructor */
  TagComponent(std::string const& name = "New GameObject") : name(name) {}
  // game object name
  std::string name;
  /** serialize */
  static auto serialize(void* emitter, Core::EntityHandle const& handle)
      -> void;
  /** deserialize */
  static auto deserialize(void* compAoS, Core::EntityHandle const& handle)
      -> void;
};

SE_EXPORT struct TransformComponent {
  /** constructor */
  TransformComponent() = default;
  /** decomposed transform - translation */
  Math::vec3 translation = {0.0f, 0.0f, 0.0f};
  /** decomposed transform - eulerAngles */
  Math::vec3 eulerAngles = {0.0f, 0.0f, 0.0f};
  /** decomposed transform - scale */
  Math::vec3 scale = {1.0f, 1.0f, 1.0f};
  /** integrated world transform */
  Math::Transform transform = {};
  /** previous integrated world transform */
  Math::Transform previousTransform = {};
  /** check whether the transform is a static one */
  uint32_t static_param = 1;
  /** get transform */
  auto getTransform() noexcept -> Math::mat4;
  /** get rotated forward */
  auto getRotatedForward() const noexcept -> Math::vec3 {
    Math::vec4 rotated = Math::mat4::rotateZ(eulerAngles.z) *
                         Math::mat4::rotateY(eulerAngles.y) *
                         Math::mat4::rotateX(eulerAngles.x) *
                         Math::vec4(0, 0, -1, 0);
    return Math::vec3(rotated.x, rotated.y, rotated.z);
  }
  /** serialize */
  static auto serialize(void* emitter, Core::EntityHandle const& handle)
      -> void;
  /** deserialize */
  static auto deserialize(void* compAoS, Core::EntityHandle const& handle)
      -> void;
};

/** Game object handle is also the entity handle contained */
SE_EXPORT using GameObjectHandle = Core::EntityHandle;
SE_EXPORT inline GameObjectHandle NULL_GO = Core::NULL_ENTITY;
/** Game object is a hierarchical wrapper of entity */
SE_EXPORT struct GameObject {
  auto getEntity() noexcept -> Core::Entity { return Core::Entity{entity}; }
  GameObjectHandle parent = NULL_GO;
  Core::EntityHandle entity = Core::NULL_ENTITY;
  std::vector<GameObjectHandle> children = {};
};

SE_EXPORT struct Scene : public Core::Resource {
  /** add a new entity */
  auto createGameObject(GameObjectHandle parent = NULL_GO) noexcept
      -> GameObjectHandle;
  /** remove an entity */
  auto removeGameObject(GameObjectHandle handle) noexcept -> void;
  /** get an game object */
  auto getGameObject(GameObjectHandle handle) noexcept -> GameObject*;
  /** move an game object */
  auto moveGameObject(GameObjectHandle handle) noexcept -> void;
  /** serialize scene */
  auto serialize(std::filesystem::path path) noexcept -> void;
  /** deserialize scene */
  auto deserialize(std::filesystem::path path) noexcept -> void;
  /** release */
  auto release() noexcept -> void;
  /** name description */
  std::string name = "new scene";
  /** mapping handle to GameObject */
  std::unordered_map<GameObjectHandle, GameObject> gameObjects;
  /** show wether the scene is modified */
  bool isDirty = false;
  /** get name */
  virtual auto getName() const noexcept -> char const* override {
    return name.c_str();
  }
};

/**
 * Predefined components in GFX Module, including:
 * - CameraComponent
 * - MeshReference
 * - MeshRenderer
 * - LightComponent
 */

SE_EXPORT struct CameraComponent {
  auto getViewMat() noexcept -> Math::mat4;
  auto getProjectionMat() const noexcept -> Math::mat4;

  enum struct ProjectType {
    PERSPECTIVE,
    ORTHOGONAL,
  };

  /** serialize */
  static auto serialize(void* emitter, Core::EntityHandle const& handle)
      -> void;
  /** deserialize */
  static auto deserialize(void* compAoS, Core::EntityHandle const& handle)
      -> void;

  float fovy = 45.f;
  float aspect = 1;
  float near = 0.1f;
  float far = 100.0f;

  float left_right = 0;
  float bottom_top = 0;
  ProjectType projectType = ProjectType::PERSPECTIVE;
  bool isPrimaryCamera = false;

 private:
  Math::mat4 view;
  Math::mat4 projection;
};

SE_EXPORT struct MeshReference {
  /* constructor */
  MeshReference() = default;
  /** mesh */
  Mesh* mesh = nullptr;
  /** custom primitive flag */
  size_t customPrimitiveFlag = 0;
  /** serialize */
  static auto serialize(void* emitter, Core::EntityHandle const& handle)
      -> void;
  /** deserialize */
  static auto deserialize(void* compAoS, Core::EntityHandle const& handle)
      -> void;
};

SE_EXPORT struct MeshRenderer {
  /* constructor */
  MeshRenderer() = default;
  /** materials in renderer */
  std::vector<Material*> materials = {};
  /** serialize */
  static auto serialize(void* emitter, Core::EntityHandle const& handle)
      -> void;
  /** deserialize */
  static auto deserialize(void* compAoS, Core::EntityHandle const& handle)
      -> void;
};

SE_EXPORT struct LightComponent {
  /* constructor */
  LightComponent() = default;
  /* light type */
  enum struct LightType {
    // singular lights
    DIRECTIONAL,
    POINT,
    SPOT,
    // area lights
    TRIANGLE,
    RECTANGLE,
    MESH_PRIMITIVE,
    // environment mao
    ENVIRONMENT,
    // virtual light
    VPL,
    MAX_ENUM,
  } type;
  /* light intensity to scale the light */
  Math::vec3 intensity = Math::vec3(1, 1, 1);
  /* packed data associated with light */
  Math::vec4 packed_data_0;
  Math::vec4 packed_data_1;
  /* */
  bool isDirty = false;
  /** texture guid */
  GFX::Texture* texture = nullptr;
  /** serialize */
  static auto serialize(void* emitter, Core::EntityHandle const& handle)
      -> void;
  /** deserialize */
  static auto deserialize(void* compAoS, Core::EntityHandle const& handle)
      -> void;
};

SE_EXPORT auto to_string(LightComponent::LightType type) noexcept
    -> std::string;

/******************************************************************
 * Extension to GFX::Buffer
 *******************************************************************/

struct BufferView {
  GFX::Buffer* buffer = nullptr;
};

SE_EXPORT template <class T>
struct StructuredUniformBufferView : public BufferView {
  /** update use structure */
  auto setStructure(T const& x, uint32_t idx) noexcept -> void;
  /** get buffer binding */
  auto getBufferBinding(uint32_t idx) noexcept -> RHI::BufferBinding;
};

SE_EXPORT template <class T>
struct StructuredArrayUniformBufferView : public BufferView {
  /** update use structure */
  auto setStructure(T* x, uint32_t idx) noexcept -> void;
  /** get buffer binding */
  auto getBufferBinding(uint32_t idx) noexcept -> RHI::BufferBinding;
  /** array size */
  uint32_t size;
};

SE_EXPORT template <class T>
struct StructuredArrayMultiStorageBufferView : public BufferView {
  /** update use structure */
  auto setStructure(T* x, uint32_t idx, int subNum = -1) noexcept -> void;
  /** get buffer binding */
  auto getBufferBinding(uint32_t idx) noexcept -> RHI::BufferBinding;
  /** array size */
  uint32_t size = 0;
};

#pragma region IMPL_EXTENDED_BUFFER

template <class T>
auto StructuredUniformBufferView<T>::setStructure(T const& x,
                                                  uint32_t idx) noexcept
    -> void {
  std::future<bool> mapped =
      buffer->buffer->mapAsync(0, 0, sizeof(T) * MULTIFRAME_FLIGHTS_COUNT);
  if (mapped.get()) {
    void* data = buffer->buffer->getMappedRange(
        sizeof(T) * idx, sizeof(T) * MULTIFRAME_FLIGHTS_COUNT);
    memcpy(data, &x, sizeof(T));
    buffer->buffer->unmap();
  }
}

template <class T>
auto StructuredUniformBufferView<T>::getBufferBinding(uint32_t idx) noexcept
    -> RHI::BufferBinding {
  if (buffer == nullptr) return RHI::BufferBinding{nullptr, 0, 0};
  return RHI::BufferBinding{buffer->buffer.get(), idx * sizeof(T), sizeof(T)};
}

template <class T>
auto StructuredArrayUniformBufferView<T>::setStructure(T* x,
                                                       uint32_t idx) noexcept
    -> void {
  std::future<bool> mapped = buffer->buffer->mapAsync(
      0, 0, sizeof(T) * MULTIFRAME_FLIGHTS_COUNT * size);
  if (mapped.get()) {
    void* data = buffer->buffer->getMappedRange(
        sizeof(T) * idx * size, sizeof(T) * MULTIFRAME_FLIGHTS_COUNT * size);
    memcpy(data, x, sizeof(T) * size);
    buffer->buffer->unmap();
  }
}

template <class T>
auto StructuredArrayUniformBufferView<T>::getBufferBinding(
    uint32_t idx) noexcept -> RHI::BufferBinding {
  if (buffer == nullptr) return RHI::BufferBinding{nullptr, 0, 0};
  return RHI::BufferBinding{buffer->buffer.get(), idx * sizeof(T) * size,
                            sizeof(T) * size};
}

template <class T>
auto StructuredArrayMultiStorageBufferView<T>::setStructure(T* x, uint32_t idx,
                                                            int subNum) noexcept
    -> void {
  if (subNum == 0) return;
  if (subNum == -1) subNum = size;
  subNum = std::min(uint32_t(subNum), size);

  std::future<bool> mapped = buffer->buffer->mapAsync(
      0, 0, sizeof(T) * MULTIFRAME_FLIGHTS_COUNT * subNum);
  if (mapped.get()) {
    void* data = buffer->buffer->getMappedRange(
        sizeof(T) * idx * size, sizeof(T) * MULTIFRAME_FLIGHTS_COUNT * subNum);
    memcpy(data, x, sizeof(T) * subNum);
    buffer->buffer->unmap();
  }
}

template <class T>
auto StructuredArrayMultiStorageBufferView<T>::getBufferBinding(
    uint32_t idx) noexcept -> RHI::BufferBinding {
  if (buffer == nullptr) return RHI::BufferBinding{nullptr, 0, 0};
  return RHI::BufferBinding{buffer->buffer.get(), idx * sizeof(T) * size,
                            sizeof(T) * size};
}

#pragma endregion

struct GFXManager;

SE_EXPORT enum struct Ext {
  VideoClip,
};

SE_EXPORT struct Extension {
  virtual auto foo(uint32_t id, void* data) noexcept -> void* {
    return nullptr;
  }

 protected:
  friend GFXManager;
  virtual auto startUp() noexcept -> void = 0;
  virtual auto onUpdate() noexcept -> void{};
};

/** A singleton manager manages graphic components and resources. */
SE_EXPORT struct GFXManager : public Core::Manager {
  // online resource registers
  /** create / register online buffer resource */
  auto registerBufferResource(Core::GUID guid,
                              RHI::BufferDescriptor const& desc) noexcept
      -> void;
  auto registerBufferResource(Core::GUID guid, void* data, uint32_t size,
                              RHI::BufferUsagesFlags usage) noexcept -> void;
  template <class T>
  auto createStructuredUniformBuffer() noexcept
      -> StructuredUniformBufferView<T>;
  template <class T>
  auto createStructuredArrayUniformBuffer(uint32_t array_size) noexcept
      -> StructuredArrayUniformBufferView<T>;
  template <class T>
  auto createStructuredArrayMultiStorageBuffer(
      uint32_t array_size, uint32_t additionalUsage = 0) noexcept
      -> StructuredArrayMultiStorageBufferView<T>;
  /** create / register online texture resource */
  auto registerTextureResource(Core::GUID guid,
                               RHI::TextureDescriptor const& desc) noexcept
      -> void;
  auto registerTextureResource(
      Core::GUID guid, Image::Image<Image::COLOR_R8G8B8A8_UINT>* image) noexcept
      -> void;
  auto registerTextureResource(
      Core::GUID guid, Image::Image<Image::COLOR_R32G32B32A32_FLOAT>* image) noexcept
      -> void;
  auto registerTextureResource(Core::GUID guid,
                               Image::Texture_Host* image) noexcept -> void;
  auto registerTextureResource(char const* filepath) noexcept -> Core::GUID;
  auto registerTextureResourceCubemap(
      Core::GUID guid, std::array<char const*, 6> images) noexcept -> void;
  auto registerTextureResourceCubemap(
      Core::GUID guid,
      std::array<Image::Image<Image::COLOR_R8G8B8A8_UINT>*, 6> images) noexcept
      -> void;
  /** create / register online sampler resource */
  auto registerSamplerResource(Core::GUID guid,
                               RHI::SamplerDescriptor const& desc) noexcept
      -> void;
  /** create / register online shader resource */
  auto registerShaderModuleResource(
      Core::GUID guid, RHI::ShaderModuleDescriptor const& desc) noexcept
      -> void;
  auto registerShaderModuleResource(
      Core::GUID guid, char const* filepath,
      RHI::ShaderModuleDescriptor const& desc) noexcept -> void;
  auto registerShaderModuleResource(
      char const* filepath, RHI::ShaderModuleDescriptor const& desc) noexcept
      -> Core::GUID;
  // ofline resource request
  /** request offline texture resource */
  auto requestOfflineTextureResource(Core::ORID orid) noexcept -> Core::GUID;
  /** request offline mesh resource */
  auto requestOfflineMeshResource(Core::ORID orid) noexcept -> Core::GUID;
  /** request offline material resource */
  auto registerMaterialResource(char const* filepath) noexcept -> Core::GUID;
  auto requestOfflineMaterialResource(Core::ORID orid) noexcept -> Core::GUID;
  /** RHI layer */
  RHI::RHILayer* rhiLayer = nullptr;
  /** 
   * GlobalSamplerTable :: common samplers.
   * We suggest to use a global table to store all samplers,
   * to facilitate sampler resource to be reused over the system.
   * We can actually fetch the samplers by descriptors
   * as well as some simplified settings, by the function:
   * 
   * auto fetch() noexcept -> RHI::Sampler*
   */
  struct GlobalSamplerTable {
    std::unordered_map<uint64_t, RHI::Sampler*> hash_samplers;
    auto fetch(RHI::SamplerDescriptor const& desc) noexcept -> RHI::Sampler*;
    auto fetch(RHI::AddressMode address, RHI::FilterMode filter, RHI::MipmapFilterMode mipmap) noexcept -> RHI::Sampler*;
  } samplerTable;
  /** config singleton */
  GFXConfig config = {};
  /** start up the GFX manager */
  virtual auto startUp() noexcept -> void override;
  /** shut down the GFX manager */
  virtual auto shutDown() noexcept -> void override;
  /** update */
  auto onUpdate() noexcept -> void;
  /* get singleton */
  static inline auto get() noexcept -> GFXManager* { return singleton; }
  /** add extension */
  template <class T>
  auto addExt(Ext name) noexcept -> void {
    extensions[name] = std::make_unique<T>();
  }
  /** get extension */
  template <class T>
  auto getExt(Ext name) noexcept -> T* {
    return reinterpret_cast<T*>(extensions[name].get());
  }
  /** register material template */
  auto registerMaterialTemplate(uint32_t bsdf_id,
                                std::string const& name) noexcept
      -> MaterialTemplate&;
  /** register material template */
  auto getMaterialTemplate(uint32_t bsdf_id) noexcept -> MaterialTemplate*;

 private:
  /** singleton */
  static GFXManager* singleton;
  /** extensions */
  std::unordered_map<Ext, std::unique_ptr<Extension>> extensions;
  /** material template */
  std::unordered_map<uint32_t, MaterialTemplate> material_templates;
};

SE_EXPORT struct SBTsDescriptor {
  /** @indexing: By default, traceRayEXT always uses the ray generation shader
   * at index 0. Therefore we currently support single record slot for a ray
   * generation SBT. */
  struct RayGenerationSBT {
    /** A ray generation record only has a ray generation shader. */
    struct RayGenerationRecord {
      GFX::ShaderModule* rayGenShader = nullptr;
    };
    /** As defaultly 0 is chosen, we only provide one record slot*/
    RayGenerationRecord rgenRecord = {};
  } rgenSBT;
  /** @indexing: When a ray didn't intersect anything, traversal calls
   * the index missIndex miss shader, specified in traceRayEXT call. */
  struct MissSBT {
    /** A ray miss record only has a miss shader. */
    struct MissRecord {
      GFX::ShaderModule* missShader = nullptr;
    };
    /** There could be multiple miss shader to be selected from */
    std::vector<MissRecord> rmissRecords = {};
  } missSBT;
  /** @indexing: Traversal calls the corresponding shader from the hit record
   * with index. instanceShaderBindingTableRecordOffset (from TLAS) +
   * sbtRecordOffset (from traceRayEXT call)
   * + sbtRecordStride (from traceRayEXT call)* geometryIndex (from BLAS) */
  struct HitGroupSBT {
    /** A hit group record includes a closest hit shader, an optional any hit
     * shader, and an optional intersection shader (only for procedural hit
     * groups). */
    struct HitGroupRecord {
      GFX::ShaderModule* closetHitShader = nullptr;
      GFX::ShaderModule* anyHitShader = nullptr;
      GFX::ShaderModule* intersectionShader = nullptr;
    };
    /** There could be hit group shader to be selected from */
    std::vector<HitGroupRecord> hitGroupRecords = {};
  } hitGroupSBT;
  struct CallableSBT {
    /** A callable record includes only a callable shader. */
    struct CallableRecord {
      GFX::ShaderModule* callableShader = nullptr;
    };
    /** There could be hit group shader to be selected from */
    std::vector<CallableRecord> callableRecords = {};
  } callableSBT;

  operator RHI::SBTsDescriptor() const;
};


template <class T>
auto GFXManager::createStructuredUniformBuffer() noexcept
    -> StructuredUniformBufferView<T> {
  Core::GUID guid =
      Core::ResourceManager::get()->requestRuntimeGUID<GFX::Buffer>();
  RHI::BufferDescriptor desc = {
      .size = sizeof(T) * MULTIFRAME_FLIGHTS_COUNT,
      .usage = (uint32_t)RHI::BufferUsage::UNIFORM,
      .memoryProperties = uint32_t(RHI::MemoryProperty::HOST_VISIBLE_BIT) |
                          uint32_t(RHI::MemoryProperty::HOST_COHERENT_BIT)};
  registerBufferResource(guid, desc);
  StructuredUniformBufferView<T> view;
  view.buffer = Core::ResourceManager::get()->getResource<GFX::Buffer>(guid);
  return view;
}

template <class T>
auto GFXManager::createStructuredArrayUniformBuffer(
    uint32_t array_size) noexcept -> StructuredArrayUniformBufferView<T> {
  Core::GUID guid =
      Core::ResourceManager::get()->requestRuntimeGUID<GFX::Buffer>();
  RHI::BufferDescriptor desc = {
      .size = sizeof(T) * MULTIFRAME_FLIGHTS_COUNT * array_size,
      .usage = (uint32_t)RHI::BufferUsage::UNIFORM,
      .memoryProperties = uint32_t(RHI::MemoryProperty::HOST_VISIBLE_BIT) |
                          uint32_t(RHI::MemoryProperty::HOST_COHERENT_BIT)};
  registerBufferResource(guid, desc);
  StructuredArrayUniformBufferView<T> view;
  view.buffer = Core::ResourceManager::get()->getResource<GFX::Buffer>(guid);
  view.size = array_size;
  return view;
}

template <class T>
auto GFXManager::createStructuredArrayMultiStorageBuffer(
    uint32_t array_size, uint32_t additionalUsage) noexcept
    -> StructuredArrayMultiStorageBufferView<T> {
  Core::GUID guid =
      Core::ResourceManager::get()->requestRuntimeGUID<GFX::Buffer>();
  RHI::BufferDescriptor desc = {
      .size = sizeof(T) * MULTIFRAME_FLIGHTS_COUNT * array_size,
      .usage = (uint32_t)RHI::BufferUsage::STORAGE | additionalUsage,
      .memoryProperties = uint32_t(RHI::MemoryProperty::HOST_VISIBLE_BIT) |
                          uint32_t(RHI::MemoryProperty::HOST_COHERENT_BIT)};
  registerBufferResource(guid, desc);
  StructuredArrayMultiStorageBufferView<T> view;
  view.buffer = Core::ResourceManager::get()->getResource<GFX::Buffer>(guid);
  view.size = array_size;
  return view;
}
}

namespace SIByL::Core {
SE_EXPORT template <>
inline auto initComponentOnRegister<GFX::MeshRenderer>(Core::Entity& entity, GFX::MeshRenderer& component) noexcept
    -> bool {
  GFX::MeshReference* reference = entity.getComponent<GFX::MeshReference>();
  if (reference == nullptr) return false;
  component.materials.resize(reference->mesh->submeshes.size());
  return true;
}
}