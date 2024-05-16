#pragma once
#include <se.core.hpp>
#include <se.rhi.hpp>
#include <span>
#include <unordered_map>
#include <type_traits>
#include <tinygltf/tiny_gltf.h>

#define MULTIFRAME_FLIGHTS_COUNT 2

// ===========================================================================
// Base class definition

namespace se::gfx {
/**
 * Texture view index, describing a unique
 * definition of the texture view.
 */
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

// resource -------------------------------------------------

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

// loader -------------------------------------------------

struct ShaderHandle {
  ex::resource<se::gfx::ShaderModule> handle;
  auto get() noexcept -> se::gfx::ShaderModule* { return handle.handle().get(); }
};

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

struct SerializeData;

struct SIByL_API Buffer : public Resource {
  /** ctors & rval copies */
  Buffer() = default;
  Buffer(Buffer&& buffer) = default;
  Buffer(Buffer const& buffer) = delete;
  auto operator=(Buffer&& buffer) -> Buffer& = default;
  auto operator=(Buffer const& buffer) -> Buffer& = delete;
  /** the host and device buffer */
  std::unique_ptr<rhi::Buffer> buffer = nullptr;    size_t buffer_stamp = 0;
  std::unique_ptr<rhi::Buffer> previous = nullptr;  size_t previous_stamp = 0;
  std::vector<unsigned char> host;                  size_t host_stamp = 0;
  rhi::BufferUsages usages = 0;
  /** lazy update host buffer to device */
  auto hostToDevice() noexcept -> void;
  /** lazy update host device to host */
  auto deviceToHost() noexcept -> void;
  /** get the host buffer, if not exists fetch from newest device */
  auto getHost() noexcept -> std::vector<unsigned char>&;
  /** get the device buffer, if not exists fetch from newest host */
  auto getDevice() noexcept -> rhi::Buffer*;
  /** get the resource binding of the entire buffer */
  auto getBindingResource() noexcept -> rhi::BindingResource;
  template<class T>
  auto getHostAsStructuredArray() noexcept -> std::span<T> {
    return std::span<T>{reinterpret_cast<T*>(getHost().data()), getHost().size() / sizeof(T)};
  }
  /** get name of the resource */
  virtual auto getName() const noexcept -> char const* override;
};

struct SIByL_API BufferLoader {
  using result_type = std::shared_ptr<Buffer>;

  struct from_empty_tag {};
  struct from_gltf_tag {};
  struct from_host_tag {};
  struct from_desc_tag {};

  result_type operator()(from_empty_tag);
  result_type operator()(from_desc_tag, rhi::BufferDescriptor desc);
  result_type operator()(from_gltf_tag, tinygltf::Buffer const& buffer);
  result_type operator()(from_host_tag, se::buffer const& buffer, rhi::BufferUsages usages);
};

struct SIByL_API BufferHandle {
  ex::resource<Buffer> handle;
  auto get() noexcept -> Buffer* { return handle.handle().get(); }
  Buffer* operator->() { return get(); }
};

struct SIByL_API BufferView {
  struct SIByL_API View {
    BufferHandle buffer = {};
    size_t byteOffset = 0;
    size_t byteLength = 0;
    int target = 0;
  };
  struct SIByL_API Accessor {
    size_t byteOffset;
    int32_t componentType;
    int32_t count;
    std::string type;
  };
};

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
  /** Get the width of the texture */
  auto getWidth() noexcept -> uint32_t;
  /** Get the height of the texture */
  auto getHeight() noexcept -> uint32_t;
private:
  std::unordered_map<ViewIndex, std::unique_ptr<rhi::TextureView>> viewPool;
};

struct TextureHandle {
  ex::resource<Texture> handle;
  auto get() noexcept -> Texture* { return handle.handle().get(); }
  Texture* operator->() { return get(); }
};

struct SIByL_API TextureLoader {
  using result_type = std::shared_ptr<Texture>;

  struct from_desc_tag {};
  struct from_file_tag {};

  result_type operator()(from_desc_tag, rhi::TextureDescriptor const& desc);
  result_type operator()(from_file_tag, std::filesystem::path const& path);
};

// Base class definition
// ===========================================================================
// Material Resource

struct SIByL_API Material : public Resource {
  /* cast material to gltf material structure. */
  operator tinygltf::Material() const;
  virtual auto getName() const noexcept -> char const* override;


  vec3 baseOrDiffuseColor = vec3(1.f);
  float roughnessFactor = 1.f;
  float metallicFactor = 1.f;
  int32_t bxdf = 0;    // bxdf index
  bool doubleSided = true;
  std::string name;
};

struct MaterialHandle {
  ex::resource<Material> handle;
  auto get() noexcept -> Material* { return handle.handle().get(); }
  Material* operator->() { return get(); }
};

struct SIByL_API MaterialLoader {
  using result_type = std::shared_ptr<Material>;

  struct from_empty_tag {};
  struct from_gltf_tag {};

  result_type operator()(from_empty_tag);
};

struct SIByL_API IBxDF {
  virtual ~IBxDF() = default;
};

struct SIByL_API Mesh : public Resource {
  /* cast material to gltf mesh structure. */
  Mesh() = default;
  Mesh(Mesh&&) = default;
  operator tinygltf::Mesh() const;

  struct SIByL_API MeshPrimitive {
    MeshPrimitive() = default;
    MeshPrimitive(MeshPrimitive&&) = default;
    auto serialize(SerializeData& data) -> tinygltf::Primitive;
    size_t offset;
    size_t size;
    size_t baseVertex;
    size_t numVertex;
    MaterialHandle material;
    vec3 max, min;
    rhi::BLASDescriptor blasDesc;
    rhi::BLASDescriptor uvblasDesc;
    // blas for ray tracing the geometry
    std::unique_ptr<rhi::BLAS> prim_blas, back_blas;
    // blas for sampling the texcoordinates
    std::unique_ptr<rhi::BLAS> prim_uv_blas, back_uv_blas;
  };

  virtual auto getName() const noexcept -> char const* override;

  size_t timestamp = 0, prim_blas_timestamp = 0, back_blas_timestamp = 0;
  std::string name;
  size_t vertex_offset = 0;
  size_t index_offset = 0;
  std::vector<MeshPrimitive> primitives;
};

struct MeshHandle {
  ex::resource<Mesh> handle;
  auto get() noexcept -> Mesh* { return handle.handle().get(); }
  Mesh* operator->() { return get(); }
};

struct SIByL_API MeshLoader {
  using result_type = std::shared_ptr<Mesh>;

  struct from_empty_tag {};

  result_type operator()(from_empty_tag);
};

// Base class definition
// ===========================================================================
// Material Resource

struct SIByL_API Node {
  ex::entity entity;
  ex::registry* registry;
  template<class T, class ...Args>
  auto addComponent(Args&&... args) const -> T& { 
    return registry->emplace<T>(entity, std::forward<Args>(args)...); }
  template<class T>
  auto getComponent() const -> T* { return registry->try_get<T>(entity); }
  template<class T>
  auto removeComponent() const -> void { registry->remove<T>(entity); }
};

// Component: Node property
struct SIByL_API NodeProperty {
  std::string name;
  std::vector<Node> children;
};

// Component: Transform
struct SIByL_API Transform {
  vec3 translation = { 0.0f, 0.0f, 0.0f };
  vec3 scale = { 1.0f, 1.0f, 1.0f };
  Quaternion rotation = { 0.0f, 0.0f, 0.0f, 1.f };
  auto local() const noexcept -> se::mat4;
  auto forward() const noexcept -> se::vec3;
  mat4 global = {};
  float oddScaling = 1.f;
};

// Component: Transform
struct SIByL_API MeshRenderer {
  MeshHandle mesh;
  std::vector<rhi::BLASInstance> blasInstance;
  std::vector<rhi::BLASInstance> uvblasInstance;
};

// Component: Transform
struct SIByL_API Light {
  /* light type */
  enum struct LightType {
    DIRECTIONAL,
    POINT,
    SPOT,
    TRIANGLE,
    RECTANGLE,
    MESH_PRIMITIVE,
    ENVIRONMENT,
    VPL,
    MAX_ENUM,
  } type;
  /* light intensity to scale the light */
  se::vec3 intensity = se::vec3(1, 1, 1);
  /* packed data associated with light */
  se::vec4 packed_data_0;
  se::vec4 packed_data_1;
  bool isDirty = false;
  /** texture guid */
  TextureHandle texture;
};

struct SIByL_API Camera {
  enum struct ProjectType {
    PERSPECTIVE,
    ORTHOGONAL,
  };
  float aspectRatio = 1.f;
  float yfov = 45.f;
  float znear = 0.1f, zfar = 100.0f;
  int32_t width = 512, height = 512;
  float left_right = 0;
  float bottom_top = 0;
  ProjectType projectType = ProjectType::PERSPECTIVE;
  auto getViewMat() noexcept -> se::mat4;
  auto getProjectionMat() const noexcept -> se::mat4;
};

struct SIByL_API SerializeData {
  tinygltf::Model* model;
  std::unordered_map<ex::entity, int32_t> nodes;
  std::unordered_map<Material*, int32_t> materials;
  auto query_material(Material*) noexcept -> int32_t;
};

struct SIByL_API DeserializeData {
  tinygltf::Model* model;
  std::vector<Node> nodes;
};

struct SIByL_API SamplerHandle{
  ex::resource<rhi::Sampler> handle;
  auto get() noexcept -> rhi::Sampler* { return handle.handle().get(); }
  rhi::Sampler* operator->() { return get(); }
};

struct SIByL_API SamplerLoader {
  using result_type = std::shared_ptr<rhi::Sampler>;

  struct from_desc_tag {};
  struct from_mode_tag {};

  result_type operator()(from_desc_tag, rhi::SamplerDescriptor const& desc);
  result_type operator()(from_mode_tag, rhi::AddressMode address, rhi::FilterMode filter, rhi::MipmapFilterMode mipmap);
};

struct SIByL_API Scene : public Resource {
  Scene();
  ~Scene();
  auto createNode(std::string const& name = "nameless") noexcept -> Node;
  auto createNode(Node parent, std::string const& name = "nameless") noexcept -> Node;
  auto destroyNode(Node const& node) noexcept -> void;

  auto serialize() noexcept -> tinygltf::Model;
  auto serialize(std::string const& path) noexcept -> void;
  auto deserialize(tinygltf::Model& model) noexcept -> void;
  auto deserialize(std::string const& path) noexcept -> void;

  virtual auto getName() const noexcept -> char const* override;

  enum struct TexcoordKind { CopyCoord0, XAtlas, };
  auto createTexcoord(TexcoordKind kind) noexcept -> void;
  
  /** mesh / geometry draw call data */
  struct GeometryDrawData {
    uint32_t vertexOffset;
    uint32_t indexOffset;
    uint32_t materialID;
    uint32_t indexSize;
    float surfaceArea;
    uint32_t lightID;
    uint32_t primitiveType;
    float oddNegativeScaling;
    rhi::AffineTransformMatrix geometryTransform = {};
    rhi::AffineTransformMatrix geometryTransformInverse = {};
  };
  
  /** standard material data */
  struct MaterialData {
    se::vec3 baseOrDiffuseColor;
    uint32_t flags;
    se::vec3 specularColor;
    uint32_t bsdf_id = 1;
    se::vec3 emissiveColor;
    uint32_t domain = 1;

    float opacity;
    float roughness;
    float metalness;
    float normalTextureScale;

    float occlusionStrength;
    float alphaCutoff;
    float transmissionFactor;
    int baseOrDiffuseTextureIndex;

    int metalRoughOrSpecularTextureIndex;
    int emissiveTextureIndex;
    int normalTextureIndex;
    int occlusionTextureIndex;

    int transmissionTextureIndex;
    int padding1;
    int padding2;
    int padding3;
  };

  /** camera data */
  struct SIByL_API CameraData {
    se::mat4 viewMat;
    se::mat4 invViewMat;
    se::mat4 projMat;
    se::mat4 invProjMat;
    se::mat4 viewProjMat;
    se::mat4 invViewProj;
    se::mat4 viewProjMatNoJitter;
    se::mat4 projMatNoJitter;
    se::vec3 posW;
    float focalLength;
    se::vec3 prevPosW;
    float rectArea;
    se::vec3 up;
    float aspectRatio;
    se::vec3 target;
    float nearZ;
    se::vec3 cameraU;
    float farZ;
    se::vec3 cameraV;
    float jitterX;
    se::vec3 cameraW;
    float jitterY;
    float frameHeight;
    float frameWidth;
    float focalDistance;
    float apertureRadius;
    float shutterSpeed;
    float ISOSpeed;
    float _padding1;
    float _padding2;
    se::vec2 clipToWindowScale;
    se::vec2 clipToWindowBias;
    CameraData() = default;
    CameraData(Camera const& camera, Transform const& transform);
  };

  struct SIByL_API GPUSceneSetting {
    bool useTexcoordTLAS = true;
  } gpuSceneSetting;

  struct SIByL_API GPUScene {
    BufferHandle position_buffer;
    BufferHandle index_buffer;
    BufferHandle vertex_buffer;
    BufferHandle texcoord_buffer;
    BufferHandle material_buffer;
    BufferHandle geometry_buffer;
    BufferHandle light_buffer;
    BufferHandle camera_buffer;

    struct TLAS {
      rhi::TLASDescriptor desc = {};
      rhi::TLASDescriptor uvdesc = {};
      std::unique_ptr<rhi::TLAS> prim = nullptr;
      std::unique_ptr<rhi::TLAS> back = nullptr;
      std::unique_ptr<rhi::TLAS> uvprim = nullptr;
    } tlas;

    auto bindingResourcePosition() noexcept -> rhi::BindingResource;
    auto bindingResourceIndex() noexcept -> rhi::BindingResource;
    auto bindingResourceVertex() noexcept -> rhi::BindingResource;
    auto bindingResourceGeometry() noexcept -> rhi::BindingResource;
    auto bindingResourceCamera() noexcept -> rhi::BindingResource;
    auto bindingResourceTLAS() noexcept -> rhi::BindingResource;
    auto bindingResourceTLASPrev() noexcept -> rhi::BindingResource;
    auto bindingResourceUvTLAS() noexcept -> rhi::BindingResource;

    auto getPositionBuffer() noexcept -> BufferHandle;
    auto getIndexBuffer() noexcept -> BufferHandle;
  } gpuScene;

  auto getGPUScene() noexcept -> GPUScene*;
  virtual auto updateTransform() noexcept -> void;
  virtual auto updateGPUScene() noexcept -> void;
  //auto createGPUScene() noexcept -> void;
  //auto integrateGPUScene() noexcept -> void;

  auto useEditorCameraView(Transform* transfrom, Camera* camera) noexcept -> void;
  auto getEditorActiveCameraIndex() noexcept -> int;
  struct EditorInfo {
    Transform* viewport_transfrom = nullptr;
    Camera* viewport_camera = nullptr;
    int active_camera_index = 0;
  } editorInfo;

  std::string name;
  std::vector<Node> roots;
  ex::registry registry;
  bool isDirty = false;
  enum struct DirtyFlagBit {
    NodeDirty = 0 << 0,
    Camera = 1 << 0,
    Geometry = 1 << 1,
  };
  uint64_t dirtyFlags = 0;
};

struct SIByL_API SceneHandle{
  ex::resource<se::gfx::Scene> handle;
  auto get() noexcept -> se::gfx::Scene* { return handle.handle().get(); }
  Scene* operator->() { return get(); }
};

struct SIByL_API SceneLoader {
  using result_type = std::shared_ptr<Scene>;

  struct from_gltf_tag {};
  struct from_scratch_tag {};

  result_type operator()(from_gltf_tag, std::string const& path);
  result_type operator()(from_scratch_tag);
};

// Material Resource
// ===========================================================================
// GFX Context definition

auto SIByL_API captureImage(TextureHandle src) noexcept -> void;
auto SIByL_API captureImage(TextureHandle src, std::string path) noexcept -> void;

struct SIByL_API GFXContext {
  static auto initialize(rhi::Device* device) noexcept -> void;
  static auto createFlights(int maxFlightNum, rhi::SwapChain* swapchain = nullptr) -> void;
  static auto getFlights() -> rhi::MultiFrameFlights*;
  static auto getDevice() noexcept -> rhi::Device*;
  static auto finalize() noexcept -> void;
  static rhi::Device* device;
  static std::unique_ptr<rhi::MultiFrameFlights> flights;
  static ex::resource_cache<Buffer, BufferLoader> buffers;
  static ex::resource_cache<Texture, TextureLoader> textures;
  static ex::resource_cache<Mesh, MeshLoader> meshs;
  static ex::resource_cache<ShaderModule, ShaderModuleLoader> shaders;
  static ex::resource_cache<Material, MaterialLoader> materials;
  static ex::resource_cache<rhi::Sampler, SamplerLoader> samplers;
  static ex::resource_cache<Scene, SceneLoader> scenes;

  // load buffer resource
  // -------------------------------------------
  static auto load_buffer_empty() noexcept -> BufferHandle;
  static auto load_buffer_gltf(
    tinygltf::Buffer const& buffer
  ) noexcept -> BufferHandle;
  static auto load_buffer_host(
    se::buffer const& buffer,
    rhi::BufferUsages usages
  ) noexcept -> BufferHandle;
  static auto create_buffer_desc(
    rhi::BufferDescriptor const& desc
  ) noexcept -> BufferHandle;

  // load texture resource
  // -------------------------------------------
  static auto create_texture_desc(
    rhi::TextureDescriptor const& desc
  ) noexcept -> TextureHandle;
  static auto create_texture_file(
    std::string const& path
  ) noexcept -> TextureHandle;

  static auto create_sampler_desc(
    rhi::SamplerDescriptor const& desc
  ) noexcept -> SamplerHandle;
  static auto create_sampler_desc(
    rhi::AddressMode address, rhi::FilterMode filter, rhi::MipmapFilterMode mipmap
  ) noexcept -> SamplerHandle;

  // load mesh resource
  // -------------------------------------------
  static auto load_mesh_empty() noexcept -> MeshHandle;
  static auto load_material_empty() noexcept -> MaterialHandle;

  // load shader module resource
  // -------------------------------------------
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

  static auto load_scene_gltf(std::string const& path) noexcept -> SceneHandle;
  static auto create_scene(std::string const& name) noexcept -> SceneHandle;
};
}