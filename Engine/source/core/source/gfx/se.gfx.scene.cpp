#define DLIB_EXPORT
#include <se.gfx.hpp>
#undef DLIB_EXPORT
#include <stack>

namespace se::gfx {
static std::array<uint64_t, 24> primes = {3,  5,  7,  11, 13, 17, 19, 23,
    29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97};

inline auto hash_vertex(std::vector<float> const& v) noexcept -> std::uint64_t {
  std::uint64_t hash = reinterpret_cast<std::uint32_t const&>(v[0]);
  for (size_t n = 1; n < std::min(primes.size(), v.size()); ++n) {
    std::uint64_t floated = reinterpret_cast<std::uint32_t const&>(v[n]);
    hash += primes[n] * floated;
  }
  return hash;
}

template <typename T>
struct ArrayAdapter {
  /**
   * Construct an array adapter.
   * @param ptr Pointer to the start of the data, with offset applied
   * @param count Number of elements in the array
   * @param byte_stride Stride betweens elements in the array
   */
  ArrayAdapter(const unsigned char* ptr, size_t count, size_t byte_stride)
      : dataPtr(ptr), elemCount(count), stride(byte_stride) {}

  /// Returns a *copy* of a single element. Can't be used to modify it.
  T operator[](size_t pos) const {
    if (pos >= elemCount)
      throw std::out_of_range(
          "Tried to access beyond the last element of an array adapter with "
          "count " +
          std::to_string(elemCount) + " while getting elemnet number " +
          std::to_string(pos));
    return *(reinterpret_cast<const T*>(dataPtr + pos * stride));
  }
  /** Pointer to the bytes */
  unsigned const char* dataPtr;
  /** Number of elements in the array */
  const size_t elemCount;
  /** Stride in bytes between two elements */
  const size_t stride;
};
/** Mesh data layout */
struct MeshDataLayout {
  /** info types in vertex */
  enum struct VertexInfo {
    POSITION,
    NORMAL,
    TANGENT,
    UV,
    COLOR,
    CUSTOM,
  };
  /** an entry of the layout */
  struct Entry {
    rhi::VertexFormat format;
    VertexInfo info;
  };
  /* the list of vertex layout */
  std::vector<Entry> layout;
  /* index format */
  rhi::IndexFormat format;
};

MeshDataLayout defaultMeshDataLayout = {{
  {rhi::VertexFormat::FLOAT32X3,
   gfx::MeshDataLayout::VertexInfo::POSITION},
  {rhi::VertexFormat::FLOAT32X3, gfx::MeshDataLayout::VertexInfo::NORMAL},
  {rhi::VertexFormat::FLOAT32X3, gfx::MeshDataLayout::VertexInfo::TANGENT},
  {rhi::VertexFormat::FLOAT32X2, gfx::MeshDataLayout::VertexInfo::UV},
},
rhi::IndexFormat::UINT32_T};

/** A setting config to guide loading of mesh resource */
struct MeshLoaderConfig {
  MeshDataLayout layout = {};
  bool usePositionBuffer = true;
  bool residentOnHost = true;
  bool residentOnDevice = false;
  bool deduplication = true;
};

MeshLoaderConfig defaultMeshLoadConfig = { defaultMeshDataLayout, true, true, false, true};

Scene::Scene() {
  gpuScene.position_buffer = GFXContext::load_buffer_empty();
  gpuScene.index_buffer = GFXContext::load_buffer_empty();
  gpuScene.vertex_buffer = GFXContext::load_buffer_empty();
  gpuScene.texcoord_buffer = GFXContext::load_buffer_empty();
  gpuScene.material_buffer = GFXContext::load_buffer_empty();
  gpuScene.light_buffer = GFXContext::load_buffer_empty();
  gpuScene.geometry_buffer = GFXContext::load_buffer_empty();
  gpuScene.camera_buffer = GFXContext::load_buffer_empty();
  gpuScene.camera_buffer->host.resize(sizeof(Scene::CameraData));
  gpuScene.position_buffer->usages =
    (uint32_t)rhi::BufferUsageBit::VERTEX |
    (uint32_t)rhi::BufferUsageBit::SHADER_DEVICE_ADDRESS |
    (uint32_t)rhi::BufferUsageBit::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY |
    (uint32_t)rhi::BufferUsageBit::STORAGE;
  gpuScene.index_buffer->usages =
    (uint32_t)rhi::BufferUsageBit::INDEX |
    (uint32_t)rhi::BufferUsageBit::SHADER_DEVICE_ADDRESS |
    (uint32_t)rhi::BufferUsageBit::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY |
    (uint32_t)rhi::BufferUsageBit::STORAGE;
  gpuScene.vertex_buffer->usages =
    (uint32_t)rhi::BufferUsageBit::VERTEX |
    (uint32_t)rhi::BufferUsageBit::SHADER_DEVICE_ADDRESS |
      (uint32_t)rhi::BufferUsageBit::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY |
    (uint32_t)rhi::BufferUsageBit::STORAGE;
  gpuScene.texcoord_buffer->usages =
    (uint32_t)rhi::BufferUsageBit::VERTEX |
    (uint32_t)rhi::BufferUsageBit::SHADER_DEVICE_ADDRESS |
      (uint32_t)rhi::BufferUsageBit::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY |
    (uint32_t)rhi::BufferUsageBit::STORAGE;
  gpuScene.material_buffer->usages =
    (uint32_t)rhi::BufferUsageBit::SHADER_DEVICE_ADDRESS |
    (uint32_t)rhi::BufferUsageBit::STORAGE;
  gpuScene.light_buffer->usages =
    (uint32_t)rhi::BufferUsageBit::SHADER_DEVICE_ADDRESS |
    (uint32_t)rhi::BufferUsageBit::STORAGE;
  gpuScene.geometry_buffer->usages =
    (uint32_t)rhi::BufferUsageBit::SHADER_DEVICE_ADDRESS |
    (uint32_t)rhi::BufferUsageBit::STORAGE;
  gpuScene.camera_buffer->usages =
    (uint32_t)rhi::BufferUsageBit::SHADER_DEVICE_ADDRESS |
    (uint32_t)rhi::BufferUsageBit::STORAGE;
}

Scene::~Scene() {
  float a = 1.f;
}

auto Scene::createNode(std::string const& name) noexcept -> Node {
  auto entity = registry.create();
  auto node = Node{ entity, &registry };
  registry.emplace<NodeProperty>(entity, name);
  registry.emplace<Transform>(entity);

  roots.push_back(node);
  return node;
}

auto Scene::createNode(Node parent, std::string const& name) noexcept -> Node {
  auto entity = registry.create();
  auto node = Node{ entity, &registry };
  registry.emplace<NodeProperty>(entity, name);
  registry.get<NodeProperty>(parent.entity).children.push_back(node);
  return node;
}

auto Scene::destroyNode(Node const& node) noexcept -> void {
  registry.destroy(node.entity);
}

auto Scene::serialize(std::string const& path) noexcept -> void {
  tinygltf::Model model = serialize();
  tinygltf::TinyGLTF gltf;
  gltf.WriteGltfSceneToFile(&model, path,
    true, // embedImages
    true, // embedBuffers
    true, // pretty print
    false); // write binary
}

auto Scene::deserialize(tinygltf::Model& model) noexcept -> void {
  // register all the nodes first
  DeserializeData data;
  data.model = &model;
  data.nodes.resize(model.nodes.size());
  for (size_t i = 0; i < model.nodes.size(); ++i)
    data.nodes[i] = createNode(model.nodes[i].name);
  // find root nodes
  roots.clear(); // not all created nodes are roots
  auto& scene = model.scenes[0];
  for (auto root : scene.nodes) {
    roots.emplace_back(data.nodes[root]);
  }
}

auto Scene::deserialize(std::string const& path) noexcept -> void {
  tinygltf::TinyGLTF loader;
  tinygltf::Model model;
  std::string err;
  std::string warn;
  bool ret = loader.LoadASCIIFromFile(&model, &err, &warn, path);
  if (!warn.empty()) {
    root::print::error("Scene::deserialize warn::" + warn); return;
  } if (!err.empty()) {
    root::print::error("Scene::deserialize error::" + err); return;
  } if (!ret) {
    root::print::error("Failed to parse glTF"); return;
  }
  deserialize(model);
}

auto Scene::getName() const noexcept -> char const* {
  return name.c_str();
}

auto Scene::serialize() noexcept -> tinygltf::Model {
  tinygltf::Model m;
  SerializeData data;
  data.model = &m;
  tinygltf::Scene scene;
  // register all the nodes first
  auto node_view = registry.view<NodeProperty, Transform>();
  for (auto entity : node_view)
    data.nodes.emplace(entity, data.nodes.size());
  for (auto entity : node_view) {
    int const node_id = data.nodes[entity];
    auto [prop, transform] = node_view.get<NodeProperty, Transform>(entity);
    m.nodes.emplace_back(tinygltf::Node{});
    m.nodes[node_id].name = prop.name;
    for (auto& child : prop.children)
      m.nodes[node_id].children.push_back(data.nodes[child.entity]);
    m.nodes[node_id].translation = { transform.translation.x,transform.translation.y, transform.translation.z };
    m.nodes[node_id].scale = { transform.scale.x,transform.scale.y, transform.scale.z };
    m.nodes[node_id].rotation = { transform.rotation.x,transform.rotation.y, transform.rotation.z, transform.rotation.w };
  }
  // declare root nodes
  for (auto& node : roots) {
    int node_id = data.nodes[node.entity];
    scene.nodes.push_back(node_id);
  }
  m.buffers.resize(4);
  // store position buffer
  m.buffers[0].data = gpuScene.position_buffer.get()->host;
  m.buffers[1].data = gpuScene.index_buffer.get()->host;
  m.buffers[2].data = gpuScene.vertex_buffer.get()->host;
  m.buffers[3].data = gpuScene.texcoord_buffer.get()->getHost();

  auto add_view_accessor = [&](
    tinygltf::BufferView bufferView,
    tinygltf::Accessor accessor
  ) -> int {
    int view_id = m.bufferViews.size();
    m.bufferViews.push_back(bufferView);
    int accessor_id = m.accessors.size();
    accessor.bufferView = view_id;
    m.accessors.push_back(accessor);
    return accessor_id;
  };
  auto add_accessor = [&](tinygltf::Accessor accessor) -> int {
    int accessor_id = m.accessors.size();
    m.accessors.push_back(accessor);
    return accessor_id;
  };

  std::unordered_map<Material*, int32_t> material_map;
  auto add_material = [&](Material* material) -> int {
    auto iter = material_map.find(material);
    if (iter != material_map.end()) {
      return iter->second;
    }
    int material_id = m.materials.size();
    tinygltf::Material gltf_material;
    gltf_material.pbrMetallicRoughness.baseColorFactor = {
        material->baseOrDiffuseColor.r, material->baseOrDiffuseColor.g,
        material->baseOrDiffuseColor.b, 1. };
    gltf_material.pbrMetallicRoughness.roughnessFactor = double(material->roughnessFactor);
    gltf_material.pbrMetallicRoughness.metallicFactor = double(material->metallicFactor);
    m.materials.push_back(gltf_material);
    material_map[material] = material_id;
    return material_id;
  };

  // store index buffer
  auto mesh_view = registry.view<MeshRenderer>();
  for (auto entity : mesh_view) {
    int const node_id = data.nodes[entity];
    auto& se_mesh = mesh_view.get<MeshRenderer>(entity);
    int const mesh_id = m.meshes.size();
    m.meshes.emplace_back(tinygltf::Mesh{}); auto& gltf_mesh = m.meshes.back();
    for (auto& primitive : se_mesh.mesh.get()->primitives) {
      tinygltf::Primitive gltf_primitive;
      { // position buffer
        tinygltf::BufferView bufferView;
        bufferView.buffer = 0;
        bufferView.byteOffset = primitive.baseVertex * sizeof(float) * 3 + se_mesh.mesh.get()->vertex_offset;
        bufferView.byteLength = primitive.numVertex * sizeof(float) * 3;
        bufferView.target = TINYGLTF_TARGET_ARRAY_BUFFER;
        tinygltf::Accessor accessor;
        accessor.byteOffset = 0;
        accessor.componentType = TINYGLTF_COMPONENT_TYPE_FLOAT;
        accessor.count = primitive.numVertex;
        accessor.type = TINYGLTF_TYPE_VEC3;
        accessor.maxValues = { primitive.max.x, primitive.max.y, primitive.max.z };
        accessor.minValues = { primitive.min.x, primitive.min.y, primitive.min.z };
        gltf_primitive.attributes["POSITION"] = add_view_accessor(bufferView, accessor);
      }
      { // index buffer
        tinygltf::BufferView bufferView;
        bufferView.buffer = 1;
        bufferView.byteOffset = primitive.offset * sizeof(uint32_t) + se_mesh.mesh.get()->index_offset;
        bufferView.byteLength = primitive.size * sizeof(uint32_t);
        bufferView.target = TINYGLTF_TARGET_ELEMENT_ARRAY_BUFFER;
        tinygltf::Accessor accessor;
        accessor.byteOffset = 0;
        accessor.componentType = TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT;
        accessor.count = primitive.size;
        accessor.type = TINYGLTF_TYPE_SCALAR;
        gltf_primitive.indices = add_view_accessor(bufferView, accessor);
      }
      { // vertex buffer
        tinygltf::BufferView bufferView;
        bufferView.buffer = 2;
        size_t vertexByteOffset = primitive.baseVertex * sizeof(float) * 3 + se_mesh.mesh.get()->vertex_offset;
        bufferView.byteOffset = vertexByteOffset / 3 * 8;
        bufferView.byteLength = primitive.numVertex * sizeof(float) * 8;
        bufferView.byteStride = 8 * sizeof(float);
        bufferView.target = TINYGLTF_TARGET_ARRAY_BUFFER;
        int view_id = m.bufferViews.size();
        m.bufferViews.push_back(bufferView);
        // normal
        tinygltf::Accessor accessor;
        accessor.bufferView = view_id;
        accessor.byteOffset = 0;
        accessor.componentType = TINYGLTF_COMPONENT_TYPE_FLOAT;
        accessor.count = primitive.numVertex;
        accessor.type = TINYGLTF_TYPE_VEC3;
        gltf_primitive.attributes["NORMAL"] = add_accessor(accessor);
        // tangent
        accessor.byteOffset = sizeof(float) * 3;
        gltf_primitive.attributes["TANGENT"] = add_accessor(accessor);
        // coord
        accessor.byteOffset = sizeof(float) * 6;
        accessor.type = TINYGLTF_TYPE_VEC2;
        gltf_primitive.attributes["TEXCOORD_0"] = add_accessor(accessor);
      }
      // texcoord1 buffer
      if(m.buffers[3].data.size() > 0) {
        tinygltf::BufferView bufferView;
        bufferView.buffer = 3;
        bufferView.byteOffset = (primitive.baseVertex * sizeof(float) * 3 + se_mesh.mesh.get()->vertex_offset) * 2 / 3;
        bufferView.byteLength = primitive.numVertex * sizeof(float) * 2;
        bufferView.target = TINYGLTF_TARGET_ARRAY_BUFFER;
        tinygltf::Accessor accessor;
        accessor.byteOffset = 0;
        accessor.componentType = TINYGLTF_COMPONENT_TYPE_FLOAT;
        accessor.count = primitive.numVertex;
        accessor.type = TINYGLTF_TYPE_VEC2;
        gltf_primitive.attributes["TEXCOORD_1"] = add_view_accessor(bufferView, accessor);
      }
      gltf_primitive.mode = TINYGLTF_MODE_TRIANGLES;
      gltf_primitive.material = add_material(primitive.material.get());
      gltf_mesh.primitives.emplace_back(gltf_primitive);
    }
    m.nodes[node_id].mesh = mesh_id;
  }

  auto camera_view = registry.view<Camera>();
  for (auto entity : camera_view) {
    int const node_id = data.nodes[entity];
    auto& se_camera = camera_view.get<Camera>(entity);
    tinygltf::Camera gltf_camera = {};
    if (se_camera.projectType == Camera::ProjectType::PERSPECTIVE) {
      gltf_camera.type = "perspective";
      gltf_camera.perspective.aspectRatio = se_camera.aspectRatio;
      gltf_camera.perspective.yfov = se_camera.yfov;
      gltf_camera.perspective.znear = se_camera.znear;
      gltf_camera.perspective.zfar = se_camera.zfar;
    }
    int const camera_id = m.cameras.size();
    m.cameras.emplace_back(gltf_camera);
    m.nodes[node_id].camera = camera_id;
  }

  m.scenes.emplace_back(scene);
  return m;
}

auto Scene::updateTransform() noexcept -> void {
  std::stack<float> is_odd_scaling;
  std::stack<se::mat4> global_transfrom;
  is_odd_scaling.push(1.f);
  global_transfrom.push(se::mat4{});
  auto node_view = registry.view<NodeProperty, Transform>();

  std::function<void(ex::entity)> fn_process_per_node = [&](ex::entity entity) {
    auto [prop, transform] = node_view.get<NodeProperty, Transform>(entity);
    se::mat4 objectMat = global_transfrom.top();
    float oddScaling = is_odd_scaling.top();
    objectMat = objectMat * transform.local();
    oddScaling *= transform.scale.x * transform.scale.y * transform.scale.z;
    // update transform information
    if (oddScaling != 0) oddScaling / std::abs(oddScaling);
    transform.oddScaling = oddScaling >= 0;
    transform.global = objectMat;
    // process children nodes
    global_transfrom.push(objectMat);
    is_odd_scaling.push(oddScaling);
    for (auto child : prop.children)
      fn_process_per_node(child.entity);
    is_odd_scaling.pop();
    global_transfrom.pop();
  };

  for (auto& root : roots) {
    fn_process_per_node(root.entity);
  }
}

Scene::CameraData::CameraData(Camera const& camera, Transform const& transform) {
  nearZ = camera.znear;
  farZ = camera.zfar;
  posW = transform.translation;
  target = transform.translation + transform.forward();
  viewMat = se::transpose(se::lookAt(posW, target, se::vec3(0, 1, 0)).m);
  invViewMat = se::inverse(viewMat);
  projMat = se::transpose(camera.getProjectionMat());
  invProjMat = se::inverse(projMat);
  viewProjMat = viewMat * projMat;
  invViewProj = se::inverse(viewProjMat);
  // Ray tracing related vectors
  focalDistance = 1;
  aspectRatio = camera.aspectRatio;
  up = se::vec3(0, 1, 0);
  cameraW = se::normalize(target - posW) * focalDistance;
  cameraU = se::normalize(se::cross(cameraW, up));
  cameraV = se::normalize(se::cross(cameraU, cameraW));
  const float ulen = focalDistance * std::tan(se::radians(camera.yfov) * 0.5f) * aspectRatio;
  cameraU *= ulen;
  const float vlen = focalDistance * std::tan(se::radians(camera.yfov) * 0.5f);
  cameraV *= vlen;
  jitterX = 0;
  jitterY = 0;
  clipToWindowScale = se::vec2(0.5f * camera.width, -0.5f * camera.height);
  clipToWindowBias = se::vec2(0.f) + se::vec2(camera.width, camera.height) * 0.5f;
  rectArea = 4 * ulen * vlen / (focalDistance * focalDistance);
}

auto Scene::useEditorCameraView(Transform* transfrom, Camera* camera) noexcept -> void {
  editorInfo.viewport_transfrom = transfrom;
  editorInfo.viewport_camera = camera;
}

auto Scene::getEditorActiveCameraIndex() noexcept -> int {
  return editorInfo.active_camera_index;
}

auto Scene::updateGPUScene() noexcept -> void {
  // update camera information
  size_t camera_index = 0;
  if ((dirtyFlags & (uint64_t)DirtyFlagBit::Camera) != 0) {
    auto invalid_camera = [&](Transform& transform, Camera& camera) {
      CameraData camData(camera, transform);
      if (gpuScene.camera_buffer->host.size() < (camera_index + 1) * sizeof(CameraData)) {
        gpuScene.camera_buffer->host.resize((camera_index + 1) * sizeof(CameraData));
      }
      memcpy(&gpuScene.camera_buffer->host[(camera_index++) * sizeof(CameraData)], &camData, sizeof(CameraData));
      gpuScene.camera_buffer->host_stamp++;
    };
    auto view_transform_camera = registry.view<Transform, Camera>();
    for (auto entity : view_transform_camera) {
      auto [transform, camera] = view_transform_camera.get<Transform, Camera>(entity);
      invalid_camera(transform, camera);
    }
    if (editorInfo.viewport_transfrom != nullptr && 
        editorInfo.viewport_camera != nullptr) {
      invalid_camera(*editorInfo.viewport_transfrom, *editorInfo.viewport_camera);
    }
  }

  // update geometry information
  auto node_mesh_view = registry.view<Transform, MeshRenderer>();
  if ((dirtyFlags & (uint64_t)DirtyFlagBit::Geometry) != 0) {
    size_t geometry_index = 0;
    for (auto entity : node_mesh_view) {
      auto [transform, mesh_renderer] = node_mesh_view.get<Transform, MeshRenderer>(entity);
      for (auto& primitive : mesh_renderer.mesh->primitives) {
        GeometryDrawData geometry;
        geometry.vertexOffset = primitive.baseVertex + mesh_renderer.mesh->vertex_offset / (sizeof(float) * 3);
        geometry.indexOffset = primitive.offset + mesh_renderer.mesh->index_offset / sizeof(uint32_t);
        geometry.indexSize = primitive.size;
        geometry.geometryTransform = transform.global;
        geometry.geometryTransformInverse = se::inverse(transform.global);
        geometry.oddNegativeScaling = transform.oddScaling;
        geometry.materialID = 0;
        geometry.primitiveType = 0;
        geometry.lightID = 0;
        if (gpuScene.geometry_buffer->host.size() < (geometry_index + 1) * sizeof(GeometryDrawData)) {
          gpuScene.geometry_buffer->host.resize((geometry_index + 1) * sizeof(GeometryDrawData));
        }
        memcpy(&gpuScene.geometry_buffer->host[geometry_index * sizeof(GeometryDrawData)], &geometry, sizeof(GeometryDrawData));
        geometry_index++;
      }
    }
  }

  gpuScene.position_buffer->hostToDevice();
  gpuScene.index_buffer->hostToDevice();
  gpuScene.vertex_buffer->hostToDevice();
  gpuScene.geometry_buffer->hostToDevice();
  gpuScene.camera_buffer->hostToDevice();
  gpuScene.texcoord_buffer->hostToDevice();
  gpuScene.tlas.desc = {};

  if ((dirtyFlags & (uint64_t)DirtyFlagBit::Geometry) != 0) {
    for (auto entity : node_mesh_view) {
      auto [transform, mesh_renderer] = node_mesh_view.get<Transform, MeshRenderer>(entity);
      mesh_renderer.blasInstance.resize(mesh_renderer.mesh->primitives.size());
      mesh_renderer.uvblasInstance.resize(mesh_renderer.mesh->primitives.size());
      size_t primitive_index = 0;
      for (auto& primitive : mesh_renderer.mesh->primitives) {
        primitive.back_blas = std::move(primitive.prim_blas);
        primitive.blasDesc.allowCompaction = true;
        primitive.blasDesc.triangleGeometries.push_back(rhi::BLASTriangleGeometry{
          gpuScene.position_buffer->buffer.get(),
          gpuScene.index_buffer->buffer.get(),
          rhi::IndexFormat::UINT32_T,
          uint32_t(primitive.numVertex),
          uint32_t(primitive.baseVertex + mesh_renderer.mesh->vertex_offset / (sizeof(float) * 3)),
          uint32_t(primitive.size / 3),
          uint32_t(primitive.offset * sizeof(uint32_t) + mesh_renderer.mesh->index_offset),
          rhi::AffineTransformMatrix{},
          (uint32_t)rhi::BLASGeometryFlagBits::NO_DUPLICATE_ANY_HIT_INVOCATION,
          0 });
        primitive.prim_blas = GFXContext::device->createBLAS(primitive.blasDesc);
        // update the instance of the mesh resource
        rhi::BLASInstance& instance = mesh_renderer.blasInstance[primitive_index++];
        instance.blas = primitive.prim_blas.get();
        instance.transform = transform.global;
        instance.instanceCustomIndex = 0; // geometry_start
        instance.instanceShaderBindingTableRecordOffset = 0;
        gpuScene.tlas.desc.instances.push_back(instance);
    
        if (gpuSceneSetting.useTexcoordTLAS) {
          // create the blas for sampling from the texture coordinates.
          primitive.back_uv_blas = std::move(primitive.prim_uv_blas);
          primitive.uvblasDesc.allowCompaction = true;
          primitive.uvblasDesc.triangleGeometries.push_back(rhi::BLASTriangleGeometry{
            gpuScene.texcoord_buffer->buffer.get(),
            gpuScene.index_buffer->buffer.get(),
            rhi::IndexFormat::UINT32_T,
            uint32_t(primitive.numVertex),
            uint32_t(primitive.baseVertex + mesh_renderer.mesh->vertex_offset / (sizeof(float) * 3)),
            uint32_t(primitive.size / 3),
            uint32_t(primitive.offset * sizeof(uint32_t) + mesh_renderer.mesh->index_offset),
            rhi::AffineTransformMatrix{},
            (uint32_t)rhi::BLASGeometryFlagBits::NO_DUPLICATE_ANY_HIT_INVOCATION,
            0, 2 * sizeof(float), 0, rhi::BLASTriangleGeometry::VertexFormat::RG32 });
          primitive.prim_uv_blas = GFXContext::device->createBLAS(primitive.uvblasDesc);
          // update the instance of the mesh resource
          rhi::BLASInstance& uvinstance = mesh_renderer.uvblasInstance[primitive_index - 1];
          uvinstance.blas = primitive.prim_uv_blas.get();
          uvinstance.transform = {};
          uvinstance.instanceCustomIndex = 0; // geometry_start
          uvinstance.instanceShaderBindingTableRecordOffset = 0;
          gpuScene.tlas.uvdesc.instances.push_back(uvinstance);
        }
      }
    }
    
    gpuScene.tlas.prim = GFXContext::device->createTLAS(gpuScene.tlas.desc);
    gpuScene.tlas.uvprim = GFXContext::device->createTLAS(gpuScene.tlas.uvdesc);
  }
  
  // set the dirty flag to 0
  dirtyFlags = 0;
}

auto Scene::getGPUScene() noexcept -> GPUScene* { return &gpuScene; }

auto Scene::createTexcoord(TexcoordKind kind) noexcept -> void {
  if (kind == TexcoordKind::CopyCoord0) {
    std::vector<unsigned char>& host = gpuScene.vertex_buffer->getHost();
    std::span<float> span(reinterpret_cast<float*>(host.data()), host.size() / sizeof(float));
    std::vector<unsigned char>& texcoord = gpuScene.texcoord_buffer->getHost();
    texcoord.resize(span.size() / 8 * 2 * sizeof(float));
    std::span<float> target_span(reinterpret_cast<float*>(texcoord.data()), texcoord.size() / sizeof(float));
    for (size_t i = 0; i * 8 < span.size(); ++i) {
      target_span[i * 2 + 0] = span[i * 8 + 6];
      target_span[i * 2 + 1] = span[i * 8 + 7];
    }
 }
}

auto Scene::GPUScene::bindingResourcePosition() noexcept -> rhi::BindingResource {
  return rhi::BindingResource{ {position_buffer->buffer.get(), 0, position_buffer->buffer->size()} };
}

auto Scene::GPUScene::bindingResourceIndex() noexcept -> rhi::BindingResource {
  return rhi::BindingResource{ {index_buffer->buffer.get(), 0, index_buffer->buffer->size()} };
}

auto Scene::GPUScene::bindingResourceVertex() noexcept -> rhi::BindingResource {
  return rhi::BindingResource{ {vertex_buffer->buffer.get(), 0, vertex_buffer->buffer->size()} };
}

auto Scene::GPUScene::bindingResourceGeometry() noexcept -> rhi::BindingResource {
  return rhi::BindingResource{ {geometry_buffer->buffer.get(), 0, geometry_buffer->buffer->size()} };
}

auto Scene::GPUScene::bindingResourceCamera() noexcept -> rhi::BindingResource {
  return rhi::BindingResource{ {camera_buffer->buffer.get(), 0, camera_buffer->buffer->size()} };
}

auto Scene::GPUScene::bindingResourceTLAS() noexcept -> rhi::BindingResource {
  return rhi::BindingResource{ tlas.prim.get() };
}

auto Scene::GPUScene::bindingResourceTLASPrev() noexcept -> rhi::BindingResource {
  return rhi::BindingResource{ tlas.back.get() };
}

auto Scene::GPUScene::bindingResourceUvTLAS() noexcept -> rhi::BindingResource {
  return rhi::BindingResource{ tlas.uvprim.get() };
}

struct glTFLoaderEnv {
  std::string directory;
  std::unordered_map<tinygltf::Texture const*, TextureHandle> textures;
  std::unordered_map<tinygltf::Material const*, MaterialHandle> materials;
  std::vector<std::vector<se::mat4>> skinning_matrices;
  std::vector<std::vector<int>> skinning_indices;
  //std::vector<std::vector<GFX::AnimationComponent::AnimationSampler>> anim_samplers;
  //std::vector<std::unordered_map<uint32_t, std::vector<GFX::AnimationComponent::AnimationChannel>>> anim_channels;
  std::unordered_map<int, Node> node2go;
};

auto loadGLTFMaterial(tinygltf::Material const* glmaterial, tinygltf::Model const* model,
    glTFLoaderEnv& env, gfx::Scene& gfxscene, MeshLoaderConfig meshConfig = {}) noexcept
    -> MaterialHandle {
  if (env.materials.find(glmaterial) != env.materials.end()) {
    return env.materials[glmaterial];
  }

  MaterialHandle mat = GFXContext::load_material_empty();
  std::string name = glmaterial->name;

  auto to_sampler = [&](int sampler_idx) {
    tinygltf::Sampler const& sampler = model->samplers[sampler_idx];
    rhi::SamplerDescriptor desc;
    // Min and Mipmap filter
    if (sampler.minFilter == TINYGLTF_TEXTURE_FILTER_NEAREST) {
      desc.minFilter = rhi::FilterMode::NEAREST;
    } else if (sampler.minFilter == TINYGLTF_TEXTURE_FILTER_LINEAR) {
      desc.minFilter = rhi::FilterMode::LINEAR;
    } else if (sampler.minFilter == TINYGLTF_TEXTURE_FILTER_NEAREST_MIPMAP_NEAREST) {
      desc.minFilter = rhi::FilterMode::NEAREST;
      desc.mipmapFilter = rhi::MipmapFilterMode::NEAREST;
    } else if (sampler.minFilter == TINYGLTF_TEXTURE_FILTER_LINEAR_MIPMAP_NEAREST) {
      desc.minFilter = rhi::FilterMode::LINEAR;
      desc.mipmapFilter = rhi::MipmapFilterMode::NEAREST;
    } else if (sampler.minFilter == TINYGLTF_TEXTURE_FILTER_NEAREST_MIPMAP_LINEAR) {
      desc.minFilter = rhi::FilterMode::NEAREST;
      desc.mipmapFilter = rhi::MipmapFilterMode::LINEAR;
    } else if (sampler.minFilter == TINYGLTF_TEXTURE_FILTER_LINEAR_MIPMAP_LINEAR) {
      desc.minFilter = rhi::FilterMode::LINEAR;
      desc.mipmapFilter = rhi::MipmapFilterMode::LINEAR;
    }
    // Mag filter
    if (sampler.magFilter == TINYGLTF_TEXTURE_FILTER_NEAREST) {
      desc.magFilter = rhi::FilterMode::NEAREST;
    } else if (sampler.magFilter == TINYGLTF_TEXTURE_FILTER_LINEAR) {
      desc.magFilter = rhi::FilterMode::LINEAR;
    }
    // WarpS
    if (sampler.wrapS == TINYGLTF_TEXTURE_WRAP_REPEAT) {
      desc.addressModeU = rhi::AddressMode::REPEAT;
    } else if (sampler.wrapS == TINYGLTF_TEXTURE_WRAP_CLAMP_TO_EDGE) {
      desc.addressModeU = rhi::AddressMode::CLAMP_TO_EDGE;
    } else if (sampler.wrapS == TINYGLTF_TEXTURE_WRAP_MIRRORED_REPEAT) {
      desc.addressModeU = rhi::AddressMode::MIRROR_REPEAT;
    }
    // WarpT
    if (sampler.wrapT == TINYGLTF_TEXTURE_WRAP_REPEAT) {
      desc.addressModeV = rhi::AddressMode::REPEAT;
    } else if (sampler.wrapT == TINYGLTF_TEXTURE_WRAP_CLAMP_TO_EDGE) {
      desc.addressModeV = rhi::AddressMode::CLAMP_TO_EDGE;
    } else if (sampler.wrapT == TINYGLTF_TEXTURE_WRAP_MIRRORED_REPEAT) {
      desc.addressModeV = rhi::AddressMode::MIRROR_REPEAT;
    }
    return desc;
  };

  // load diffuse information
  { // load diffuse color
    if (glmaterial->pbrMetallicRoughness.baseColorFactor.size() > 0) {
      mat->baseOrDiffuseColor = se::vec3{
        (float)glmaterial->pbrMetallicRoughness.baseColorFactor[0],
        (float)glmaterial->pbrMetallicRoughness.baseColorFactor[1],
        (float)glmaterial->pbrMetallicRoughness.baseColorFactor[2],
      };
      mat->roughnessFactor = (float)glmaterial->pbrMetallicRoughness.roughnessFactor;
      mat->metallicFactor = (float)glmaterial->pbrMetallicRoughness.metallicFactor;
    }
  }
  //{ // load diffuse texture
  //  if (glmaterial->pbrMetallicRoughness.baseColorTexture.index != -1) {
  //    tinygltf::Texture const& texture = model->textures[glmaterial->pbrMetallicRoughness.baseColorTexture.index];
  //    Core::GUID texBasecolor = loadGLTFMaterialTextures(&texture, model, env, gfxscene, meshConfig);
  //    gfxmat.textures["base_color"] = GFX::Material::TextureEntry{texBasecolor, 0, to_sampler(texture.sampler)};
  //  }
  //}

  return mat;
}

static inline auto loadGLTFMesh(tinygltf::Mesh const& gltfmesh,
  Node& gfxNode, Scene& scene, int node_id, tinygltf::Model const* model,
  glTFLoaderEnv& env) noexcept -> MeshHandle {
#define INDEX_TYPE uint32_t
  // Load meshes into Runtime resource managers.
  rhi::Device* device = GFXContext::device;
  std::vector<INDEX_TYPE> indexBuffer_uint = {};
  std::vector<float> vertexBuffer = {};
  std::vector<float> PositionBuffer = {};
  std::vector<uint64_t> JointIndexBuffer = {};
  std::vector<float> JointweightsBuffer = {};
  // Create GFX mesh, and add it to resource manager
  size_t submesh_index_offset = 0;
  size_t submesh_vertex_offset = 0;
  MeshHandle mesh = GFXContext::load_mesh_empty();
  // For each primitive
  for (auto const& meshPrimitive : gltfmesh.primitives) {
    std::vector<INDEX_TYPE> indexArray_uint = {};
    std::vector<float> vertexBuffer_positionOnly = {};
    std::vector<float> vertexBuffer_normalOnly = {};
    std::vector<float> vertexBuffer_uvOnly = {};
    std::vector<float> vertexBuffer_tangentOnly = {};
    std::vector<uint64_t> vertexBuffer_joints = {};
    std::vector<float> vertexBuffer_weights = {};
    auto const& indicesAccessor = model->accessors[meshPrimitive.indices];
    auto const& bufferView = model->bufferViews[indicesAccessor.bufferView];
    auto const& buffer = model->buffers[bufferView.buffer];
    auto const dataAddress =
        buffer.data.data() + bufferView.byteOffset + indicesAccessor.byteOffset;
    auto const byteStride = indicesAccessor.ByteStride(bufferView);
    uint64_t const count = indicesAccessor.count;
    se::vec3 positionMax, positionMin;
    switch (indicesAccessor.componentType) {
      case TINYGLTF_COMPONENT_TYPE_BYTE: {
        ArrayAdapter<char> originIndexArray(dataAddress, count, byteStride);
        for (size_t i = 0; i < count; ++i)
          indexArray_uint.emplace_back(INDEX_TYPE(originIndexArray[i]));
      } break;
      case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE: {
        ArrayAdapter<unsigned char> originIndexArray(dataAddress, count,
                                                     byteStride);
        for (size_t i = 0; i < count; ++i)
          indexArray_uint.emplace_back(INDEX_TYPE(originIndexArray[i]));
      } break;
      case TINYGLTF_COMPONENT_TYPE_SHORT: {
        ArrayAdapter<short> originIndexArray(dataAddress, count, byteStride);
        for (size_t i = 0; i < count; ++i)
          indexArray_uint.emplace_back(INDEX_TYPE(originIndexArray[i]));
      } break;
      case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: {
        ArrayAdapter<unsigned short> originIndexArray(dataAddress, count,
                                                      byteStride);
        for (size_t i = 0; i < count; ++i)
          indexArray_uint.emplace_back(INDEX_TYPE(originIndexArray[i]));
      } break;
      case TINYGLTF_COMPONENT_TYPE_INT: {
        ArrayAdapter<int> originIndexArray(dataAddress, count, byteStride);
        for (size_t i = 0; i < count; ++i)
          indexArray_uint.emplace_back(INDEX_TYPE(originIndexArray[i]));
      } break;
      case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT: {
        ArrayAdapter<unsigned int> originIndexArray(dataAddress, count,
                                                    byteStride);
        for (size_t i = 0; i < count; ++i)
          indexArray_uint.emplace_back(INDEX_TYPE(originIndexArray[i]));
      } break;
      default:
        break;
    }
    // We re-arrange the indices so that it describe a simple list of
    // triangles
    switch (meshPrimitive.mode) {
      // case TINYGLTF_MODE_TRIANGLE_FAN: // TODO
      // case TINYGLTF_MODE_TRIANGLE_STRIP: // TODO
      case TINYGLTF_MODE_TRIANGLES:  // this is the simpliest case to handle
        {
          for (auto const& attribute : meshPrimitive.attributes) {
            auto const attribAccessor = model->accessors[attribute.second];
            auto const& bufferView =
                model->bufferViews[attribAccessor.bufferView];
            auto const& buffer = model->buffers[bufferView.buffer];
            auto const dataPtr = buffer.data.data() + bufferView.byteOffset +
                                 attribAccessor.byteOffset;
            auto const byte_stride = attribAccessor.ByteStride(bufferView);
            auto const count = attribAccessor.count;
            if (attribute.first == "POSITION") {
              switch (attribAccessor.type) {
                case TINYGLTF_TYPE_VEC3: {
                  positionMax = { attribAccessor.maxValues[0], attribAccessor.maxValues[1],attribAccessor.maxValues[2] };
                  positionMin = { attribAccessor.minValues[0], attribAccessor.minValues[1],attribAccessor.minValues[2] };
                  switch (attribAccessor.componentType) {
                    case TINYGLTF_COMPONENT_TYPE_FLOAT:
                      // 3D vector of float
                      ArrayAdapter<se::vec3> positions(dataPtr, count,
                                                         byte_stride);
                      for (size_t i{0}; i < indexArray_uint.size() / 3; ++i) {
                        // get the i'th triange's indexes
                        auto f0 = indexArray_uint[3 * i + 0];
                        auto f1 = indexArray_uint[3 * i + 1];
                        auto f2 = indexArray_uint[3 * i + 2];
                        // get the 3 normal vectors for that face
                        se::vec3 p0, p1, p2;
                        p0 = positions[f0];
                        p1 = positions[f1];
                        p2 = positions[f2];
                        // Put them in the array in the correct order
                        vertexBuffer_positionOnly.push_back(p0.x);
                        vertexBuffer_positionOnly.push_back(p0.y);
                        vertexBuffer_positionOnly.push_back(p0.z);

                        vertexBuffer_positionOnly.push_back(p1.x);
                        vertexBuffer_positionOnly.push_back(p1.y);
                        vertexBuffer_positionOnly.push_back(p1.z);

                        vertexBuffer_positionOnly.push_back(p2.x);
                        vertexBuffer_positionOnly.push_back(p2.y);
                        vertexBuffer_positionOnly.push_back(p2.z);
                      }
                  }
                  break;
                  case TINYGLTF_COMPONENT_TYPE_DOUBLE: {
                    switch (attribAccessor.type) {
                      case TINYGLTF_TYPE_VEC3: {
                        ArrayAdapter<se::dvec3> positions(dataPtr, count,
                                                            byte_stride);
                        for (size_t i{0}; i < indexArray_uint.size() / 3; ++i) {
                          // get the i'th triange's indexes
                          auto f0 = indexArray_uint[3 * i + 0];
                          auto f1 = indexArray_uint[3 * i + 1];
                          auto f2 = indexArray_uint[3 * i + 2];
                          // get the 3 normal vectors for that face
                          se::dvec3 p0, p1, p2;
                          p0 = positions[f0];
                          p1 = positions[f1];
                          p2 = positions[f2];
                          // Put them in the array in the correct order
                          vertexBuffer_positionOnly.push_back(p0.x);
                          vertexBuffer_positionOnly.push_back(p0.y);
                          vertexBuffer_positionOnly.push_back(p0.z);

                          vertexBuffer_positionOnly.push_back(p1.x);
                          vertexBuffer_positionOnly.push_back(p1.y);
                          vertexBuffer_positionOnly.push_back(p1.z);

                          vertexBuffer_positionOnly.push_back(p2.x);
                          vertexBuffer_positionOnly.push_back(p2.y);
                          vertexBuffer_positionOnly.push_back(p2.z);
                        }
                      } break;
                      default:
                        // TODO Handle error
                        break;
                    }
                    break;
                    default:
                      break;
                  }
                } break;
              }
            }
            if (attribute.first == "NORMAL") {
              switch (attribAccessor.type) {
                case TINYGLTF_TYPE_VEC3: {
                  switch (attribAccessor.componentType) {
                    case TINYGLTF_COMPONENT_TYPE_FLOAT: {
                      ArrayAdapter<se::vec3> normals(dataPtr, count,
                                                       byte_stride);
                      // For each triangle :
                      for (size_t i{0}; i < indexArray_uint.size() / 3; ++i) {
                        // get the i'th triange's indexes
                        auto f0 = indexArray_uint[3 * i + 0];
                        auto f1 = indexArray_uint[3 * i + 1];
                        auto f2 = indexArray_uint[3 * i + 2];
                        // get the 3 normal vectors for that face
                        se::vec3 n0, n1, n2;
                        n0 = normals[f0];
                        n1 = normals[f1];
                        n2 = normals[f2];
                        // Put them in the array in the correct order
                        vertexBuffer_normalOnly.push_back(n0.x);
                        vertexBuffer_normalOnly.push_back(n0.y);
                        vertexBuffer_normalOnly.push_back(n0.z);

                        vertexBuffer_normalOnly.push_back(n1.x);
                        vertexBuffer_normalOnly.push_back(n1.y);
                        vertexBuffer_normalOnly.push_back(n1.z);

                        vertexBuffer_normalOnly.push_back(n2.x);
                        vertexBuffer_normalOnly.push_back(n2.y);
                        vertexBuffer_normalOnly.push_back(n2.z);
                      }
                    } break;
                    case TINYGLTF_COMPONENT_TYPE_DOUBLE: {
                      ArrayAdapter<se::dvec3> normals(dataPtr, count,
                                                        byte_stride);
                      // IMPORTANT: We need to reorder normals (and texture
                      // coordinates into "facevarying" order) for each face
                      // For each triangle :
                      for (size_t i{0}; i < indexArray_uint.size() / 3; ++i) {
                        // get the i'th triange's indexes
                        auto f0 = indexArray_uint[3 * i + 0];
                        auto f1 = indexArray_uint[3 * i + 1];
                        auto f2 = indexArray_uint[3 * i + 2];
                        // get the 3 normal vectors for that face
                        se::dvec3 n0, n1, n2;
                        n0 = normals[f0];
                        n1 = normals[f1];
                        n2 = normals[f2];
                        // Put them in the array in the correct order
                        vertexBuffer_normalOnly.push_back(n0.x);
                        vertexBuffer_normalOnly.push_back(n0.y);
                        vertexBuffer_normalOnly.push_back(n0.z);

                        vertexBuffer_normalOnly.push_back(n1.x);
                        vertexBuffer_normalOnly.push_back(n1.y);
                        vertexBuffer_normalOnly.push_back(n1.z);

                        vertexBuffer_normalOnly.push_back(n2.x);
                        vertexBuffer_normalOnly.push_back(n2.y);
                        vertexBuffer_normalOnly.push_back(n2.z);
                      }
                    } break;
                  }
                }
              }
            }
            if (attribute.first == "TEXCOORD_0") {
              switch (attribAccessor.type) {
                case TINYGLTF_TYPE_VEC2: {
                  switch (attribAccessor.componentType) {
                    case TINYGLTF_COMPONENT_TYPE_FLOAT: {
                      ArrayAdapter<se::vec2> uvs(dataPtr, count, byte_stride);
                      for (size_t i{0}; i < indexArray_uint.size() / 3; ++i) {
                        // get the i'th triange's indexes
                        auto f0 = indexArray_uint[3 * i + 0];
                        auto f1 = indexArray_uint[3 * i + 1];
                        auto f2 = indexArray_uint[3 * i + 2];
                        // get the texture coordinates for each triangle's
                        // vertices
                        se::vec2 uv0, uv1, uv2;
                        uv0 = uvs[f0];
                        uv1 = uvs[f1];
                        uv2 = uvs[f2];
                        // push them in order into the mesh data
                        vertexBuffer_uvOnly.push_back(uv0.x);
                        vertexBuffer_uvOnly.push_back(uv0.y);

                        vertexBuffer_uvOnly.push_back(uv1.x);
                        vertexBuffer_uvOnly.push_back(uv1.y);

                        vertexBuffer_uvOnly.push_back(uv2.x);
                        vertexBuffer_uvOnly.push_back(uv2.y);
                      }

                    } break;
                    case TINYGLTF_COMPONENT_TYPE_DOUBLE: {
                      ArrayAdapter<se::dvec2> uvs(dataPtr, count,
                                                    byte_stride);
                      for (size_t i{0}; i < indexArray_uint.size() / 3; ++i) {
                        // get the i'th triange's indexes
                        auto f0 = indexArray_uint[3 * i + 0];
                        auto f1 = indexArray_uint[3 * i + 1];
                        auto f2 = indexArray_uint[3 * i + 2];

                        se::dvec2 uv0, uv1, uv2;
                        uv0 = uvs[f0];
                        uv1 = uvs[f1];
                        uv2 = uvs[f2];

                        vertexBuffer_uvOnly.push_back(uv0.x);
                        vertexBuffer_uvOnly.push_back(uv0.y);

                        vertexBuffer_uvOnly.push_back(uv1.x);
                        vertexBuffer_uvOnly.push_back(uv1.y);

                        vertexBuffer_uvOnly.push_back(uv2.x);
                        vertexBuffer_uvOnly.push_back(uv2.y);
                      }
                    } break;
                    default:
                      root::print::error(
                          "GFX :: tinygltf :: unrecognized vector type for "
                          "UV");
                  }
                } break;
                default:
                  root::print::error("GFX :: tinygltf :: unreconized componant type for UV");
              }
            }
            if (attribute.first == "TANGENT") {
              switch (attribAccessor.type) {
                case TINYGLTF_TYPE_VEC3: {
                  switch (attribAccessor.componentType) {
                    case TINYGLTF_COMPONENT_TYPE_FLOAT: {
                      ArrayAdapter<se::vec3> tangents(dataPtr, count,
                                                       byte_stride);
                      // For each triangle :
                      for (size_t i{0}; i < indexArray_uint.size() / 3; ++i) {
                        // get the i'th triange's indexes
                        auto f0 = indexArray_uint[3 * i + 0];
                        auto f1 = indexArray_uint[3 * i + 1];
                        auto f2 = indexArray_uint[3 * i + 2];
                        // get the 3 normal vectors for that face
                        se::vec3 n0, n1, n2;
                        n0 = tangents[f0];
                        n1 = tangents[f1];
                        n2 = tangents[f2];
                        // Put them in the array in the correct order
                        vertexBuffer_tangentOnly.push_back(n0.x);
                        vertexBuffer_tangentOnly.push_back(n0.y);
                        vertexBuffer_tangentOnly.push_back(n0.z);

                        vertexBuffer_tangentOnly.push_back(n1.x);
                        vertexBuffer_tangentOnly.push_back(n1.y);
                        vertexBuffer_tangentOnly.push_back(n1.z);

                        vertexBuffer_tangentOnly.push_back(n2.x);
                        vertexBuffer_tangentOnly.push_back(n2.y);
                        vertexBuffer_tangentOnly.push_back(n2.z);
                      }
                    } break;
                    case TINYGLTF_COMPONENT_TYPE_DOUBLE: {
                      ArrayAdapter<se::dvec3> tangents(dataPtr, count,
                                                        byte_stride);
                      // IMPORTANT: We need to reorder normals (and texture
                      // coordinates into "facevarying" order) for each face
                      // For each triangle :
                      for (size_t i{0}; i < indexArray_uint.size() / 3; ++i) {
                        // get the i'th triange's indexes
                        auto f0 = indexArray_uint[3 * i + 0];
                        auto f1 = indexArray_uint[3 * i + 1];
                        auto f2 = indexArray_uint[3 * i + 2];
                        // get the 3 normal vectors for that face
                        se::dvec3 n0, n1, n2;
                        n0 = tangents[f0];
                        n1 = tangents[f1];
                        n2 = tangents[f2];
                        // Put them in the array in the correct order
                        vertexBuffer_tangentOnly.push_back(n0.x);
                        vertexBuffer_tangentOnly.push_back(n0.y);
                        vertexBuffer_tangentOnly.push_back(n0.z);

                        vertexBuffer_tangentOnly.push_back(n1.x);
                        vertexBuffer_tangentOnly.push_back(n1.y);
                        vertexBuffer_tangentOnly.push_back(n1.z);

                        vertexBuffer_tangentOnly.push_back(n2.x);
                        vertexBuffer_tangentOnly.push_back(n2.y);
                        vertexBuffer_tangentOnly.push_back(n2.z);
                      }
                    } break;
                  }
                }
              }
            }  
            if (attribute.first == "JOINTS_0") {
              switch (attribAccessor.type) {
                case TINYGLTF_TYPE_VEC4: {
                  switch (attribAccessor.componentType) {
                    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: {
                      ArrayAdapter<se::Vector4<uint16_t>> joints(dataPtr, count, byte_stride);
                      // For each triangle :
                      for (size_t i{0}; i < indexArray_uint.size() / 3; ++i) {
                        // get the i'th triange's indexes
                        auto f0 = indexArray_uint[3 * i + 0];
                        auto f1 = indexArray_uint[3 * i + 1];
                        auto f2 = indexArray_uint[3 * i + 2];
                        // get the 3 normal vectors for that face
                        se::Vector4<uint16_t> j0, j1, j2;
                        j0 = joints[f0];
                        j1 = joints[f1];
                        j2 = joints[f2];
                        // Put them in the array in the correct order
                        vertexBuffer_joints.push_back(j0.x);
                        vertexBuffer_joints.push_back(j0.y);
                        vertexBuffer_joints.push_back(j0.z);
                        vertexBuffer_joints.push_back(j0.w);

                        vertexBuffer_joints.push_back(j1.x);
                        vertexBuffer_joints.push_back(j1.y);
                        vertexBuffer_joints.push_back(j1.z);
                        vertexBuffer_joints.push_back(j1.w);

                        vertexBuffer_joints.push_back(j2.x);
                        vertexBuffer_joints.push_back(j2.y);
                        vertexBuffer_joints.push_back(j2.z);
                        vertexBuffer_joints.push_back(j2.w);
                      }
                    } break;
                  }
                }
              }
            }  
            if (attribute.first == "WEIGHTS_0") {
              switch (attribAccessor.type) {
                case TINYGLTF_TYPE_VEC4: {
                  switch (attribAccessor.componentType) {
                    case TINYGLTF_COMPONENT_TYPE_FLOAT: {
                      ArrayAdapter<se::vec4> weights(dataPtr, count, byte_stride);
                      // For each triangle :
                      for (size_t i{0}; i < indexArray_uint.size() / 3; ++i) {
                        // get the i'th triange's indexes
                        auto f0 = indexArray_uint[3 * i + 0];
                        auto f1 = indexArray_uint[3 * i + 1];
                        auto f2 = indexArray_uint[3 * i + 2];
                        // get the 3 normal vectors for that face
                        se::vec4 w0, w1, w2;
                        w0 = weights[f0];
                        w1 = weights[f1];
                        w2 = weights[f2];
                        // Put them in the array in the correct order
                        vertexBuffer_weights.push_back(w0.x);
                        vertexBuffer_weights.push_back(w0.y);
                        vertexBuffer_weights.push_back(w0.z);
                        vertexBuffer_weights.push_back(w0.w);

                        vertexBuffer_weights.push_back(w1.x);
                        vertexBuffer_weights.push_back(w1.y);
                        vertexBuffer_weights.push_back(w1.z);
                        vertexBuffer_weights.push_back(w1.w);

                        vertexBuffer_weights.push_back(w2.x);
                        vertexBuffer_weights.push_back(w2.y);
                        vertexBuffer_weights.push_back(w2.z);
                        vertexBuffer_weights.push_back(w2.w);
                      }
                    } break;
                    case TINYGLTF_COMPONENT_TYPE_DOUBLE: {
                      ArrayAdapter<se::dvec4> tangents(dataPtr, count, byte_stride);
                      // IMPORTANT: We need to reorder normals (and texture
                      // coordinates into "facevarying" order) for each face
                      // For each triangle :
                      for (size_t i{0}; i < indexArray_uint.size() / 3; ++i) {
                        // get the i'th triange's indexes
                        auto f0 = indexArray_uint[3 * i + 0];
                        auto f1 = indexArray_uint[3 * i + 1];
                        auto f2 = indexArray_uint[3 * i + 2];
                        // get the 3 normal vectors for that face
                        se::dvec4 w0, w1, w2;
                        w0 = tangents[f0];
                        w1 = tangents[f1];
                        w2 = tangents[f2];
                        // Put them in the array in the correct order
                        vertexBuffer_weights.push_back(w0.x);
                        vertexBuffer_weights.push_back(w0.y);
                        vertexBuffer_weights.push_back(w0.z);
                        vertexBuffer_weights.push_back(w0.w);

                        vertexBuffer_weights.push_back(w1.x);
                        vertexBuffer_weights.push_back(w1.y);
                        vertexBuffer_weights.push_back(w1.z);
                        vertexBuffer_weights.push_back(w1.w);

                        vertexBuffer_weights.push_back(w2.x);
                        vertexBuffer_weights.push_back(w2.y);
                        vertexBuffer_weights.push_back(w2.z);
                        vertexBuffer_weights.push_back(w2.w);
                      }
                    } break;
                  }
                }
              }
            }  
        }
          break;
        }
      default:
        root::print::error("GFX :: tinygltf :: primitive mode not implemented");
        break;
    }
    // Compute the tangent vector if no provided
    if (vertexBuffer_tangentOnly.size() == 0) {
      for (size_t i = 0; i < indexArray_uint.size(); i += 3) {
        if (vertexBuffer_uvOnly.size() == 0) {
            // if has no uv
            for (int i = 0; i < 3; ++i) {
              vertexBuffer_tangentOnly.push_back(0);
              vertexBuffer_tangentOnly.push_back(0);
              vertexBuffer_tangentOnly.push_back(0);
            }
        } else {
            size_t i0 = i + 0; size_t i1 = i + 1; size_t i2 = i + 2;
            se::vec3 pos1 = {vertexBuffer_positionOnly[i0 * 3 + 0],
                               vertexBuffer_positionOnly[i0 * 3 + 1],
                               vertexBuffer_positionOnly[i0 * 3 + 2]};
            se::vec3 pos2 = {vertexBuffer_positionOnly[i1 * 3 + 0],
                               vertexBuffer_positionOnly[i1 * 3 + 1],
                               vertexBuffer_positionOnly[i1 * 3 + 2]};
            se::vec3 pos3 = {vertexBuffer_positionOnly[i2 * 3 + 0],
                               vertexBuffer_positionOnly[i2 * 3 + 1],
                               vertexBuffer_positionOnly[i2 * 3 + 2]};
            se::vec2 uv1  = {vertexBuffer_uvOnly[i0 * 2 + 0] - int(vertexBuffer_uvOnly[i0 * 2 + 0]),
                               vertexBuffer_uvOnly[i0 * 2 + 1] - int(vertexBuffer_uvOnly[i0 * 2 + 1]) };
            se::vec2 uv2  = {vertexBuffer_uvOnly[i1 * 2 + 0] - int(vertexBuffer_uvOnly[i1 * 2 + 0]),
                               vertexBuffer_uvOnly[i1 * 2 + 1] - int(vertexBuffer_uvOnly[i1 * 2 + 1]) };
            se::vec2 uv3  = {vertexBuffer_uvOnly[i2 * 2 + 0] - int(vertexBuffer_uvOnly[i2 * 2 + 0]),
                               vertexBuffer_uvOnly[i2 * 2 + 1] - int(vertexBuffer_uvOnly[i2 * 2 + 1]) };

            se::vec3 tangent;
            se::vec3 edge1 = pos2 - pos1;
            se::vec3 edge2 = pos3 - pos1;
            se::vec2 deltaUV1 = uv2 - uv1;
            se::vec2 deltaUV2 = uv3 - uv1;
            float f = 1.0f / (deltaUV1.x * deltaUV2.y - deltaUV2.x * deltaUV1.y);
            tangent.x = f * (deltaUV2.y * edge1.x - deltaUV1.y * edge2.x);
            tangent.y = f * (deltaUV2.y * edge1.y - deltaUV1.y * edge2.y);
            tangent.z = f * (deltaUV2.y * edge1.z - deltaUV1.y * edge2.z);
            tangent = se::normalize(tangent);
            for (int i = 0; i < 3; ++i) {
              vertexBuffer_tangentOnly.push_back(tangent.x);
              vertexBuffer_tangentOnly.push_back(tangent.y);
              vertexBuffer_tangentOnly.push_back(tangent.z);
            }
        }
      }
    }
    // Assemble vertex buffer
    for (size_t i = 0; i < indexArray_uint.size(); ++i) {
      for (auto const& entry : defaultMeshLoadConfig.layout.layout) {
        // vertex position
        if (entry.info == MeshDataLayout::VertexInfo::POSITION) {
            if (entry.format == rhi::VertexFormat::FLOAT32X3) {
              //vertexBuffer.push_back(vertexBuffer_positionOnly[i * 3 + 0]);
              //vertexBuffer.push_back(vertexBuffer_positionOnly[i * 3 + 1]);
              //vertexBuffer.push_back(vertexBuffer_positionOnly[i * 3 + 2]);
              if (defaultMeshLoadConfig.usePositionBuffer) {
                if (vertexBuffer_positionOnly.size() != 0) {
                    PositionBuffer.push_back(vertexBuffer_positionOnly[i * 3 + 0]);
                    PositionBuffer.push_back(vertexBuffer_positionOnly[i * 3 + 1]);
                    PositionBuffer.push_back(vertexBuffer_positionOnly[i * 3 + 2]);
                } else {
                
                }
              }
            } else {
              root::print::error(
                  "GFX :: SceneNodeLoader_assimp :: unwanted vertex format for "
                  "POSITION attributes.");
              return {};
            }
        } else if (entry.info == MeshDataLayout::VertexInfo::NORMAL) {
            if (vertexBuffer_normalOnly.size() == 0) { // if normal is not provided
              vertexBuffer.push_back(0.f);
              vertexBuffer.push_back(0.f);
              vertexBuffer.push_back(0.f);            
            } else {
              vertexBuffer.push_back(vertexBuffer_normalOnly[i * 3 + 0]);
              vertexBuffer.push_back(vertexBuffer_normalOnly[i * 3 + 1]);
              vertexBuffer.push_back(vertexBuffer_normalOnly[i * 3 + 2]);            
            }
        } else if (entry.info == MeshDataLayout::VertexInfo::UV) {
            if (vertexBuffer_uvOnly.size() == 0) { // if uv is not provided
              vertexBuffer.push_back(0.f);
              vertexBuffer.push_back(0.f);
            } else {
              vertexBuffer.push_back(vertexBuffer_uvOnly[i * 2 + 0] - int(vertexBuffer_uvOnly[i * 2 + 0]));
              vertexBuffer.push_back(vertexBuffer_uvOnly[i * 2 + 1] - int(vertexBuffer_uvOnly[i * 2 + 1]));
            }
        } else if (entry.info == MeshDataLayout::VertexInfo::TANGENT) {
            vertexBuffer.push_back(vertexBuffer_tangentOnly[i * 3 + 0]);
            vertexBuffer.push_back(vertexBuffer_tangentOnly[i * 3 + 1]);
            vertexBuffer.push_back(vertexBuffer_tangentOnly[i * 3 + 2]);
        } else if (entry.info == MeshDataLayout::VertexInfo::COLOR) {
            // Optional: vertex colors
            vertexBuffer.push_back(0);
            vertexBuffer.push_back(0);
            vertexBuffer.push_back(0);
        } else if (entry.info == MeshDataLayout::VertexInfo::CUSTOM) {
        }
      }

      indexBuffer_uint.push_back(i);
    }
    // Assemble skin buffer
    if (vertexBuffer_joints.size() != 0) {
      JointIndexBuffer.insert(JointIndexBuffer.end(), vertexBuffer_joints.begin(), vertexBuffer_joints.end());
      JointweightsBuffer.insert(JointweightsBuffer.end(), vertexBuffer_weights.begin(), vertexBuffer_weights.end());
    }

    //loadGLTFMaterial()
    Mesh::MeshPrimitive sePrimitive; 
    sePrimitive.offset = submesh_index_offset;
    sePrimitive.size = indexArray_uint.size();
    sePrimitive.baseVertex = submesh_vertex_offset;
    sePrimitive.numVertex = PositionBuffer.size() / 3 - submesh_vertex_offset;
    sePrimitive.max = positionMax;
    sePrimitive.min = positionMin;
    if (meshPrimitive.material != -1) {
      auto const& gltf_material = model->materials[meshPrimitive.material];
      sePrimitive.material = loadGLTFMaterial(&gltf_material, model, env, scene, defaultMeshLoadConfig);
    }
    mesh.get()->primitives.emplace_back(std::move(sePrimitive));
    // todo:: add material
    submesh_index_offset = indexBuffer_uint.size();
    submesh_vertex_offset = PositionBuffer.size() / 3;
  }
  // create mesh resource
  { // register mesh
    Buffer* position_buffer = scene.gpuScene.position_buffer.get();
    size_t position_size = sizeof(float) * PositionBuffer.size();
    size_t position_offset = position_buffer->host.size();
    position_buffer->host.resize(position_size + position_offset);
    memcpy(&position_buffer->host[position_offset], PositionBuffer.data(), position_size);
    mesh.get()->vertex_offset = position_offset;

    Buffer* index_buffer = scene.gpuScene.index_buffer.get();
    size_t index_size = sizeof(uint32_t) * indexBuffer_uint.size();
    size_t index_offset = index_buffer->host.size();
    index_buffer->host.resize(index_size + index_offset);
    memcpy(&index_buffer->host[index_offset], indexBuffer_uint.data(), index_size);
    mesh.get()->index_offset = index_offset;

    Buffer* vertex_buffer = scene.gpuScene.vertex_buffer.get();
    size_t vertex_size = sizeof(float) * vertexBuffer.size();
    size_t vertex_offset = vertex_buffer->host.size();
    vertex_buffer->host.resize(vertex_size + vertex_offset);
    memcpy(&vertex_buffer->host[vertex_offset], vertexBuffer.data(), vertex_size);
    
    //if (true) {
    //  mesh.jointIndexBuffer_host = Core::Buffer(sizeof(uint64_t) * JointIndexBuffer.size());
    //  memcpy(mesh.jointIndexBuffer_host.data, JointIndexBuffer.data(), mesh.jointIndexBuffer_host.size);
    //  mesh.jointIndexBufferInfo.onHost = true;
    //  mesh.jointIndexBufferInfo.size = mesh.jointIndexBuffer_host.size;

    //  mesh.jointWeightBuffer_host = Core::Buffer(sizeof(float) * JointweightsBuffer.size());
    //  memcpy(mesh.jointWeightBuffer_host.data, JointweightsBuffer.data(), mesh.jointWeightBuffer_host.size);
    //  mesh.jointWeightBufferInfo.onHost = true;
    //  mesh.jointWeightBufferInfo.size = mesh.jointWeightBuffer_host.size;
    //}
  }
  return mesh;
 }
 
SceneLoader::result_type SceneLoader::operator()(SceneLoader::from_gltf_tag, std::string const& path) {
  // load the gltf file
  tinygltf::TinyGLTF loader;
  tinygltf::Model model;
  std::string err;
  std::string warn;
  bool ret = loader.LoadASCIIFromFile(&model, &err, &warn, path);
  if (!warn.empty()) {
    root::print::error("Scene::deserialize warn::" + warn); return nullptr;
  } if (!err.empty()) {
    root::print::error("Scene::deserialize error::" + err); return nullptr;
  } if (!ret) {
    root::print::error("Failed to parse glTF"); return nullptr;
  }
  // parse the gltf file to SE scene
  // -------------------------------------------------------------
  SceneLoader::result_type scene = std::make_shared<Scene>();
  // first, create the nodes
  std::vector<Node> nodes(model.nodes.size());
  for (size_t i = 0; i < model.nodes.size(); ++i) {
    nodes[i] = scene->createNode(model.nodes[i].name);
  }
  // add the hierarchy information
  for (size_t i = 0; i < model.nodes.size(); ++i) {
    auto& children = scene->registry.get<NodeProperty>(nodes[i].entity).children;
    for (auto& child_id : model.nodes[i].children) {
      children.push_back(nodes[child_id]);
    }
  }
  // set default scene
  int scene_idx = model.defaultScene;
  if (model.scenes.empty()) {
    root::print::error("SceneLoader::from_gltf_tag :: Scene is empty");
    return nullptr;
  } else if (model.defaultScene >= int(model.scenes.size())) {
    root::print::error("SceneLoader::tinygltf :: Invalid defualtScene value");
    return nullptr;
  } else if (model.defaultScene == -1) {
    // Use the first scene.
    scene_idx = 0;
  }
  // set all root nodes for the default scene
  scene->roots.clear();
  for (auto node : model.scenes[scene_idx].nodes) {
    scene->roots.push_back(nodes[node]);
  }
  std::filesystem::path filepath = path;
  std::string const directory = filepath.parent_path().string();
  glTFLoaderEnv env;
  env.directory = directory;    
  // load tag, transform, mesh
  for (size_t i = 0; i < model.nodes.size(); ++i) {
    auto const& gltfNode = model.nodes[i];
    auto& seNode = nodes[i];
    // process the transform
    {
      auto& transform = scene->registry.get<Transform>(seNode.entity);
      if (gltfNode.scale.size() == 3)
        transform.scale = { static_cast<float>(gltfNode.scale[0]),
          static_cast<float>(gltfNode.scale[1]), static_cast<float>(gltfNode.scale[2]) };
      if (gltfNode.translation.size() == 3)
        transform.translation = { static_cast<float>(gltfNode.translation[0]),
          static_cast<float>(gltfNode.translation[1]), static_cast<float>(gltfNode.translation[2]) };
      if (gltfNode.rotation.size() == 4) {
        transform.rotation = { float(gltfNode.rotation[0]), float(gltfNode.rotation[1]),
          float(gltfNode.rotation[2]), float(gltfNode.rotation[3]) };
      }
      if (gltfNode.matrix.size() == 16) {
        se::mat4 mat = se::mat4{
          (float)gltfNode.matrix[0], (float)gltfNode.matrix[1], (float)gltfNode.matrix[2],  (float)gltfNode.matrix[3],
          (float)gltfNode.matrix[4], (float)gltfNode.matrix[5], (float)gltfNode.matrix[6],  (float)gltfNode.matrix[7],
          (float)gltfNode.matrix[8], (float)gltfNode.matrix[9], (float)gltfNode.matrix[10], (float)gltfNode.matrix[11],
          (float)gltfNode.matrix[12],(float)gltfNode.matrix[13],(float)gltfNode.matrix[14], (float)gltfNode.matrix[15],
        };
        mat = se::transpose(mat);
        se::vec3 t, s; se::Quaternion quat;
        se::decompose(mat, &t, &quat, &s);
        transform.translation = t;
        transform.rotation = { quat.x, quat.y, quat.z, quat.w };
        transform.scale = s;
      }
    }
    // process the mesh
    if (gltfNode.mesh != -1) {
      MeshHandle mesh = loadGLTFMesh(model.meshes[gltfNode.mesh], seNode, *scene.get(), i, &model, env);
      scene->gpuScene.position_buffer->host_stamp++;
      scene->gpuScene.index_buffer->host_stamp++;
      scene->gpuScene.vertex_buffer->host_stamp++;
      auto& mesh_renderer = scene->registry.emplace<MeshRenderer>(seNode.entity);
      mesh_renderer.mesh = mesh;
    }
    // process the camera
    if (gltfNode.camera != -1) {
      auto& camera = scene->registry.emplace<Camera>(seNode.entity);
      auto& gltf_camera = model.cameras[gltfNode.camera];
      if (gltf_camera.type == "perspective") {
        camera.zfar = gltf_camera.perspective.zfar;
        camera.znear = gltf_camera.perspective.znear;
        camera.yfov = gltf_camera.perspective.yfov;
        camera.aspectRatio = gltf_camera.perspective.aspectRatio;
      }
    }
  }

  //processGLTFMesh(gfxNode, node_id, model, env, gfxscene, meshConfig);
  //processGLTFAnimation(gfxNode, node_id, model, env, gfxscene, meshConfig);

  //// process the meshes for all the following nodes
  //tinygltf::Node const& node = model->nodes[node_id];
  //for (uint32_t i : node.children) {
  //    GameObjectHandle subNode = gfxscene.createGameObject(gfxNode);
  //    processGLTFNode(subNode, i, model, env, gfxscene, meshConfig);
  //}

  //for (int i = 0; i < env.skinning_indices.size(); ++i) {
  //  auto& skin_indices = env.skinning_indices[i];
  //  auto& skin_matrices = env.skinning_matrices[i];
  //  for (int j = 0; j < skin_indices.size(); ++j) {
  //    int const index = skin_indices[j];
  //    GFX::GameObjectHandle go = env.node2go[index];
  //    TransformComponent* trans = gfxscene.getGameObject(go)
  //      ->getEntity().getComponent<TransformComponent>();
  //    trans->flag |= (uint32_t)TransformComponent::FlagBit::IS_SKELETON_JOINT;
  //    trans->inverseJointTransform = skin_matrices[j];
  //  }
  //}

  //// post-process all nodes
  //for (auto pair : env.node2go) {
  //  MeshReference* meshref = gfxscene.getGameObject(pair.second)
  //      ->getEntity().getComponent<MeshReference>();
  //  if (meshref) {
  //    // if has joint index buffer, assign the real handle
  //    if (meshref->mesh->jointIndexBufferInfo.size > 0) {
  //      uint64_t* indices = (uint64_t*)meshref->mesh->jointIndexBuffer_host.data;
  //      size_t size = meshref->mesh->jointIndexBuffer_host.size / sizeof(uint64_t);
  //      for (size_t i = 0; i < size; ++i) {
  //        indices[i] = env.node2go[indices[i]];
  //      }
  //    }
  //  }
  //}
  return scene;
}

SceneLoader::result_type SceneLoader::operator()(SceneLoader::from_scratch_tag) {
  SceneLoader::result_type scene = std::make_shared<Scene>();
  // create node
  //Node cameraNode = scene->createNode("Main Camera");
  //auto& camera = scene->registry.emplace<Camera>(cameraNode.entity);
  //camera.zfar = 1000.f;
  //camera.znear = 0.02f;
  //camera.yfov = 45.f;
  //camera.aspectRatio = 1.f;
  // create default light
  Node envmapNode = scene->createNode("Environment");
  auto& envLight = scene->registry.emplace<Light>(envmapNode.entity);
  envLight.type = Light::LightType::ENVIRONMENT;
  envLight.intensity = se::vec3(0, 0, 0);
  return scene;
}

ex::resource_cache<Scene, SceneLoader> GFXContext::scenes;

auto GFXContext::load_scene_gltf(std::string const& path) noexcept -> SceneHandle {
  RUID const ruid = root::resource::queryRUID();
  auto ret = scenes.load(ruid, SceneLoader::from_gltf_tag{}, path);
  ret.first->second->dirtyFlags =
    (uint64_t)se::gfx::Scene::DirtyFlagBit::Camera
    | (uint64_t)se::gfx::Scene::DirtyFlagBit::Geometry;
  return SceneHandle{ ret.first->second };
}

auto GFXContext::create_scene(std::string const& name) noexcept -> SceneHandle {
  RUID const ruid = root::resource::queryRUID();
  auto ret = scenes.load(ruid, SceneLoader::from_scratch_tag{});
  ret.first->second->dirtyFlags =
    (uint64_t)se::gfx::Scene::DirtyFlagBit::Camera
    | (uint64_t)se::gfx::Scene::DirtyFlagBit::Geometry;
  return SceneHandle{ ret.first->second };
}
}