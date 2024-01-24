#pragma once
#include <Misc/SE.Core.Misc.hpp>
#include <SE.GFX-Loader.SceneNodeLoader.hpp>
#include <Print/SE.Core.Log.hpp>
#include <ECS/SE.Core.ECS.hpp>
#include <Memory/SE.Core.Memory.hpp>
#include <Resource/SE.Core.Resource.hpp>
#include <SE.Math.Geometric.hpp>
#include <Config/SE.Core.Config.hpp>
#include <SE.GFX-SerializeUtils.hpp>
#include <SE.GFX-Loader.MeshLoader.hpp>
#include <SE.RHI.hpp>

#include <span>
#include <format>
#include <string>
#include <unordered_map>
#include <vector>
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#pragma warning(disable : 4996)
#define TINYGLTF_IMPLEMENTATION
// #define STB_IMAGE_WRITE_IMPLEMENTATION
#define TINYGLTF_NOEXCEPTION
#define JSON_NOEXCEPTION
#include <tiny_obj_loader.h>
#include <tinygltf/tiny_gltf.h>
#include <SE.Image.hpp>
#include <SE.GFX-Main.hpp>
#include <tinyparser-mitsuba.h>

namespace SIByL::GFX {
static std::array<uint64_t, 24> primes = {3,  5,  7,  11, 13, 17, 19, 23,
                                          29, 31, 37, 41, 43, 47, 53, 59,
                                          61, 67, 71, 73, 79, 83, 89, 97};

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

/** Load obj file */
auto SceneNodeLoader_obj::loadSceneNode(
    std::filesystem::path const& path, GFX::Scene& gfxscene,
    MeshLoaderConfig meshConfig) noexcept -> void {
  loadSceneNode(path, gfxscene, GFX::NULL_GO, meshConfig);
}

auto SceneNodeLoader_obj::loadSceneNode(std::filesystem::path const& path,
    GFX::Scene& gfxscene,
    GameObjectHandle parent,
    MeshLoaderConfig meshConfig) noexcept
    -> GameObjectHandle {
  // load obj file
  std::string inputfile = path.string();
  tinyobj::ObjReaderConfig reader_config;
  reader_config.mtl_search_path =
      path.parent_path().string();  // Path to material files
  tinyobj::ObjReader reader;
  if (!reader.ParseFromFile(inputfile, reader_config)) {
    if (!reader.Error().empty()) {
      Core::LogManager::Error("TinyObjReader: " + reader.Error());
    }
    return GFX::NULL_GO;
  }
  if (!reader.Warning().empty()) {
    Core::LogManager::Warning("TinyObjReader: " + reader.Warning());
  }
  auto& attrib = reader.GetAttrib();
  auto& shapes = reader.GetShapes();
  auto& materials = reader.GetMaterials();

  uint32_t vertex_offset = 0;
  std::unordered_map<uint64_t, uint32_t> uniqueVertices{};

  std::vector<float> vertexBufferV = {};
  std::vector<float> positionBufferV = {};
  std::vector<uint16_t> indexBufferV = {};
  std::vector<uint32_t> indexBufferWV = {};

  // check whether tangent is need in mesh attributes
  bool needTangent = false;
  for (auto const& entry : meshConfig.layout.layout)
    if (entry.info == MeshDataLayout::VertexInfo::TANGENT) needTangent = true;

  // Loop over shapes
  GFX::Mesh mesh;
  uint64_t global_index_offset = 0;
  uint32_t submesh_vertex_offset = 0, submesh_index_offset = 0;
  for (size_t s = 0; s < shapes.size(); s++) {
    // Loop over faces(polygon)
    size_t index_offset = 0;
    for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
      size_t fv = size_t(shapes[s].mesh.num_face_vertices[f]);
      // require tangent
      if (fv != 3) {
        Core::LogManager::Error(
            "GFX :: SceneNodeLoader_obj :: non-triangle geometry not "
            "supported when required TANGENT attribute now.");
        return GFX::NULL_GO;
      }
      Math::vec3 tangent;
      Math::vec3 bitangent;
      if (needTangent) {
        Math::vec3 positions[3];
        Math::vec3 normals[3];
        Math::vec2 uvs[3];
        for (size_t v = 0; v < fv; v++) {
          // index finding
          tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
          positions[v] = {attrib.vertices[3 * size_t(idx.vertex_index) + 0],
                          attrib.vertices[3 * size_t(idx.vertex_index) + 1],
                          attrib.vertices[3 * size_t(idx.vertex_index) + 2]};
          if (attrib.normals.size() == 0) {
            normals[v] = {0, 0, 0};  
          } else {
            normals[v] = {attrib.normals[3 * size_t(idx.normal_index) + 0],
                          attrib.normals[3 * size_t(idx.normal_index) + 1],
                          attrib.normals[3 * size_t(idx.normal_index) + 2]};          
          }
          if (attrib.texcoords.size() == 0) {
            uvs[v] = {0, 0};
          } else {
            uvs[v] = {attrib.texcoords[2 * size_t(idx.texcoord_index) + 0],
                      -attrib.texcoords[2 * size_t(idx.texcoord_index) + 1]};

          }
        }
        Math::vec3 edge1 = positions[1] - positions[0];
        Math::vec3 edge2 = positions[2] - positions[0];
        Math::vec2 deltaUV1 = uvs[1] - uvs[0];
        Math::vec2 deltaUV2 = uvs[2] - uvs[0];

        float f = 1.0f / (deltaUV1.x * deltaUV2.y - deltaUV2.x * deltaUV1.y);

        tangent.x = f * (deltaUV2.y * edge1.x - deltaUV1.y * edge2.x);
        tangent.y = f * (deltaUV2.y * edge1.y - deltaUV1.y * edge2.y);
        tangent.z = f * (deltaUV2.y * edge1.z - deltaUV1.y * edge2.z);
        tangent = Math::normalize(tangent);

        bitangent.x = f * (-deltaUV2.x * edge1.x + deltaUV1.x * edge2.x);
        bitangent.y = f * (-deltaUV2.x * edge1.y + deltaUV1.x * edge2.y);
        bitangent.z = f * (-deltaUV2.x * edge1.z + deltaUV1.x * edge2.z);
        bitangent = Math::normalize(bitangent);
      }
      // Loop over vertices in the face.
      for (size_t v = 0; v < fv; v++) {
        // index finding
        tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
        // atrributes filling
        std::vector<float> vertex = {};
        std::vector<float> position = {};
        for (auto const& entry : meshConfig.layout.layout) {
          // vertex position
          if (entry.info == MeshDataLayout::VertexInfo::POSITION) {
            if (entry.format == RHI::VertexFormat::FLOAT32X3) {
              tinyobj::real_t vx =
                  attrib.vertices[3 * size_t(idx.vertex_index) + 0];
              tinyobj::real_t vy =
                  attrib.vertices[3 * size_t(idx.vertex_index) + 1];
              tinyobj::real_t vz =
                  attrib.vertices[3 * size_t(idx.vertex_index) + 2];
              vertex.push_back(vx);
              vertex.push_back(vy);
              vertex.push_back(vz);
              if (meshConfig.usePositionBuffer) {
                position.push_back(vx);
                position.push_back(vy);
                position.push_back(vz);
              }
            } else {
              Core::LogManager::Error(
                  "GFX :: SceneNodeLoader_obj :: unwanted vertex format for "
                  "POSITION attributes.");
              return GFX::NULL_GO;
            }
          } else if (entry.info == MeshDataLayout::VertexInfo::NORMAL) {
            // Check if `normal_index` is zero or positive. negative = no
            // normal data
            if (idx.normal_index >= 0) {
              tinyobj::real_t nx =
                  attrib.normals[3 * size_t(idx.normal_index) + 0];
              tinyobj::real_t ny =
                  attrib.normals[3 * size_t(idx.normal_index) + 1];
              tinyobj::real_t nz =
                  attrib.normals[3 * size_t(idx.normal_index) + 2];
              vertex.push_back(nx);
              vertex.push_back(ny);
              vertex.push_back(nz);
            } else {
              vertex.push_back(0);
              vertex.push_back(0);
              vertex.push_back(0);
            }
          } else if (entry.info == MeshDataLayout::VertexInfo::UV) {
            if (idx.texcoord_index >= 0) {
              tinyobj::real_t tx =
                  attrib.texcoords[2 * size_t(idx.texcoord_index) + 0];
              tinyobj::real_t ty =
                  attrib.texcoords[2 * size_t(idx.texcoord_index) + 1];
              vertex.push_back(tx);
              vertex.push_back(1 - ty);
            } else {
              vertex.push_back(0);
              vertex.push_back(0);
            }
          } else if (entry.info == MeshDataLayout::VertexInfo::TANGENT) {
            vertex.push_back(tangent.x);
            vertex.push_back(tangent.y);
            vertex.push_back(tangent.z);
          } else if (entry.info == MeshDataLayout::VertexInfo::COLOR) {
            // Optional: vertex colors
            tinyobj::real_t red =
                attrib.colors[3 * size_t(idx.vertex_index) + 0];
            tinyobj::real_t green =
                attrib.colors[3 * size_t(idx.vertex_index) + 1];
            tinyobj::real_t blue =
                attrib.colors[3 * size_t(idx.vertex_index) + 2];
            vertex.push_back(red);
            vertex.push_back(green);
            vertex.push_back(blue);
          } else if (entry.info == MeshDataLayout::VertexInfo::CUSTOM) {
          }
        }

        if (meshConfig.deduplication) {
          uint64_t hashed_vertex = hash_vertex(vertex);
          if (uniqueVertices.count(hashed_vertex) == 0) {
            uniqueVertices[hashed_vertex] =
                static_cast<uint32_t>(vertex_offset);
            vertexBufferV.insert(vertexBufferV.end(), vertex.begin(),
                                 vertex.end());
            positionBufferV.insert(positionBufferV.end(), position.begin(),
                                   position.end());
            ++vertex_offset;
          }

          // index filling
          if (meshConfig.layout.format == RHI::IndexFormat::UINT16_t)
            indexBufferV.push_back(uniqueVertices[hashed_vertex]);
          else if (meshConfig.layout.format == RHI::IndexFormat::UINT32_T)
            indexBufferWV.push_back(uniqueVertices[hashed_vertex]);
        } else {
          vertexBufferV.insert(vertexBufferV.end(), vertex.begin(),
                               vertex.end());
          ++vertex_offset;
          // index filling
          if (meshConfig.layout.format == RHI::IndexFormat::UINT16_t)
            indexBufferV.push_back(vertex_offset);
          else if (meshConfig.layout.format == RHI::IndexFormat::UINT32_T)
            indexBufferWV.push_back(vertex_offset);
        }
      }
      index_offset += fv;
      // per-face material
      shapes[s].mesh.material_ids[f];
    }
    global_index_offset += index_offset;

    mesh.submeshes.push_back(
        GFX::Mesh::Submesh{submesh_index_offset, uint32_t(index_offset),
                           submesh_vertex_offset, uint32_t(0)});
    submesh_index_offset = global_index_offset;
    submesh_vertex_offset = submesh_index_offset;
  }
  // create mesh resource
  {  // register mesh
    mesh.vertexBufferLayout = getVertexBufferLayout(meshConfig.layout);
    if (meshConfig.residentOnHost) {
      mesh.vertexBuffer_host =
          Core::Buffer(sizeof(float) * vertexBufferV.size());
      memcpy(mesh.vertexBuffer_host.data, vertexBufferV.data(),
             mesh.vertexBuffer_host.size);
      mesh.vertexBufferInfo.onHost = true;
      mesh.vertexBufferInfo.size = mesh.vertexBuffer_host.size;
      if (meshConfig.layout.format == RHI::IndexFormat::UINT16_t) {
        mesh.indexBuffer_host =
            Core::Buffer(sizeof(uint16_t) * indexBufferV.size());
        memcpy(mesh.indexBuffer_host.data, indexBufferV.data(),
               mesh.indexBuffer_host.size);
        mesh.indexBufferInfo.size = mesh.indexBuffer_host.size;
        mesh.indexBufferInfo.onHost = true;
      } else if (meshConfig.layout.format == RHI::IndexFormat::UINT32_T) {
        mesh.indexBuffer_host =
            Core::Buffer(sizeof(uint32_t) * indexBufferWV.size());
        memcpy(mesh.indexBuffer_host.data, indexBufferWV.data(),
               mesh.indexBuffer_host.size);
        mesh.indexBufferInfo.size = mesh.indexBuffer_host.size;
        mesh.indexBufferInfo.onHost = true;
      }
      if (meshConfig.usePositionBuffer) {
        mesh.positionBuffer_host =
            Core::Buffer(sizeof(float) * positionBufferV.size());
        memcpy(mesh.positionBuffer_host.data, positionBufferV.data(),
               mesh.positionBuffer_host.size);
        mesh.positionBufferInfo.onHost = true;
        mesh.positionBufferInfo.size = mesh.positionBuffer_host.size;
      }
    }
  }
  Core::GUID guid =
      Core::ResourceManager::get()->requestRuntimeGUID<GFX::Mesh>();
  Core::ORID orid = Core::requestORID();
  Core::ResourceManager::get()->database.registerResource(orid, guid);
  Core::ResourceManager::get()->addResource<GFX::Mesh>(guid, std::move(mesh));
  GFX::Mesh* meshResourceRef =
      Core::ResourceManager::get()->getResource<GFX::Mesh>(guid);
  meshResourceRef->ORID = orid;
  meshResourceRef->serialize();
  // bind scene
  GameObjectHandle rootNode = gfxscene.createGameObject(parent);
  gfxscene.getGameObject(rootNode)
      ->getEntity()
      .getComponent<TagComponent>()
      ->name = path.filename().string();
  gfxscene.getGameObject(rootNode)
      ->getEntity()
      .addComponent<MeshReference>()
      ->mesh = meshResourceRef;
  return rootNode;
}

struct glTFLoaderEnv {
  std::string directory;
  std::unordered_map<tinygltf::Texture const*, Core::GUID> textures;
  std::unordered_map<tinygltf::Material const*, Core::GUID> materials;
  std::vector<std::vector<Math::mat4>> skinning_matrices;
  std::vector<std::vector<int>> skinning_indices;
  std::vector<std::vector<GFX::AnimationComponent::AnimationSampler>> anim_samplers;
  std::vector<std::unordered_map<uint32_t, std::vector<GFX::AnimationComponent::AnimationChannel>>> anim_channels;
  std::unordered_map<int, GFX::GameObjectHandle> node2go;
};

auto loadGLTFAnimation(tinygltf::Model const* model, glTFLoaderEnv& env) {
  env.anim_samplers.resize(model->animations.size());
  env.anim_channels.resize(model->animations.size());
  for (size_t i = 0; i < model->animations.size(); i++) {
    const tinygltf::Animation& animation = model->animations[i];
    // load samplers
    env.anim_samplers[i].resize(animation.samplers.size());
    for (size_t j = 0; j < animation.samplers.size(); j++) {
      GFX::AnimationComponent::AnimationSampler& sampler_comp = env.anim_samplers[i][j];
      const tinygltf::AnimationSampler& sampler = animation.samplers[j];
      { // copy input buffer to sampler
        const tinygltf::Accessor& accessor = model->accessors[sampler.input];
        assert(accessor.type == TINYGLTF_TYPE_SCALAR);
        const tinygltf::BufferView& bufferView = model->bufferViews[accessor.bufferView];
        const tinygltf::Buffer& buffer = model->buffers[bufferView.buffer];
        const float* ptr = reinterpret_cast<const float*>(
            buffer.data.data() + accessor.byteOffset + bufferView.byteOffset);
        sampler_comp.inputs.resize(accessor.count);
        memcpy(sampler_comp.inputs.data(), ptr, accessor.count * sizeof(float));
      }
      {  // copy output buffer to sampler
        const tinygltf::Accessor& accessor = model->accessors[sampler.output];
        const tinygltf::BufferView& bufferView = model->bufferViews[accessor.bufferView];
        const tinygltf::Buffer& buffer = model->buffers[bufferView.buffer];
        const float* ptr = reinterpret_cast<const float*>(
            buffer.data.data() + accessor.byteOffset + bufferView.byteOffset);
        if (accessor.type == TINYGLTF_TYPE_VEC4) {
          sampler_comp.outputsVec4.resize(accessor.count);
          memcpy(sampler_comp.outputsVec4.data(), ptr, accessor.count * sizeof(Math::vec4));
        } else if (accessor.type == TINYGLTF_TYPE_VEC3) {
          sampler_comp.outputsVec3.resize(accessor.count);
          memcpy(sampler_comp.outputsVec3.data(), ptr, accessor.count * sizeof(Math::vec3));
        }
      }
      if (sampler.interpolation == "LINEAR") {
        sampler_comp.interpolation = GFX::AnimationComponent::AnimationSampler::
            InterpolationType::LINEAR;
      } else if (sampler.interpolation == "STEP") {
        sampler_comp.interpolation = GFX::AnimationComponent::AnimationSampler::
            InterpolationType::STEP;
      } else if (sampler.interpolation == "CUBICSPLINE") {
        sampler_comp.interpolation = GFX::AnimationComponent::AnimationSampler::
            InterpolationType::CUBICSPLINE;
      }
    }
    // load channels
    for (size_t j = 0; j < animation.channels.size(); j++) {
      uint32_t const target_node = animation.channels[j].target_node;
      std::vector<GFX::AnimationComponent::AnimationChannel>& channels = env.anim_channels[i][target_node];
      std::string const target = animation.channels[j].target_path;
      GFX::AnimationComponent::AnimationChannel channel;
      channel.samplerIndex = animation.channels[j].sampler;
      if (target.compare("translation") == 0) {
        channel.path = GFX::AnimationComponent::AnimationChannel::PathType::TRANSLATION;
      } else if (target.compare("rotation") == 0) {
        channel.path = GFX::AnimationComponent::AnimationChannel::PathType::ROTATION;
      } else if (target.compare("scale") == 0) {
        channel.path = GFX::AnimationComponent::AnimationChannel::PathType::SCALE;
      }
      channels.emplace_back(channel);
    }
  }
}

auto loadGLTFSkinningMatrices(tinygltf::Model const* model, glTFLoaderEnv& env) {
  env.skinning_matrices.resize(model->skins.size());
  for (size_t s = 0; s < model->skins.size(); s++) {
    const tinygltf::Skin& skin = model->skins[s];
    if (skin.inverseBindMatrices > -1) {
      if (skin.joints.size() > 0) {
        const tinygltf::Accessor& accessor =
            model->accessors[skin.inverseBindMatrices];
        assert(accessor.type == TINYGLTF_TYPE_MAT4);

        const tinygltf::BufferView& bufferView =
            model->bufferViews[accessor.bufferView];

        const tinygltf::Buffer& buffer = model->buffers[bufferView.buffer];

        const float* ptr = reinterpret_cast<const float*>(
            buffer.data.data() + accessor.byteOffset + bufferView.byteOffset);
        std::cout << "count = " << accessor.count << std::endl;

        std::vector<Math::mat4> inverse_bind_matrices(accessor.count);

        for (size_t j = 0; j < skin.joints.size(); j++) {
          Math::mat4 m;
          memcpy(&m, ptr + j * 16, 16 * sizeof(float));
          inverse_bind_matrices[j] = m;
        }

        env.skinning_matrices[s] = inverse_bind_matrices;
        // also get the joints indices
        if (env.skinning_indices.size() <= s + 1) {
          env.skinning_indices.resize(s + 1);
        }
        env.skinning_indices[s] = skin.joints;
      }
    }
  }
}

auto loadGLTFMaterialTextures(tinygltf::Texture const* gltexture, tinygltf::Model const* model,
                              glTFLoaderEnv& env, GFX::Scene& gfxscene,
                              MeshLoaderConfig meshConfig = {}) noexcept
    -> Core::GUID {
  if (env.textures.find(gltexture) != env.textures.end()) {
    return env.textures[gltexture];
  }

  tinygltf::Image glimage = model->images[gltexture->source];
  std::string tex_path = env.directory + "\\" + glimage.uri;
  if (!std::filesystem::exists(tex_path) ||
      std::filesystem::is_directory(tex_path)) {
    if (glimage.image.size() > 0) {
      // inlined image, just save it to local
      tex_path = env.directory + "\\" + std::to_string(gltexture->source) + ".png";
      Image::PNG::writePNG(tex_path, glimage.width, glimage.height,
                           glimage.component, (float*)glimage.image.data());
    } else {
      Core::LogManager::Error("GFX :: gltf loader :: Cannot find image properly!");
    }
  }
  Core::GUID guid =
      GFX::GFXManager::get()->registerTextureResource(tex_path.c_str());

  env.textures[gltexture] = guid;
  return guid;
}

auto loadGLTFMaterial(tinygltf::Material const* glmaterial, tinygltf::Model const* model,
                      glTFLoaderEnv& env, GFX::Scene& gfxscene,
                      MeshLoaderConfig meshConfig = {}) noexcept
    -> Core::GUID {
  if (env.materials.find(glmaterial) != env.materials.end()) {
    return env.materials[glmaterial];
  }

  std::string name = glmaterial->name;
  GFX::Material gfxmat;
  gfxmat.ORID = Core::requestORID();
  if (name == "") name = std::to_string(gfxmat.ORID);
  gfxmat.path = "content/materials/" +
                Core::StringHelper::invalidFileFolderName(name) + ".mat";
  gfxmat.BxDF = 0;
  gfxmat.name = name;

  auto to_sampler = [&](int sampler_idx) {
    tinygltf::Sampler const& sampler = model->samplers[sampler_idx];
    RHI::SamplerDescriptor desc;
    // Min and Mipmap filter
    if (sampler.minFilter == TINYGLTF_TEXTURE_FILTER_NEAREST) {
      desc.minFilter = RHI::FilterMode::NEAREST;
    } else if (sampler.minFilter == TINYGLTF_TEXTURE_FILTER_LINEAR) {
      desc.minFilter = RHI::FilterMode::LINEAR;
    } else if (sampler.minFilter ==
               TINYGLTF_TEXTURE_FILTER_NEAREST_MIPMAP_NEAREST) {
      desc.minFilter = RHI::FilterMode::NEAREST;
      desc.mipmapFilter = RHI::MipmapFilterMode::NEAREST;
    } else if (sampler.minFilter ==
               TINYGLTF_TEXTURE_FILTER_LINEAR_MIPMAP_NEAREST) {
      desc.minFilter = RHI::FilterMode::LINEAR;
      desc.mipmapFilter = RHI::MipmapFilterMode::NEAREST;
    } else if (sampler.minFilter ==
               TINYGLTF_TEXTURE_FILTER_NEAREST_MIPMAP_LINEAR) {
      desc.minFilter = RHI::FilterMode::NEAREST;
      desc.mipmapFilter = RHI::MipmapFilterMode::LINEAR;
    } else if (sampler.minFilter ==
               TINYGLTF_TEXTURE_FILTER_LINEAR_MIPMAP_LINEAR) {
      desc.minFilter = RHI::FilterMode::LINEAR;
      desc.mipmapFilter = RHI::MipmapFilterMode::LINEAR;
    }
    // Mag filter
    if (sampler.magFilter == TINYGLTF_TEXTURE_FILTER_NEAREST) {
      desc.magFilter = RHI::FilterMode::NEAREST;
    } else if (sampler.magFilter == TINYGLTF_TEXTURE_FILTER_LINEAR) {
      desc.magFilter = RHI::FilterMode::LINEAR;
    }
    // WarpS
    if (sampler.wrapS == TINYGLTF_TEXTURE_WRAP_REPEAT) {
      desc.addressModeU = RHI::AddressMode::REPEAT;
    } else if (sampler.wrapS == TINYGLTF_TEXTURE_WRAP_CLAMP_TO_EDGE) {
      desc.addressModeU = RHI::AddressMode::CLAMP_TO_EDGE;
    } else if (sampler.wrapS == TINYGLTF_TEXTURE_WRAP_MIRRORED_REPEAT) {
      desc.addressModeU = RHI::AddressMode::MIRROR_REPEAT;
    }
    // WarpT
    if (sampler.wrapT == TINYGLTF_TEXTURE_WRAP_REPEAT) {
      desc.addressModeV = RHI::AddressMode::REPEAT;
    } else if (sampler.wrapT == TINYGLTF_TEXTURE_WRAP_CLAMP_TO_EDGE) {
      desc.addressModeV = RHI::AddressMode::CLAMP_TO_EDGE;
    } else if (sampler.wrapT == TINYGLTF_TEXTURE_WRAP_MIRRORED_REPEAT) {
      desc.addressModeV = RHI::AddressMode::MIRROR_REPEAT;
    }
    return desc;
  };

  // load diffuse information
  { // load diffuse texture
    if (glmaterial->pbrMetallicRoughness.baseColorTexture.index != -1) {
      tinygltf::Texture const& texture = model->textures[glmaterial->pbrMetallicRoughness.baseColorTexture.index];
      Core::GUID texBasecolor = loadGLTFMaterialTextures(&texture, model, env, gfxscene, meshConfig);
      gfxmat.textures["base_color"] = GFX::Material::TextureEntry{texBasecolor, 0, to_sampler(texture.sampler)};
    }
  }
  { // load diffuse color
    if (glmaterial->pbrMetallicRoughness.baseColorFactor.size() > 0) {
      gfxmat.baseOrDiffuseColor = Math::vec3{
        (float)glmaterial->pbrMetallicRoughness.baseColorFactor[0],
        (float)glmaterial->pbrMetallicRoughness.baseColorFactor[1],
        (float)glmaterial->pbrMetallicRoughness.baseColorFactor[2],
      };
      gfxmat.roughness = (float)glmaterial->pbrMetallicRoughness.roughnessFactor;
      gfxmat.metalness = (float)glmaterial->pbrMetallicRoughness.metallicFactor;
    }
  }
  gfxmat.serialize();
  Core::GUID matID =
      GFX::GFXManager::get()->registerMaterialResource(gfxmat.path.c_str());
  env.materials[glmaterial] = matID;
  return matID;
}

 static inline auto loadGLTFMesh(tinygltf::Mesh const& gltfmesh,
                                GameObjectHandle const& gfxNode, int node_id,
                                tinygltf::Model const* model,
                                glTFLoaderEnv& env, GFX::Scene& gfxscene,
                                MeshLoaderConfig meshConfig = {}) noexcept
    -> Core::GUID {
#define INDEX_TYPE uint32_t
  // Load meshes into Runtime resource managers.
  RHI::Device* device = GFX::GFXManager::get()->rhiLayer->getDevice();
  std::vector<INDEX_TYPE> indexBuffer_uint = {};
  std::vector<float> vertexBuffer = {};
  std::vector<float> PositionBuffer = {};
  std::vector<uint64_t> JointIndexBuffer = {};
  std::vector<float> JointweightsBuffer = {};
  // Create GFX mesh, and add it to resource manager
  GFX::Mesh mesh;
  uint32_t submesh_index_offset = 0;
  uint32_t submesh_vertex_offset = 0;
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
                  switch (attribAccessor.componentType) {
                    case TINYGLTF_COMPONENT_TYPE_FLOAT:
                      // 3D vector of float
                      ArrayAdapter<Math::vec3> positions(dataPtr, count,
                                                         byte_stride);
                      for (size_t i{0}; i < indexArray_uint.size() / 3; ++i) {
                        // get the i'th triange's indexes
                        auto f0 = indexArray_uint[3 * i + 0];
                        auto f1 = indexArray_uint[3 * i + 1];
                        auto f2 = indexArray_uint[3 * i + 2];
                        // get the 3 normal vectors for that face
                        Math::vec3 p0, p1, p2;
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
                        ArrayAdapter<Math::dvec3> positions(dataPtr, count,
                                                            byte_stride);
                        for (size_t i{0}; i < indexArray_uint.size() / 3; ++i) {
                          // get the i'th triange's indexes
                          auto f0 = indexArray_uint[3 * i + 0];
                          auto f1 = indexArray_uint[3 * i + 1];
                          auto f2 = indexArray_uint[3 * i + 2];
                          // get the 3 normal vectors for that face
                          Math::dvec3 p0, p1, p2;
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
                      ArrayAdapter<Math::vec3> normals(dataPtr, count,
                                                       byte_stride);
                      // For each triangle :
                      for (size_t i{0}; i < indexArray_uint.size() / 3; ++i) {
                        // get the i'th triange's indexes
                        auto f0 = indexArray_uint[3 * i + 0];
                        auto f1 = indexArray_uint[3 * i + 1];
                        auto f2 = indexArray_uint[3 * i + 2];
                        // get the 3 normal vectors for that face
                        Math::vec3 n0, n1, n2;
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
                      ArrayAdapter<Math::dvec3> normals(dataPtr, count,
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
                        Math::dvec3 n0, n1, n2;
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
                      ArrayAdapter<Math::vec2> uvs(dataPtr, count, byte_stride);
                      for (size_t i{0}; i < indexArray_uint.size() / 3; ++i) {
                        // get the i'th triange's indexes
                        auto f0 = indexArray_uint[3 * i + 0];
                        auto f1 = indexArray_uint[3 * i + 1];
                        auto f2 = indexArray_uint[3 * i + 2];
                        // get the texture coordinates for each triangle's
                        // vertices
                        Math::vec2 uv0, uv1, uv2;
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
                      ArrayAdapter<Math::dvec2> uvs(dataPtr, count,
                                                    byte_stride);
                      for (size_t i{0}; i < indexArray_uint.size() / 3; ++i) {
                        // get the i'th triange's indexes
                        auto f0 = indexArray_uint[3 * i + 0];
                        auto f1 = indexArray_uint[3 * i + 1];
                        auto f2 = indexArray_uint[3 * i + 2];

                        Math::dvec2 uv0, uv1, uv2;
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
                      Core::LogManager::Error(
                          "GFX :: tinygltf :: unrecognized vector type for "
                          "UV");
                  }
                } break;
                default:
                  Core::LogManager::Error(
                      "GFX :: tinygltf :: unreconized componant type for UV");
              }
            }
            if (attribute.first == "TANGENT") {
              switch (attribAccessor.type) {
                case TINYGLTF_TYPE_VEC3: {
                  switch (attribAccessor.componentType) {
                    case TINYGLTF_COMPONENT_TYPE_FLOAT: {
                      ArrayAdapter<Math::vec3> tangents(dataPtr, count,
                                                       byte_stride);
                      // For each triangle :
                      for (size_t i{0}; i < indexArray_uint.size() / 3; ++i) {
                        // get the i'th triange's indexes
                        auto f0 = indexArray_uint[3 * i + 0];
                        auto f1 = indexArray_uint[3 * i + 1];
                        auto f2 = indexArray_uint[3 * i + 2];
                        // get the 3 normal vectors for that face
                        Math::vec3 n0, n1, n2;
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
                      ArrayAdapter<Math::dvec3> tangents(dataPtr, count,
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
                        Math::dvec3 n0, n1, n2;
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
                      ArrayAdapter<Math::Vector4<uint16_t>> joints(dataPtr, count, byte_stride);
                      // For each triangle :
                      for (size_t i{0}; i < indexArray_uint.size() / 3; ++i) {
                        // get the i'th triange's indexes
                        auto f0 = indexArray_uint[3 * i + 0];
                        auto f1 = indexArray_uint[3 * i + 1];
                        auto f2 = indexArray_uint[3 * i + 2];
                        // get the 3 normal vectors for that face
                        Math::Vector4<uint16_t> j0, j1, j2;
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
                      ArrayAdapter<Math::vec4> weights(dataPtr, count, byte_stride);
                      // For each triangle :
                      for (size_t i{0}; i < indexArray_uint.size() / 3; ++i) {
                        // get the i'th triange's indexes
                        auto f0 = indexArray_uint[3 * i + 0];
                        auto f1 = indexArray_uint[3 * i + 1];
                        auto f2 = indexArray_uint[3 * i + 2];
                        // get the 3 normal vectors for that face
                        Math::vec4 w0, w1, w2;
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
                      ArrayAdapter<Math::dvec4> tangents(dataPtr, count, byte_stride);
                      // IMPORTANT: We need to reorder normals (and texture
                      // coordinates into "facevarying" order) for each face
                      // For each triangle :
                      for (size_t i{0}; i < indexArray_uint.size() / 3; ++i) {
                        // get the i'th triange's indexes
                        auto f0 = indexArray_uint[3 * i + 0];
                        auto f1 = indexArray_uint[3 * i + 1];
                        auto f2 = indexArray_uint[3 * i + 2];
                        // get the 3 normal vectors for that face
                        Math::dvec4 w0, w1, w2;
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
        Core::LogManager::Error(
            "GFX :: tinygltf :: primitive mode not implemented");
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
            Math::vec3 pos1 = {vertexBuffer_positionOnly[i0 * 3 + 0],
                               vertexBuffer_positionOnly[i0 * 3 + 1],
                               vertexBuffer_positionOnly[i0 * 3 + 2]};
            Math::vec3 pos2 = {vertexBuffer_positionOnly[i1 * 3 + 0],
                               vertexBuffer_positionOnly[i1 * 3 + 1],
                               vertexBuffer_positionOnly[i1 * 3 + 2]};
            Math::vec3 pos3 = {vertexBuffer_positionOnly[i2 * 3 + 0],
                               vertexBuffer_positionOnly[i2 * 3 + 1],
                               vertexBuffer_positionOnly[i2 * 3 + 2]};
            Math::vec2 uv1  = {vertexBuffer_uvOnly[i0 * 2 + 0],
                               vertexBuffer_uvOnly[i0 * 2 + 1]};
            Math::vec2 uv2  = {vertexBuffer_uvOnly[i1 * 2 + 0],
                               vertexBuffer_uvOnly[i1 * 2 + 1]};
            Math::vec2 uv3  = {vertexBuffer_uvOnly[i2 * 2 + 0],
                               vertexBuffer_uvOnly[i2 * 2 + 1]};
            Math::vec3 tangent;
            Math::vec3 edge1 = pos2 - pos1;
            Math::vec3 edge2 = pos3 - pos1;
            Math::vec2 deltaUV1 = uv2 - uv1;
            Math::vec2 deltaUV2 = uv3 - uv1;
            float f = 1.0f / (deltaUV1.x * deltaUV2.y - deltaUV2.x * deltaUV1.y);
            tangent.x = f * (deltaUV2.y * edge1.x - deltaUV1.y * edge2.x);
            tangent.y = f * (deltaUV2.y * edge1.y - deltaUV1.y * edge2.y);
            tangent.z = f * (deltaUV2.y * edge1.z - deltaUV1.y * edge2.z);
            tangent = Math::normalize(tangent);
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
      for (auto const& entry : meshConfig.layout.layout) {
        // vertex position
        if (entry.info == MeshDataLayout::VertexInfo::POSITION) {
            if (entry.format == RHI::VertexFormat::FLOAT32X3) {
              vertexBuffer.push_back(vertexBuffer_positionOnly[i * 3 + 0]);
              vertexBuffer.push_back(vertexBuffer_positionOnly[i * 3 + 1]);
              vertexBuffer.push_back(vertexBuffer_positionOnly[i * 3 + 2]);
              if (meshConfig.usePositionBuffer) {
                if (vertexBuffer_positionOnly.size() != 0) {
                    PositionBuffer.push_back(vertexBuffer_positionOnly[i * 3 + 0]);
                    PositionBuffer.push_back(vertexBuffer_positionOnly[i * 3 + 1]);
                    PositionBuffer.push_back(vertexBuffer_positionOnly[i * 3 + 2]);
                } else {
                
                }
              }
            } else {
              Core::LogManager::Error(
                  "GFX :: SceneNodeLoader_assimp :: unwanted vertex format for "
                  "POSITION attributes.");
              return Core::INVALID_GUID;
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
              vertexBuffer.push_back(vertexBuffer_uvOnly[i * 2 + 0]);
              vertexBuffer.push_back(vertexBuffer_uvOnly[i * 2 + 1]);            
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

    mesh.submeshes.push_back(GFX::Mesh::Submesh{
        submesh_index_offset, uint32_t(indexArray_uint.size()),
        submesh_vertex_offset, uint32_t(meshPrimitive.material)});
    submesh_index_offset = indexBuffer_uint.size();
    submesh_vertex_offset = PositionBuffer.size() / 3;
  }
  // create mesh resource
  {  // register mesh
    mesh.vertexBufferLayout = getVertexBufferLayout(meshConfig.layout);
    if (meshConfig.residentOnHost) {
      mesh.vertexBuffer_host =
          Core::Buffer(sizeof(float) * vertexBuffer.size());
      memcpy(mesh.vertexBuffer_host.data, vertexBuffer.data(),
             mesh.vertexBuffer_host.size);
      mesh.vertexBufferInfo.onHost = true;
      mesh.vertexBufferInfo.size = mesh.vertexBuffer_host.size;
      if (meshConfig.layout.format == RHI::IndexFormat::UINT16_t) {
        mesh.indexBuffer_host =
            Core::Buffer(sizeof(uint16_t) * indexBuffer_uint.size());
        memcpy(mesh.indexBuffer_host.data, indexBuffer_uint.data(),
               mesh.indexBuffer_host.size);
        mesh.indexBufferInfo.size = mesh.indexBuffer_host.size;
        mesh.indexBufferInfo.onHost = true;
      } else if (meshConfig.layout.format == RHI::IndexFormat::UINT32_T) {
        mesh.indexBuffer_host =
            Core::Buffer(sizeof(uint32_t) * indexBuffer_uint.size());
        memcpy(mesh.indexBuffer_host.data, indexBuffer_uint.data(),
               mesh.indexBuffer_host.size);
        mesh.indexBufferInfo.size = mesh.indexBuffer_host.size;
        mesh.indexBufferInfo.onHost = true;
      }
      if (meshConfig.usePositionBuffer) {
        mesh.positionBuffer_host =
            Core::Buffer(sizeof(float) * PositionBuffer.size());
        memcpy(mesh.positionBuffer_host.data, PositionBuffer.data(),
               mesh.positionBuffer_host.size);
        mesh.positionBufferInfo.onHost = true;
        mesh.positionBufferInfo.size = mesh.positionBuffer_host.size;
      }
      if (true) {
        mesh.jointIndexBuffer_host = Core::Buffer(sizeof(uint64_t) * JointIndexBuffer.size());
        memcpy(mesh.jointIndexBuffer_host.data, JointIndexBuffer.data(), mesh.jointIndexBuffer_host.size);
        mesh.jointIndexBufferInfo.onHost = true;
        mesh.jointIndexBufferInfo.size = mesh.jointIndexBuffer_host.size;

        mesh.jointWeightBuffer_host = Core::Buffer(sizeof(float) * JointweightsBuffer.size());
        memcpy(mesh.jointWeightBuffer_host.data, JointweightsBuffer.data(), mesh.jointWeightBuffer_host.size);
        mesh.jointWeightBufferInfo.onHost = true;
        mesh.jointWeightBufferInfo.size = mesh.jointWeightBuffer_host.size;
      }
    }
  }

  Core::GUID guid =
      Core::ResourceManager::get()->requestRuntimeGUID<GFX::Mesh>();
  Core::ORID orid = Core::requestORID();
  Core::ResourceManager::get()->database.registerResource(orid, guid);
  Core::ResourceManager::get()->addResource<GFX::Mesh>(guid,
                                                       std::move(mesh));
  GFX::Mesh* meshResourceRef =
      Core::ResourceManager::get()->getResource<GFX::Mesh>(guid);
  meshResourceRef->ORID = orid;
  meshResourceRef->serialize();

  return guid;
 }
 
void DecomposeMatrixToComponents(Math::mat4 matrix, Math::vec3& translation,
                                 Math::vec3& rotation, Math::vec3& scale) {
  scale[0] = Math::length(Math::vec3(matrix.data[0][0], matrix.data[1][0], matrix.data[2][0]));
  scale[1] = Math::length(Math::vec3(matrix.data[0][1], matrix.data[1][1], matrix.data[2][1]));
  scale[2] = Math::length(Math::vec3(matrix.data[0][2], matrix.data[1][2], matrix.data[2][2]));
  
  matrix.data[0][0] /= scale[0]; matrix.data[1][0] /= scale[0]; matrix.data[2][0] /= scale[0];
  matrix.data[0][1] /= scale[1]; matrix.data[1][1] /= scale[1]; matrix.data[2][1] /= scale[1];
  matrix.data[0][2] /= scale[2]; matrix.data[1][2] /= scale[2]; matrix.data[2][2] /= scale[2];

  const float RAD2DEG = 180.f / Math::float_Pi;
  rotation[0] = RAD2DEG * atan2f(matrix.data[1][2], matrix.data[2][2]);
  rotation[1] = RAD2DEG * atan2f(-matrix.data[0][2],
                sqrtf(matrix.data[1][2] * matrix.data[1][2] + matrix.data[2][2] * matrix.data[2][2]));
  rotation[2] = RAD2DEG * atan2f(matrix.data[0][1], matrix.data[0][0]);

  translation[0] = matrix.data[0][3];
  translation[1] = matrix.data[1][3];
  translation[2] = matrix.data[2][3];
}

 static inline auto processGLTFTransform(GameObjectHandle const& gfxNode,
    int node_id, tinygltf::Model const* model, glTFLoaderEnv& env, GFX::Scene& gfxscene,
    MeshLoaderConfig meshConfig = {}) noexcept -> void {
  tinygltf::Node const& node = model->nodes[node_id];
  gfxscene.getGameObject(gfxNode)->getEntity().getComponent<TagComponent>()
      ->name = (node.name == "") ? "nameless" : std::string(node.name);
  TransformComponent* transform = gfxscene.getGameObject(gfxNode)
    ->getEntity().getComponent<TransformComponent>();
  if (node.scale.size() == 3)
    transform->scale = {static_cast<float>(node.scale[0]),
                        static_cast<float>(node.scale[1]),
                        static_cast<float>(node.scale[2])};
  if (node.translation.size() == 3)
    transform->translation = {static_cast<float>(node.translation[0]),
                              static_cast<float>(node.translation[1]),
                              static_cast<float>(node.translation[2])};
  if (node.rotation.size() == 4) {
    double qx = node.rotation[0];
    double qy = node.rotation[1];
    double qz = node.rotation[2];
    double qw = node.rotation[3];
    double roll = atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy));
    double pitch = -std::asin(2 * (qw * qy - qz * qx));
    double yaw = atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz));
    roll *= 180. / Math::double_Pi;
    pitch *= 180. / Math::double_Pi;
    yaw *= 180. / Math::double_Pi;
    transform->eulerAngles = {static_cast<float>(roll),
                              static_cast<float>(pitch),
                              static_cast<float>(yaw)};
  }
  if (node.matrix.size() == 16) {
    Math::mat4 mat = Math::mat4 {
      (float)node.matrix[0], (float)node.matrix[1], (float)node.matrix[2],  (float)node.matrix[3],
      (float)node.matrix[4], (float)node.matrix[5], (float)node.matrix[6],  (float)node.matrix[7],
      (float)node.matrix[8], (float)node.matrix[9], (float)node.matrix[10], (float)node.matrix[11],
      (float)node.matrix[12],(float)node.matrix[13],(float)node.matrix[14], (float)node.matrix[15],
    };
    //DecomposeMatrixToComponents(mat, transform->translation,
    //                            transform->eulerAngles, transform->scale);
  }
 }

 static inline auto processGLTFMesh(GameObjectHandle const& gfxNode, int node_id,
                                   tinygltf::Model const* model,
                                   glTFLoaderEnv& env, GFX::Scene& gfxscene,
                                   MeshLoaderConfig meshConfig = {}) noexcept
    -> void {
  tinygltf::Node const& node = model->nodes[node_id];
  if (node.mesh == -1) return;

  Core::GUID meshGUID = loadGLTFMesh(model->meshes[node.mesh], gfxNode, node_id,
                                     model, env, gfxscene, meshConfig);
  gfxscene.getGameObject(gfxNode)->getEntity().addComponent<MeshReference>();
  GFX::Mesh* meshResourceRef = Core::ResourceManager::get()->getResource<GFX::Mesh>(meshGUID);
  meshResourceRef->serialize();

  if (node.skin != -1 && meshResourceRef->jointIndexBuffer_host.size > 0) {
    std::span<uint64_t> indices(
        (uint64_t*)meshResourceRef->jointIndexBuffer_host.data,
        meshResourceRef->jointIndexBuffer_host.size / sizeof(uint64_t));
    std::vector<int>& map = env.skinning_indices[node.skin];
    for (int i = 0; i < indices.size(); ++i) {
      indices[i] = map[indices[i]];
    }
  }

  // bind scene
  gfxscene.getGameObject(gfxNode)
      ->getEntity()
      .addComponent<MeshReference>()
      ->mesh = meshResourceRef;

  gfxscene.getGameObject(gfxNode)->getEntity().addComponent<MeshRenderer>();
   MeshRenderer* meshRenderer =
      gfxscene.getGameObject(gfxNode)->getEntity().getComponent<MeshRenderer>();

   for (auto const& meshPrimitive : model->meshes[node.mesh].primitives) {
    int matID = meshPrimitive.material;
    Core::GUID matGUID;
    if (matID == -1) {
      matGUID = GFX::GFXManager::get()->registerMaterialResource("WhiteBSDF.mat");
    } else
      matGUID = loadGLTFMaterial(&model->materials[matID], model, env, gfxscene, meshConfig);
    meshRenderer->materials.push_back(
        Core::ResourceManager::get()->getResource<GFX::Material>(matGUID));
  }
}
 
 static inline auto processGLTFAnimation(GameObjectHandle const& gfxNode, int node_id,
    tinygltf::Model const* model, glTFLoaderEnv& env, GFX::Scene& gfxscene,
    MeshLoaderConfig meshConfig = {}) noexcept -> void {
  if (env.anim_channels.size() == 0) return;
  size_t i = 0;
  {
    auto iter = env.anim_channels[i].find(node_id);
    if (iter != env.anim_channels[i].end()) {
      AnimationComponent* meshResourceRef = gfxscene.getGameObject(gfxNode)
        ->getEntity().addComponent<AnimationComponent>();
      std::vector<AnimationComponent::AnimationChannel>& channels = iter->second;
      for (auto channel : channels) {
        meshResourceRef->ani.samplers.push_back(env.anim_samplers[i][channel.samplerIndex]);
        channel.samplerIndex = meshResourceRef->ani.samplers.size() - 1;
        meshResourceRef->ani.channels.push_back(channel);
      }
    }
  }
 }

void QuatToAngleAxis(const std::vector<double> quaternion,
                     double& outAngleDegrees, double* axis) {
  double qx = quaternion[0];
  double qy = quaternion[1];
  double qz = quaternion[2];
  double qw = quaternion[3];

  double angleRadians = 2 * acos(qw);
  if (angleRadians == 0.0) {
    outAngleDegrees = 0.0;
    axis[0] = 0.0;
    axis[1] = 0.0;
    axis[2] = 1.0;
    return;
  }

  double denom = sqrt(1 - qw * qw);
  outAngleDegrees = angleRadians * 180.0 / 3.1415926;
  axis[0] = qx / denom;
  axis[1] = qy / denom;
  axis[2] = qz / denom;
}

static inline auto processGLTFNode(GameObjectHandle const& gfxNode, int node_id,
    tinygltf::Model const* model, glTFLoaderEnv& env, GFX::Scene& gfxscene,
    MeshLoaderConfig meshConfig = {}) noexcept -> void {
  // load tag, transform, mesh
  env.node2go[node_id] = gfxNode;
  processGLTFTransform(gfxNode, node_id, model, env, gfxscene, meshConfig);
  processGLTFMesh(gfxNode, node_id, model, env, gfxscene, meshConfig);
  processGLTFAnimation(gfxNode, node_id, model, env, gfxscene, meshConfig);
  
  // process the meshes for all the following nodes
  tinygltf::Node const& node = model->nodes[node_id];
  for (uint32_t i : node.children) {
    GameObjectHandle subNode = gfxscene.createGameObject(gfxNode);
    processGLTFNode(subNode, i, model, env, gfxscene, meshConfig);
  }
}

/** Load glTF file */
auto SceneNodeLoader_glTF::loadSceneNode(
    std::filesystem::path const& path, GFX::Scene& gfxscene,
    MeshLoaderConfig meshConfig) noexcept
    -> void {
  // use tinygltf to load file
  tinygltf::Model model;
  tinygltf::TinyGLTF loader;
  std::string err;
  std::string warn;
  std::string ext = path.extension().string();
  // load the .glb/.gltf file
  bool ret = false;
  if (ext.compare(".glb") == 0) { // assume binary glTF.
    ret = loader.LoadBinaryFromFile(&model, &err, &warn, path.string());  
  } else { // assume ascii glTF.
    ret = loader.LoadASCIIFromFile(&model, &err, &warn, path.string());  
  }
  // check the loading results
  if (!warn.empty())
    Core::LogManager::Warning(
        std::format("GFX :: tinygltf :: {0}", warn.c_str()));
  if (!err.empty())
    Core::LogManager::Error(std::format("GFX :: tinygltf :: {0}", err.c_str()));
  if (!ret) {
    Core::LogManager::Error("GFX :: tinygltf :: Failed to parse glTF");
    return;
  }
  // Find the default or first scene.
  if (model.scenes.empty()) {
    std::cerr << "GFX :: tinygltf :: Scene is empty" << std::endl;
    return; }
  std::cout << "defaultScene = " << model.defaultScene << std::endl;
  if (model.defaultScene >= int(model.scenes.size())) {
    std::cerr << "GFX :: tinygltf :: Invalid defualtScene value : " 
        << model.defaultScene << std::endl;
    return; }
  int scene_idx = model.defaultScene;
  if (scene_idx == -1) {
    // Use the first scene.
    scene_idx = 0;
  }
  auto& scene = model.scenes[scene_idx];

  std::string directory = path.parent_path().string();
  glTFLoaderEnv env;
  env.directory = directory;  
  // try load animation data
  loadGLTFAnimation(&model, env);
  // try load skinning data
  loadGLTFSkinningMatrices(&model, env);

  // create root node for the whole gltf file
  GameObjectHandle rootNode = gfxscene.createGameObject(GFX::NULL_GO);
  gfxscene.getGameObject(rootNode)->getEntity()
    .getComponent<TagComponent>()->name = path.filename().string();
  // load all root nodes for the default scene
  for (auto node : scene.nodes) {
    GameObjectHandle subNode = gfxscene.createGameObject(rootNode);
    processGLTFNode(subNode, node, &model, env, gfxscene, meshConfig);
  }

  for (int i = 0; i < env.skinning_indices.size(); ++i) {
    auto& skin_indices = env.skinning_indices[i];
    auto& skin_matrices = env.skinning_matrices[i];
    for (int j = 0; j < skin_indices.size(); ++j) {
      int const index = skin_indices[j];
      GFX::GameObjectHandle go = env.node2go[index];
      TransformComponent* trans = gfxscene.getGameObject(go)
        ->getEntity().getComponent<TransformComponent>();
      trans->flag |= (uint32_t)TransformComponent::FlagBit::IS_SKELETON_JOINT;
      trans->inverseJointTransform = skin_matrices[j];
    }
  }

  // post-process all nodes
  for (auto pair : env.node2go) {
    MeshReference* meshref = gfxscene.getGameObject(pair.second)
        ->getEntity().getComponent<MeshReference>();
    if (meshref) {
      // if has joint index buffer, assign the real handle
      if (meshref->mesh->jointIndexBufferInfo.size > 0) {
        uint64_t* indices = (uint64_t*)meshref->mesh->jointIndexBuffer_host.data;
        size_t size = meshref->mesh->jointIndexBuffer_host.size / sizeof(uint64_t);
        for (size_t i = 0; i < size; ++i) {
          indices[i] = env.node2go[indices[i]];
        }
      }
    }
  }
}

struct AssimpLoaderEnv {
  std::string directory;
  std::unordered_map<std::string, Core::GUID> textures;
  std::unordered_map<aiMaterial*, Core::GUID> materials;
  std::unordered_map<std::string, GFX::AnimationComponent> animation_map;
};

auto loadAssimpAnimation(aiScene const* scene, AssimpLoaderEnv& env) {
  if (scene->mNumAnimations == 0) return;
  aiAnimation const* aianimation = scene->mAnimations[0];
  //env.anim_samplers.resize(aianimation->mNumChannels);
  //env.anim_channels.resize(aianimation->mNumChannels);
  for (size_t i = 0; i < aianimation->mNumChannels; i++) {
    aiNodeAnim const* channel = aianimation->mChannels[i];
    std::string channel_name = channel->mNodeName.data;
    auto iter = env.animation_map.find(channel_name);
    if (iter == env.animation_map.end()) {
      env.animation_map[channel_name] = {};
      iter = env.animation_map.find(channel_name);
    }
    auto& ani = iter->second.ani;
    ani.name = channel_name;
    if (channel->mNumPositionKeys != 0) {
      GFX::AnimationComponent::AnimationSampler sampler;
      sampler.interpolation = GFX::AnimationComponent::AnimationSampler::InterpolationType::LINEAR;
      for (size_t j = 0; j < channel->mNumPositionKeys; ++j) {
        aiVectorKey const& key = channel->mPositionKeys[j];
        sampler.inputs.push_back(key.mTime / aianimation->mTicksPerSecond);
        sampler.outputsVec3.push_back(
          Math::vec3{key.mValue.x, key.mValue.y, key.mValue.z});
      }
      ani.channels.push_back(GFX::AnimationComponent::AnimationChannel{
          GFX::AnimationComponent::AnimationChannel::PathType::TRANSLATION,
          uint32_t(ani.samplers.size())});
      ani.samplers.push_back(sampler);
    }
    if (channel->mNumRotationKeys != 0) {
      GFX::AnimationComponent::AnimationSampler sampler;
      sampler.interpolation = GFX::AnimationComponent::AnimationSampler::InterpolationType::LINEAR;
      for (size_t j = 0; j < channel->mNumRotationKeys; ++j) {
        aiQuatKey const& key = channel->mRotationKeys[j];
        sampler.inputs.push_back(key.mTime / aianimation->mTicksPerSecond);
        sampler.outputsVec4.push_back(
          Math::vec4{key.mValue.x, key.mValue.y, key.mValue.z, key.mValue.w});
      }
      ani.channels.push_back(GFX::AnimationComponent::AnimationChannel{
          GFX::AnimationComponent::AnimationChannel::PathType::ROTATION,
          uint32_t(ani.samplers.size())});
      ani.samplers.push_back(sampler);
    }
    if (channel->mNumScalingKeys != 0) {
      GFX::AnimationComponent::AnimationSampler sampler;
      sampler.interpolation = GFX::AnimationComponent::AnimationSampler::InterpolationType::LINEAR;
      for (size_t j = 0; j < channel->mNumScalingKeys; ++j) {
        aiVectorKey const& key = channel->mScalingKeys[j];
        sampler.inputs.push_back(key.mTime / aianimation->mTicksPerSecond);
        sampler.outputsVec3.push_back(
          Math::vec3{key.mValue.x, key.mValue.y, key.mValue.z});
      }
      ani.channels.push_back(GFX::AnimationComponent::AnimationChannel{
          GFX::AnimationComponent::AnimationChannel::PathType::SCALE,
          uint32_t(ani.samplers.size())});
      ani.samplers.push_back(sampler);
    }
  }
}


auto loadMaterialTextures(aiMaterial* mat, aiTextureType type,
                          AssimpLoaderEnv& env) noexcept
    -> std::vector<Core::GUID> {
  std::vector<Core::GUID> textures;
  for (unsigned int i = 0; i < mat->GetTextureCount(type); i++) {
    aiString str;
    mat->GetTexture(type, i, &str);
    std::string const& name = std::string(str.C_Str());
    auto iter = env.textures.find(name);
    if (iter != env.textures.end()) {
      textures.push_back(iter->second);
    } else {
      std::string tex_path = env.directory + "\\" + str.C_Str();
      Core::GUID guid =
          GFX::GFXManager::get()->registerTextureResource(tex_path.c_str());
      env.textures[name] = guid;
      textures.push_back(guid);
    }
  }
  return textures;
}

auto loadMaterial(aiMaterial* material, AssimpLoaderEnv& env) noexcept
    -> Core::GUID {
  auto iter = env.materials.find(material);
  if (iter != env.materials.end()) {
    return iter->second;
  }

  std::string name = std::string(material->GetName().C_Str());
  GFX::Material gfxmat;
  gfxmat.path = "content/materials/" +
                Core::StringHelper::invalidFileFolderName(name) + ".mat";
  gfxmat.ORID = Core::requestORID();
  gfxmat.BxDF = 0;

  gfxmat.name = name;
  // load diffuse information
  int shading_model = 0;
  material->Get(AI_MATKEY_SHADING_MODEL, shading_model);
  { // load diffuse texture
    std::vector<Core::GUID> diffuseMaps =
        loadMaterialTextures(material, aiTextureType_DIFFUSE, env);
    aiColor3D diffuse_color(0.f, 0.f, 0.f);
    material->Get(AI_MATKEY_COLOR_DIFFUSE, diffuse_color);
    gfxmat.baseOrDiffuseColor = {diffuse_color.r, diffuse_color.g, diffuse_color.b};
    if (diffuseMaps.size() != 1) {
      Core::LogManager::Error(
          "GFX :: SceneNodeLoader_assimp :: diffuse map number is not 1.");
    } else {
      gfxmat.textures["base_color"] =
          GFX::Material::TextureEntry{diffuseMaps[0], 0};
    }
  }
  { // load normal texture
    std::vector<Core::GUID> normalMaps =
        loadMaterialTextures(material, aiTextureType_NORMALS, env);
    if (normalMaps.size() != 1) {
      // No normal map
      //Core::LogManager::Error(
      //    "GFX :: SceneNodeLoader_assimp :: roughness map number is not 1.");
    } else {
      gfxmat.textures["normal_bump"] =
          GFX::Material::TextureEntry{normalMaps[0], 0};
    }
  }
  { // load roughness texture
    std::vector<Core::GUID> roughnessMaps =
        loadMaterialTextures(material, aiTextureType_DIFFUSE_ROUGHNESS, env);
    float roughness_factor = 1.f; float eta = 1.f;
    material->Get(AI_MATKEY_ROUGHNESS_FACTOR, roughness_factor);
    material->Get(AI_MATKEY_REFRACTI, eta);
    aiColor3D emissive_color(0.f, 0.f, 0.f);
    aiColor3D specular_color(0.f, 0.f, 0.f);
    material->Get(AI_MATKEY_COLOR_EMISSIVE, emissive_color);
    material->Get(AI_MATKEY_COLOR_SPECULAR, specular_color);
    gfxmat.emissiveColor =  {emissive_color.r, emissive_color.g, emissive_color.b};
    if (roughness_factor < 1.f) {
      gfxmat.roughness = std::max(0.05f, roughness_factor);
      gfxmat.eta = 3;
      gfxmat.specularColor =  {specular_color.r, specular_color.g, specular_color.b};
      gfxmat.BxDF = 1;
    }
    if (roughnessMaps.size() != 1) {
      // no roughness map
      //Core::LogManager::Error(
      //    "GFX :: SceneNodeLoader_assimp :: roughness map number is not 1.");
    } else {
      gfxmat.textures["roughness"] =
          GFX::Material::TextureEntry{roughnessMaps[0], 0};
    }
  }
  gfxmat.serialize();
  Core::GUID matID =
      GFX::GFXManager::get()->registerMaterialResource(gfxmat.path.c_str());
  env.materials[material] = matID;
  return matID;
}

Math::vec3 createTangent(Math::vec3 n) {
  if (n[2] < float(-1 + 1e-6)) {
    return Math::vec3(0, -1, 0);
  } else {
    const float a = 1 / (1 + n[2]);
    const float b = -n[0] * n[1] * a;
    return Math::vec3(1 - n[0] * n[0] * a, b, -n[0]);
  }
}

static inline auto processAssimpMesh(GameObjectHandle const& gfxNode,
                                     aiNode const* node, aiScene const* scene,
                                     AssimpLoaderEnv& env, GFX::Scene& gfxscene,
                                     MeshLoaderConfig meshConfig = {}) noexcept
    -> void {
  auto entity = gfxscene.getGameObject(gfxNode)->getEntity();
  // set transformation
  auto* transform = entity.getComponent<TransformComponent>();
  {  // apply the transformation
    auto& input_transform = node->mTransformation;
    Math::mat4 transform_mat = {
        input_transform.a1, input_transform.a2, input_transform.a3,
        input_transform.a4, input_transform.b1, input_transform.b2,
        input_transform.b3, input_transform.b4, input_transform.c1,
        input_transform.c2, input_transform.c3, input_transform.c4,
        input_transform.d1, input_transform.d2, input_transform.d3,
        input_transform.d4,
    };
    Math::vec3 t, r, s;
    Math::Decompose(transform_mat, &t, &r, &s);
    transform->translation = t;
    transform->eulerAngles = r;
    transform->scale = s;
  }

  // set animation
  auto anim_find = env.animation_map.find(std::string(node->mName.C_Str()));
  if (anim_find != env.animation_map.end()) {
    AnimationComponent* anim = entity.addComponent<AnimationComponent>();
    *anim = anim_find->second;
  }

  // load mesh
  uint32_t vertex_offset = 0;
  std::unordered_map<uint64_t, uint32_t> uniqueVertices{};

  std::vector<float> vertexBufferV = {};
  std::vector<float> positionBufferV = {};
  std::vector<uint16_t> indexBufferV = {};
  std::vector<uint32_t> indexBufferWV = {};

  // check whether tangent is need in mesh attributes
  bool needTangent = false;
  for (auto const& entry : meshConfig.layout.layout)
    if (entry.info == MeshDataLayout::VertexInfo::TANGENT) needTangent = true;

  // Loop over shapes
  GFX::Mesh gfxmesh;
  uint64_t global_index_offset = 0;
  uint32_t submesh_vertex_offset = 0, submesh_index_offset = 0;

  if (node->mNumMeshes == 0) return;
  for (unsigned int i = 0; i < node->mNumMeshes; i++) {
    aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];

    for (unsigned int i = 0; i < mesh->mNumVertices; i++) {
      // atrributes filling
      std::vector<float> vertex = {};
      std::vector<float> position = {};
      for (auto const& entry : meshConfig.layout.layout) {
        // vertex position
        if (entry.info == MeshDataLayout::VertexInfo::POSITION) {
          if (entry.format == RHI::VertexFormat::FLOAT32X3) {
            vertex.push_back(mesh->mVertices[i].x);
            vertex.push_back(mesh->mVertices[i].y);
            vertex.push_back(mesh->mVertices[i].z);
            if (meshConfig.usePositionBuffer) {
              position.push_back(mesh->mVertices[i].x);
              position.push_back(mesh->mVertices[i].y);
              position.push_back(mesh->mVertices[i].z);
            }
          } else {
            Core::LogManager::Error(
                "GFX :: SceneNodeLoader_assimp :: unwanted vertex format for "
                "POSITION attributes.");
            return;
          }
        } else if (entry.info == MeshDataLayout::VertexInfo::NORMAL) {
          vertex.push_back(mesh->mNormals[i].x);
          vertex.push_back(mesh->mNormals[i].y);
          vertex.push_back(mesh->mNormals[i].z);
        } else if (entry.info == MeshDataLayout::VertexInfo::UV) {
          if (mesh->mTextureCoords[0]) {
            vertex.push_back(mesh->mTextureCoords[0][i].x);
            vertex.push_back(mesh->mTextureCoords[0][i].y);
          } else {
            vertex.push_back(0);
            vertex.push_back(0);
          }
        } else if (entry.info == MeshDataLayout::VertexInfo::TANGENT) {
          if (mesh->mTangents) {
            vertex.push_back(mesh->mTangents[i].x);
            vertex.push_back(mesh->mTangents[i].y);
            vertex.push_back(mesh->mTangents[i].z);
          } else {
            Math::vec3 tangent = createTangent(Math::vec3{
              mesh->mNormals[i].x,
              mesh->mNormals[i].y,
              mesh->mNormals[i].z,
            });
            //createTangent
            vertex.push_back(tangent.x);
            vertex.push_back(tangent.y);
            vertex.push_back(tangent.z);          
          }
        } else if (entry.info == MeshDataLayout::VertexInfo::COLOR) {
          // Optional: vertex colors
          vertex.push_back(mesh->mColors[0][i].r);
          vertex.push_back(mesh->mColors[0][i].g);
          vertex.push_back(mesh->mColors[0][i].b);
        } else if (entry.info == MeshDataLayout::VertexInfo::CUSTOM) {
        }
      }

      vertexBufferV.insert(vertexBufferV.end(), vertex.begin(), vertex.end());
      positionBufferV.insert(positionBufferV.end(), position.begin(),
                             position.end());
      ++vertex_offset;
    }

    size_t index_offset = 0;
    for (unsigned int i = 0; i < mesh->mNumFaces; i++) {
      aiFace face = mesh->mFaces[i];
      for (unsigned int j = 0; j < face.mNumIndices; j++) {
        if (meshConfig.layout.format == RHI::IndexFormat::UINT16_t)
          indexBufferV.push_back(face.mIndices[j]);
        else if (meshConfig.layout.format == RHI::IndexFormat::UINT32_T)
          indexBufferWV.push_back(face.mIndices[j]);
        ++index_offset;
      }
    }

    // Loop over faces(polygon)
    global_index_offset += index_offset;

    gfxmesh.submeshes.push_back(
        GFX::Mesh::Submesh{submesh_index_offset, uint32_t(index_offset),
                           submesh_vertex_offset, uint32_t(0)});
    submesh_index_offset = global_index_offset;
    submesh_vertex_offset += mesh->mNumVertices;
  }
  // create mesh resource
  {  // register mesh
    gfxmesh.vertexBufferLayout = getVertexBufferLayout(meshConfig.layout);
    if (meshConfig.residentOnHost) {
      gfxmesh.vertexBuffer_host =
          Core::Buffer(sizeof(float) * vertexBufferV.size());
      memcpy(gfxmesh.vertexBuffer_host.data, vertexBufferV.data(),
             gfxmesh.vertexBuffer_host.size);
      gfxmesh.vertexBufferInfo.onHost = true;
      gfxmesh.vertexBufferInfo.size = gfxmesh.vertexBuffer_host.size;
      if (meshConfig.layout.format == RHI::IndexFormat::UINT16_t) {
        gfxmesh.indexBuffer_host =
            Core::Buffer(sizeof(uint16_t) * indexBufferV.size());
        memcpy(gfxmesh.indexBuffer_host.data, indexBufferV.data(),
               gfxmesh.indexBuffer_host.size);
        gfxmesh.indexBufferInfo.size = gfxmesh.indexBuffer_host.size;
        gfxmesh.indexBufferInfo.onHost = true;
      } else if (meshConfig.layout.format == RHI::IndexFormat::UINT32_T) {
        gfxmesh.indexBuffer_host =
            Core::Buffer(sizeof(uint32_t) * indexBufferWV.size());
        memcpy(gfxmesh.indexBuffer_host.data, indexBufferWV.data(),
               gfxmesh.indexBuffer_host.size);
        gfxmesh.indexBufferInfo.size = gfxmesh.indexBuffer_host.size;
        gfxmesh.indexBufferInfo.onHost = true;
      }
      if (meshConfig.usePositionBuffer) {
        gfxmesh.positionBuffer_host =
            Core::Buffer(sizeof(float) * positionBufferV.size());
        memcpy(gfxmesh.positionBuffer_host.data, positionBufferV.data(),
               gfxmesh.positionBuffer_host.size);
        gfxmesh.positionBufferInfo.onHost = true;
        gfxmesh.positionBufferInfo.size = gfxmesh.positionBuffer_host.size;
      }
    }
  }
  std::string path = env.directory + std::string(node->mName.C_Str());
  Core::GUID guid =
      Core::ResourceManager::get()->requestRuntimeGUID<GFX::Mesh>();
  Core::ORID orid = Core::requestORID();
  Core::ResourceManager::get()->database.registerResource(orid, guid);
  Core::ResourceManager::get()->addResource<GFX::Mesh>(guid,
                                                       std::move(gfxmesh));
  GFX::Mesh* meshResourceRef =
      Core::ResourceManager::get()->getResource<GFX::Mesh>(guid);
  meshResourceRef->ORID = orid;
  meshResourceRef->serialize();
  // bind scene
  entity.addComponent<MeshReference>()->mesh = meshResourceRef;
  gfxscene.getGameObject(gfxNode)->getEntity().addComponent<MeshRenderer>();

  MeshRenderer* meshRenderer =
    gfxscene.getGameObject(gfxNode)->getEntity().getComponent<MeshRenderer>();
  // meshRenderer->materials.push_back(i)
  for (unsigned int i = 0; i < node->mNumMeshes; i++) {
    aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
    if (mesh->mMaterialIndex >= 0) {
      aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];
      Core::GUID matID = loadMaterial(material, env);
      meshRenderer->materials.push_back(
        Core::ResourceManager::get()->getResource<GFX::Material>(matID));
    }
  }
}

static inline auto processAssimpNode(GameObjectHandle const& gfxNode,
                                     aiNode const* node, aiScene const* scene,
                                     AssimpLoaderEnv& env, GFX::Scene& gfxscene,
                                     MeshLoaderConfig meshConfig = {}) noexcept
    -> void {
  // process all meshes
  processAssimpMesh(gfxNode, node, scene, env, gfxscene, meshConfig);
  // process the meshes for all the following nodes
  for (unsigned int i = 0; i < node->mNumChildren; i++) {
    GameObjectHandle subNode = gfxscene.createGameObject(gfxNode);
    gfxscene.getGameObject(subNode)
        ->getEntity()
        .getComponent<TagComponent>()
        ->name = std::string(node->mChildren[i]->mName.C_Str());
    processAssimpNode(subNode, node->mChildren[i], scene, env, gfxscene,
                      meshConfig);
  }
}

/** Load obj file */
auto SceneNodeLoader_assimp::loadSceneNode(
    std::filesystem::path const& path,
                                 GFX::Scene& gfxscene,
                                 MeshLoaderConfig meshConfig) noexcept -> void {
  // load obj file
  Assimp::Importer importer;
  std::string path_str = path.string();
  const aiScene* scene =
      importer.ReadFile(path_str, aiProcess_Triangulate | aiProcess_FlipUVs |
                    aiProcess_GenUVCoords | aiProcess_CalcTangentSpace);

  if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE ||
      !scene->mRootNode) {
    Core::LogManager::Error("Assimp: " +
                            std::string(importer.GetErrorString()));
    return;
  }
  std::string directory = path.parent_path().string();

  GameObjectHandle rootNode = gfxscene.createGameObject(GFX::NULL_GO);
  gfxscene.getGameObject(rootNode)
      ->getEntity()
      .getComponent<TagComponent>()
      ->name = path.filename().string();

  AssimpLoaderEnv env;
  env.directory = directory;
  loadAssimpAnimation(scene, env);
  processAssimpNode(rootNode, scene->mRootNode, scene, env, gfxscene, meshConfig);
}

struct MitsubaLoaderEnv {
  std::string directory;
  std::unordered_map<TPM_NAMESPACE::Object const*, Core::GUID> textures;
  std::unordered_map<TPM_NAMESPACE::Object const*, Core::GUID> materials;
};

auto loadMaterialTextures(TPM_NAMESPACE::Object const* node,
                          MitsubaLoaderEnv* env) noexcept -> Core::GUID {
  if (env->textures.find(node) != env->textures.end()) {
    return env->textures[node];
  }
  Core::LogManager::Assert(
      node->type() == TPM_NAMESPACE::OT_TEXTURE,
      "GFX :: Mitsuba Loader :: Try load texture node not actually texture.");
  std::string filename = node->property("filename").getString();
  std::string tex_path = env->directory + "\\" + filename;
  Core::GUID guid =
      GFX::GFXManager::get()->registerTextureResource(tex_path.c_str());
  env->textures[node] = guid;
  return guid;
}

auto loadMaterial(TPM_NAMESPACE::Object const* node,
                  MitsubaLoaderEnv* env) noexcept -> Core::GUID {
  if (env->materials.find(node) != env->materials.end()) {
    return env->materials[node];
  }
  Core::LogManager::Assert(
      node->type() == TPM_NAMESPACE::OT_BSDF,
      "GFX :: Mitsuba Loader :: Try load material node not actually bsdf.");

  TPM_NAMESPACE::Object const* mat_node = nullptr;
  if (node->pluginType() == "dielectric") {
    mat_node = node;
  }
  else if(node->pluginType() == "roughdielectric") {
    mat_node = node;
  }
  else if(node->pluginType() == "thindielectric") {
    mat_node = node;
  }
  else if (node->pluginType() == "twosided") {
    if (node->anonymousChildren().size() == 0) {
      Core::LogManager::Error("Mitsuba Loader :: Material loading exception.");
      return Core::INVALID_GUID;
    }
    mat_node = node->anonymousChildren()[0].get();
  }
  else if (node->pluginType() == "mask") {
    if (node->anonymousChildren().size() == 0) {
      Core::LogManager::Error("Mitsuba Loader :: Material loading exception.");
      return Core::INVALID_GUID;
    }
    return loadMaterial(node->anonymousChildren()[0].get(), env);
  }
  else if (node->pluginType() == "bumpmap") {
    if (node->anonymousChildren().size() == 0) {
      Core::LogManager::Error("Mitsuba Loader :: Material loading exception.");
      return Core::INVALID_GUID;
    }
    return loadMaterial(node->anonymousChildren()[0].get(), env);
  }
  else {
    float a = 1.f;
  }
  std::string name = std::string(node->id());
  GFX::Material gfxmat;
  gfxmat.path = "content/materials/" +
                Core::StringHelper::invalidFileFolderName(name) + ".mat";
  gfxmat.ORID = Core::requestORID();
  if (mat_node->pluginType() == "roughplastic") {
    gfxmat.BxDF = 1;
  } else if (mat_node->pluginType() == "diffuse") {
    gfxmat.BxDF = 0;
  } else {
    gfxmat.BxDF = 0;
  }
  gfxmat.eta = mat_node->property("int_ior").getNumber(1.5f);

  gfxmat.name = name;

  for (auto const& child : mat_node->namedChildren()) {
    if (child.first == "diffuse_reflectance" ||
        child.first == "reflectance") {
      gfxmat.textures["base_color"] = {
          loadMaterialTextures(child.second.get(), env), 0};
    } else {
      float a = 1.f;
    }
  }
  gfxmat.serialize();
  Core::GUID matID =
      GFX::GFXManager::get()->registerMaterialResource(gfxmat.path.c_str());
  env->materials[node] = matID;
  return matID;
}

Core::GUID CloneMaterial(Core::GUID guid) {
  GFX::Material* material =
      Core::ResourceManager::get()->getResource<GFX::Material>(guid);
  GFX::Material gfxmat = *material;
  gfxmat.path =
      "content/materials/" + std::to_string(Core::requestORID()) + ".mat";
  gfxmat.serialize();
  return GFX::GFXManager::get()->registerMaterialResource(gfxmat.path.c_str());
}

auto loadMeshNode(TPM_NAMESPACE::Object const* node, GFX::Scene& gfxscene,
                  GameObjectHandle const& gfxNode,
                  MitsubaLoaderEnv* env,
                  MeshLoaderConfig meshConfig = {}) noexcept -> void {
  Core::LogManager::Assert(
      node->type() == TPM_NAMESPACE::OT_SHAPE,
      "GFX :: Mitsuba Loader :: Try load shape node not actually shape.");

  if (node->pluginType() == "obj") {
    std::string filename = node->property("filename").getString();
    std::string obj_path = env->directory + "\\" + filename;
    GameObjectHandle obj_node = SceneNodeLoader_obj::loadSceneNode(
        obj_path, gfxscene, gfxNode, meshConfig);
    GFX::MeshRenderer* renderer = gfxscene.getGameObject(obj_node)
                                      ->getEntity()
                                      .addComponent<GFX::MeshRenderer>();
    TPM_NAMESPACE::Transform transform =
        node->property("to_world").getTransform();
    Math::mat4 transform_mat = {
        transform.matrix[0], transform.matrix[1], transform.matrix[2], transform.matrix[3],
        transform.matrix[4], transform.matrix[5], transform.matrix[6], transform.matrix[7],
        transform.matrix[8], transform.matrix[9], transform.matrix[10], transform.matrix[11],
        transform.matrix[12], transform.matrix[13], transform.matrix[14], transform.matrix[15]
    };
    Math::vec3 t, r, s;
    Math::Decompose(transform_mat, &t, &r, &s);
    GFX::TransformComponent* transform_component =
        gfxscene.getGameObject(obj_node)
            ->getEntity()
            .getComponent<GFX::TransformComponent>();
    transform_component->translation = t;
    transform_component->scale = s;
    transform_component->eulerAngles = r;
    if (node->anonymousChildren().size() > 0) {
      TPM_NAMESPACE::Object* mat_node = node->anonymousChildren()[0].get();
      Core::GUID mat = loadMaterial(mat_node, env);
      renderer->materials.emplace_back(
          Core::ResourceManager::get()->getResource<GFX::Material>(mat));
    }
  }
  if (node->pluginType() == "cube") {
    std::string engine_path = Core::RuntimeConfig::get()->string_property("engine_path");
    std::string obj_path = engine_path + "\\" + "Binaries\\Runtime\\meshes\\cube.obj";
    GameObjectHandle obj_node = SceneNodeLoader_obj::loadSceneNode(
        obj_path, gfxscene, gfxNode, meshConfig);
    GFX::MeshRenderer* renderer = gfxscene.getGameObject(obj_node)
                                      ->getEntity()
                                      .addComponent<GFX::MeshRenderer>();
    TPM_NAMESPACE::Transform transform =
        node->property("to_world").getTransform();
    Math::mat4 transform_mat = {
        transform.matrix[0],  transform.matrix[1],  transform.matrix[2],
        transform.matrix[3],  transform.matrix[4],  transform.matrix[5],
        transform.matrix[6],  transform.matrix[7],  transform.matrix[8],
        transform.matrix[9],  transform.matrix[10], transform.matrix[11],
        transform.matrix[12], transform.matrix[13], transform.matrix[14],
        transform.matrix[15]};
    Math::vec3 t, r, s;
    Math::Decompose(transform_mat, &t, &r, &s);
    GFX::TransformComponent* transform_component =
        gfxscene.getGameObject(obj_node)
            ->getEntity()
            .getComponent<GFX::TransformComponent>();
    transform_component->translation = t;
    transform_component->scale = s;
    transform_component->eulerAngles = r / Math::float_Pi * 180;
    if (node->anonymousChildren().size() > 0) {
      TPM_NAMESPACE::Object* mat_node = node->anonymousChildren()[0].get();
      Core::GUID mat = loadMaterial(mat_node, env);
      renderer->materials.emplace_back(
          Core::ResourceManager::get()->getResource<GFX::Material>(mat));
    }
  }
  if (node->pluginType() == "rectangle") {
    std::string engine_path =
        Core::RuntimeConfig::get()->string_property("engine_path");
    std::string obj_path =
        engine_path + "\\" + "Binaries\\Runtime\\meshes\\rectangle.obj";
    GameObjectHandle obj_node = SceneNodeLoader_obj::loadSceneNode(
        obj_path, gfxscene, gfxNode, meshConfig);
    GFX::MeshRenderer* renderer = gfxscene.getGameObject(obj_node)
                                      ->getEntity()
                                      .addComponent<GFX::MeshRenderer>();
    TPM_NAMESPACE::Transform transform =
        node->property("to_world").getTransform();
    Math::mat4 transform_mat = {
        transform.matrix[0],  transform.matrix[1],  transform.matrix[2],
        transform.matrix[3],  transform.matrix[4],  transform.matrix[5],
        transform.matrix[6],  transform.matrix[7],  transform.matrix[8],
        transform.matrix[9],  transform.matrix[10], transform.matrix[11],
        transform.matrix[12], transform.matrix[13], transform.matrix[14],
        transform.matrix[15]};
    Math::vec3 t, r, s;
    Math::Decompose(transform_mat, &t, &r, &s);
    GFX::TransformComponent* transform_component =
        gfxscene.getGameObject(obj_node)
            ->getEntity()
            .getComponent<GFX::TransformComponent>();
    transform_component->translation = t;
    transform_component->scale = s;
    transform_component->eulerAngles = r / Math::float_Pi * 180;
    if (node->anonymousChildren().size() > 0) {
      TPM_NAMESPACE::Object* mat_node = node->anonymousChildren()[0].get();
      Core::GUID mat = loadMaterial(mat_node, env);
      renderer->materials.emplace_back(
          Core::ResourceManager::get()->getResource<GFX::Material>(mat));
    }
  }
  if (node->pluginType() == "sphere") {
    std::string engine_path = Core::RuntimeConfig::get()->string_property("engine_path");
    std::string obj_path = engine_path + "\\" + "Binaries\\Runtime\\meshes\\sphere.obj";
    GameObjectHandle obj_node = SceneNodeLoader_obj::loadSceneNode(
        obj_path, gfxscene, gfxNode, meshConfig);
    GFX::MeshRenderer* renderer = gfxscene.getGameObject(obj_node)
                                      ->getEntity()
                                      .addComponent<GFX::MeshRenderer>();
    TPM_NAMESPACE::Transform transform =
        node->property("to_world").getTransform();
    float radius = node->property("radius").getNumber();
    TPM_NAMESPACE::Vector center = node->property("center").getVector();
    GFX::TransformComponent* transform_component =
        gfxscene.getGameObject(obj_node)
            ->getEntity()
            .getComponent<GFX::TransformComponent>();
    transform_component->translation = Math::vec3(center.x, center.y, center.z);
    transform_component->scale = Math::vec3(radius);
    transform_component->eulerAngles = Math::vec3(0);

    if (node->anonymousChildren().size() > 0) {
      Core::GUID mat = Core::INVALID_GUID;
      Math::vec3 radiance = Math::vec3(0);
      for (auto& subnode : node->anonymousChildren()) {
        if (subnode->type() == TPM_NAMESPACE::OT_BSDF) {
            // material
          TPM_NAMESPACE::Object* mat_node = subnode.get();
          mat = loadMaterial(mat_node, env);
        } else if (subnode->type() == TPM_NAMESPACE::OT_EMITTER) {
          if (subnode->pluginType() == "area") {
            TPM_NAMESPACE::Color rgb = subnode->property("radiance").getColor();
            radiance = Math::vec3(rgb.r, rgb.g, rgb.b);
          }
        }
      }
      if (mat != Core::INVALID_GUID) {
        if (radiance.x != 0 || radiance.y != 0 || radiance.z != 0) {
          mat = CloneMaterial(mat);
        }
        GFX::Material* mat_ptr =
            Core::ResourceManager::get()->getResource<GFX::Material>(mat);
        mat_ptr->emissiveColor = radiance;
        mat_ptr->serialize();
        renderer->materials.emplace_back(mat_ptr);
      }
    }
  }
  else {
    Core::LogManager::Error("GFX :: Mitsuba Loader :: Unkown mesh type.");
  }
  //if (env->textures.find(node) != env->textures.end()) {
  //  return env->textures[node];
  //}
  //Core::LogManager::Assert(
  //    node->type() == TPM_NAMESPACE::OT_TEXTURE,
  //    "GFX :: Mitsuba Loader :: Try load texture node not actually texture.");
  //std::string filename = node->property("filename").getString();
  //std::string tex_path = env->directory + "\\" + filename;
  //Core::GUID guid =
  //    GFX::GFXManager::get()->registerTextureResource(tex_path.c_str());
  //env->textures[node] = guid;
  //return guid;
}

static inline auto processMitsubaNode(
    GameObjectHandle const& gfxNode, 
    TPM_NAMESPACE::Object const* node,
    MitsubaLoaderEnv* env,
    GFX::Scene& gfxscene,
    MeshLoaderConfig meshConfig = {}) noexcept
    -> void {
  switch (node->type()) {
    case TPM_NAMESPACE::OT_SCENE:
      break;
    case TPM_NAMESPACE::OT_BSDF:
      loadMaterial(node, env);
      break;
    case TPM_NAMESPACE::OT_FILM:
      break;
    case TPM_NAMESPACE::OT_INTEGRATOR:
      break;
    case TPM_NAMESPACE::OT_MEDIUM:
      break;
    case TPM_NAMESPACE::OT_PHASE:
      break;
    case TPM_NAMESPACE::OT_RFILTER:
      break;
    case TPM_NAMESPACE::OT_SAMPLER:
      break;
    case TPM_NAMESPACE::OT_SENSOR:
      break;
    case TPM_NAMESPACE::OT_SHAPE: 
      loadMeshNode(node, gfxscene, gfxNode, env, meshConfig);
      break;
    case TPM_NAMESPACE::OT_SUBSURFACE:
      break;
    case TPM_NAMESPACE::OT_TEXTURE:
      break;
    case TPM_NAMESPACE::OT_VOLUME:
      break;
    case TPM_NAMESPACE::_OT_COUNT:
      break;
    default:
      break;
  }

  //// process all meshes
  //processAssimpMesh(gfxNode, node, scene, env, gfxscene, meshConfig);
  //// process the meshes for all the following nodes
  //for (unsigned int i = 0; i < node->mNumChildren; i++) {
  //  GameObjectHandle subNode = gfxscene.createGameObject(gfxNode);
  //  gfxscene.getGameObject(subNode)
  //      ->getEntity()
  //      .getComponent<TagComponent>()
  //      ->name = std::string(node->mChildren[i]->mName.C_Str());
  //  processAssimpNode(subNode, node->mChildren[i], scene, env, gfxscene,
  //                    meshConfig);
  //}
}

/** Load mitsuba file */
auto SceneNodeLoader_mitsuba::loadSceneNode(
    std::filesystem::path const& path,
    GFX::Scene& gfxscene,
    MeshLoaderConfig meshConfig) noexcept -> void {
  // load mitsuba file
  TPM_NAMESPACE::SceneLoader loader;
  auto scene = loader.loadFromFile(path.string().c_str());

  MitsubaLoaderEnv env;
  GameObjectHandle rootNode = gfxscene.createGameObject(GFX::NULL_GO);
  gfxscene.getGameObject(rootNode)
      ->getEntity()
      .getComponent<TagComponent>()
      ->name = path.filename().string();
  env.directory = path.parent_path().string();

  for (auto& object : scene.anonymousChildren()) {
    processMitsubaNode(rootNode, object.get(), &env, gfxscene, meshConfig);
  }
  for (auto& object : scene.namedChildren()) {
    processMitsubaNode(rootNode, object.second.get(), &env, gfxscene, meshConfig);
  }

  float a = 1.f;
  //Assimp::Importer importer;
  //std::string path_str = path.string();
  //const aiScene* scene =
  //    importer.ReadFile(path_str, aiProcess_Triangulate | aiProcess_FlipUVs |
  //                                    aiProcess_CalcTangentSpace);

  //if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE ||
  //    !scene->mRootNode) {
  //  Core::LogManager::Error("Assimp: " +
  //                          std::string(importer.GetErrorString()));
  //  return;
  //}
  //std::string directory = path.parent_path().string();

  //GameObjectHandle rootNode = gfxscene.createGameObject(GFX::NULL_GO);
  //gfxscene.getGameObject(rootNode)
  //    ->getEntity()
  //    .getComponent<TagComponent>()
  //    ->name = path.filename().string();

  //AssimpLoaderEnv env;
  //env.directory = directory;
  //processAssimpNode(rootNode, scene->mRootNode, scene, env, gfxscene,
  //                  meshConfig);
}
}  // namespace SIByL::GFX