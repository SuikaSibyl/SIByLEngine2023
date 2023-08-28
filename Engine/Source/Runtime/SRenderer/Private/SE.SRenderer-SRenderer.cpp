#include "../Public/Core/SE.SRenderer-SRenderer.hpp"
#include <SE.Math.ShaderCommon.hpp>
#include "../../../../Shaders/SRenderer/include/common/light.hlsli"
#include "../../../../Shaders/SRenderer/include/common/octahedral.hlsli"
#include "../../../../Shaders/SRenderer/include/common/packing.hlsli"
#include <Config/SE.Core.Config.hpp>

namespace SIByL {

namespace Impl_Light {
using namespace Math;

void packLightColor(Math::vec3 radiance,
                    SRenderer::PolymorphicLightInfo& lightInfo) {
  float intensity = std::max(radiance.r, std::max(radiance.g, radiance.b));
  if (intensity > 0.0) {
    float logRadiance = Math::saturate(
        (log2(intensity) - kPolymorphicLightMinLog2Radiance) /
        (kPolymorphicLightMaxLog2Radiance - kPolymorphicLightMinLog2Radiance));
    uint32_t packedRadiance =
        std::min(uint32_t(std::ceil(logRadiance * 65534.0)) + 1, 0xffffu);
    float unpackedRadiance = unpackLightRadiance(packedRadiance);

    float3 normalizedRadiance = saturate(radiance / unpackedRadiance);

    lightInfo.logRadiance = lightInfo.logRadiance & ~0xffffu;
    lightInfo.logRadiance |= packedRadiance;
    lightInfo.colorTypeAndFlags = lightInfo.colorTypeAndFlags & ~0xffffffu;
    lightInfo.colorTypeAndFlags |= Pack_R8G8B8_UFLOAT(normalizedRadiance);
  }
}

float3 unpackLightColor(in_ref(SRenderer::PolymorphicLightInfo) lightInfo) {
  float3 color = Unpack_R8G8B8_UFLOAT(lightInfo.colorTypeAndFlags & 0xffffffu);
  float radiance = unpackLightRadiance(lightInfo.logRadiance & 0xffff);
  return color * float3(radiance);
}

uint32_t asuint(float x) {
  union FloatUIntUnion {
    float floatValue;
    uint32_t uintValue;
  } parser;
  parser.floatValue = x;
  return parser.uintValue;
}

std::vector<SRenderer::PolymorphicLightInfo> createPolymorphicLights(
    GFX::LightComponent const& light_component,
    Math::mat4 const& objMat) {
  std::vector<SRenderer::PolymorphicLightInfo> lights;
  switch (light_component.type) {
    case GFX::LightComponent::LightType::DIRECTIONAL: {
      SRenderer::PolymorphicLightInfo polymorphic;
      polymorphic.colorTypeAndFlags =
          (uint32_t)PolymorphicLightType::kDirectional
          << kPolymorphicLightTypeShift;
      packLightColor(light_component.intensity, polymorphic);
      float3 lightColor = unpackLightColor(polymorphic);
      Math::vec3 direction = Math::Transform(objMat) * Math::vec3(0, 0, 1);
      polymorphic.databyte0 =
          UnitVectorToUnorm32Octahedron(float3(normalize(direction)));
      lights.emplace_back(polymorphic);
    } break;
    case GFX::LightComponent::LightType::POINT: {
      SRenderer::PolymorphicLightInfo polymorphic;
      polymorphic.colorTypeAndFlags =
          (uint32_t)PolymorphicLightType::kPoint
          << kPolymorphicLightTypeShift;
      packLightColor(light_component.intensity, polymorphic);
      Math::vec4 pos = objMat * Math::vec4(0, 0, 0, 1);
      polymorphic.center = {pos.x, pos.y, pos.z};
      float3 lightColor = unpackLightColor(polymorphic);
      Math::vec3 direction = Math::Transform(objMat) * Math::vec3(0, 0, 1);
      polymorphic.databyte0 =
          UnitVectorToUnorm32Octahedron(float3(normalize(direction)));
      lights.emplace_back(polymorphic);
    } break;
    case GFX::LightComponent::LightType::SPOT: {
      SRenderer::PolymorphicLightInfo polymorphic;
      polymorphic.colorTypeAndFlags = (uint32_t)PolymorphicLightType::kSpot
                                      << kPolymorphicLightTypeShift;
      packLightColor(light_component.intensity, polymorphic);
      Math::vec4 pos = objMat * Math::vec4(0, 0, 0, 1);
      polymorphic.center = {pos.x, pos.y, pos.z};
      float3 lightColor = unpackLightColor(polymorphic);
      Math::vec3 direction = Math::Transform(objMat) * Math::vec3(0, 0, 1);
      polymorphic.databyte0 =
          UnitVectorToUnorm32Octahedron(float3(normalize(direction)));
      polymorphic.databyte1 = asuint(light_component.packed_data_0.x);
      polymorphic.scalars = asuint(light_component.packed_data_0.y);
      lights.emplace_back(polymorphic);
    } break;
    case GFX::LightComponent::LightType::TRIANGLE: {
    } break;
    case GFX::LightComponent::LightType::RECTANGLE: {
    } break;
    case GFX::LightComponent::LightType::MESH_PRIMITIVE: {
    } break;
    case GFX::LightComponent::LightType::ENVIRONMENT: {
    } break;
    case GFX::LightComponent::LightType::VPL: {
    } break;
    default:
      break;
  }
  return lights;
}
}

#pragma region SRENDERER_IMPL

GFX::MeshDataLayout SRenderer::meshDataLayout = {
    {
        {RHI::VertexFormat::FLOAT32X3,
         GFX::MeshDataLayout::VertexInfo::POSITION},
        {RHI::VertexFormat::FLOAT32X3, GFX::MeshDataLayout::VertexInfo::NORMAL},
        {RHI::VertexFormat::FLOAT32X3,
         GFX::MeshDataLayout::VertexInfo::TANGENT},
        {RHI::VertexFormat::FLOAT32X2, GFX::MeshDataLayout::VertexInfo::UV},
    },
    RHI::IndexFormat::UINT32_T};

GFX::MeshLoaderConfig SRenderer::meshLoadConfig = {meshDataLayout, true, true,
                                                   false, true};

RHI::VertexBufferLayout SRenderer::vertexBufferLayout =
    GFX::getVertexBufferLayout(meshDataLayout);

auto SRenderer::init(GFX::Scene& scene) noexcept -> void {
  RHI::Device* device = GFX::GFXManager::get()->rhiLayer->getDevice();

  // create common descriptor data
  RHI::ShaderStagesFlags stages = (uint32_t)RHI::ShaderStages::VERTEX |
                                  (uint32_t)RHI::ShaderStages::FRAGMENT |
                                  (uint32_t)RHI::ShaderStages::RAYGEN |
                                  (uint32_t)RHI::ShaderStages::CLOSEST_HIT |
                                  (uint32_t)RHI::ShaderStages::INTERSECTION |
                                  (uint32_t)RHI::ShaderStages::MISS |
                                  (uint32_t)RHI::ShaderStages::CALLABLE |
                                  (uint32_t)RHI::ShaderStages::ANY_HIT |
                                  (uint32_t)RHI::ShaderStages::COMPUTE;

  sceneDataBuffers.global_uniform_buffer =
      GFX::GFXManager::get()->createStructuredUniformBuffer<GlobalUniforms>();
  sceneDataBuffers.scene_info_buffer =
      GFX::GFXManager::get()
          ->createStructuredUniformBuffer<SceneInfoUniforms>();

  // create default setting
  std::string engine_path =
      Core::RuntimeConfig::get()->string_property("engine_path");
  defaultResources.defaultTexture =
      GFX::GFXManager::get()->registerTextureResource(
          (engine_path + "../Engine/Binaries/Runtime/textures/uv_checker.png")
              .c_str());

  // create custom primitive records
  {
    customPrimitive.sphere.customGeometry = RHI::BLASCustomGeometry{
        RHI::AffineTransformMatrix{},
        std::vector<Math::bounds3>{Math::bounds3{{-1, -1, -1}, {+1, +1, +1}}},
        (uint32_t)RHI::BLASGeometryFlagBits::NO_DUPLICATE_ANY_HIT_INVOCATION,
    };
    customPrimitive.sphere.meshRecord.submesh_geometry.push_back(
        GeometryDrawData{
            0,                             // vertexOffset;
            0,                             // indexOffset;
            0,                             // materialID;
            0,                             // indexSize;
            0,                             // padding0;
            0,                             // padding1;
            0,                             // padding2;
            1.f,                           // oddNegativeScaling;
            RHI::AffineTransformMatrix{},  // geometryTransform
            RHI::AffineTransformMatrix{},  // geometryTransformInverse
        });
    customPrimitive.sphere.meshRecord.blas_desc = RHI::BLASDescriptor{};
    customPrimitive.sphere.meshRecord.blas_desc.customGeometries.push_back(
        customPrimitive.sphere.customGeometry);
    if (device->getRayTracingExtension() != nullptr)
      customPrimitive.sphere.meshRecord.blases =
          device->createBLAS(customPrimitive.sphere.meshRecord.blas_desc);
  }

  raCommon.csm_info_device =
      GFX::GFXManager::get()
          ->createStructuredUniformBuffer<RACommon::CascadeShadowmapData>();

  // pack scene
  packScene(scene);
}

struct SceneDataPackState {
  bool invalidVIPBuffer = false;
  bool invalidGeometryBuffer = false;
  bool dirtyGeometryBuffer = false;
  bool invalidLightBuffer = false;
  bool invalidAccumulation = false;
  bool invalidLightDistBuffer = false;

  bool invalidTLAS = false;
};

auto invalid_mesh_record(GFX::Mesh* mesh, SRenderer* srenderer,
                                SceneDataPackState& state) noexcept
    -> std::unordered_map<GFX::Mesh*,
                          SRenderer::SceneDataPack::MeshRecord>::iterator {
  srenderer->sceneDataPack.mesh_record[mesh] = {};
  auto meshRecord = srenderer->sceneDataPack.mesh_record.find(mesh);
  {  // create mesh record, add all data buffers
    uint32_t vertex_offset = srenderer->sceneDataPack.vertex_buffer_cpu.size();
    uint32_t position_offset =
        srenderer->sceneDataPack.position_buffer_cpu.size();
    srenderer->sceneDataPack.vertex_buffer_cpu.resize(
        vertex_offset + mesh->vertexBuffer_host.size / sizeof(float));
    memcpy(&(srenderer->sceneDataPack.vertex_buffer_cpu[vertex_offset]),
           mesh->vertexBuffer_host.data, mesh->vertexBuffer_host.size);
    if (srenderer->config.enableRayTracing) {
      srenderer->sceneDataPack.position_buffer_cpu.resize(
          position_offset + mesh->positionBuffer_host.size / sizeof(float));
      memcpy(&(srenderer->sceneDataPack.position_buffer_cpu[position_offset]),
             mesh->positionBuffer_host.data, mesh->positionBuffer_host.size);
    }
    uint32_t index_offset = srenderer->sceneDataPack.index_buffer_cpu.size();
    srenderer->sceneDataPack.index_buffer_cpu.resize(
        index_offset + mesh->indexBuffer_host.size / sizeof(uint32_t));
    memcpy(&(srenderer->sceneDataPack.index_buffer_cpu[index_offset]),
           mesh->indexBuffer_host.data, mesh->indexBuffer_host.size);
    // add all submeshes
    for (auto& submesh : mesh->submeshes) {
      SRenderer::GeometryDrawData geometry;
      geometry.vertexOffset =
          submesh.baseVertex + vertex_offset * sizeof(float) /
                                   SRenderer::vertexBufferLayout.arrayStride;
      geometry.indexOffset = submesh.offset + index_offset;
      geometry.materialID = 0;
      geometry.indexSize = submesh.size;
      geometry.geometryTransform = {};
      meshRecord->second.submesh_geometry.push_back(geometry);
      if (srenderer->config.enableRayTracing) {
        RHI::BLASDescriptor& blasDesc = meshRecord->second.blas_desc;
        blasDesc.triangleGeometries.push_back(RHI::BLASTriangleGeometry{
            nullptr, nullptr, nullptr, RHI::IndexFormat::UINT32_T,
            uint32_t(mesh->positionBuffer_host.size / (sizeof(float) * 3)),
            geometry.vertexOffset, geometry.indexSize / 3,
            uint32_t(geometry.indexOffset * sizeof(uint32_t)),
            RHI::AffineTransformMatrix{},
            (uint32_t)
                RHI::BLASGeometryFlagBits::NO_DUPLICATE_ANY_HIT_INVOCATION,
            geometry.materialID});
      }
    }
    // set dirty of Vertex/Index/Position Buffer:
    state.invalidVIPBuffer = true;
  }
  return meshRecord;
}

auto luminance(Math::vec3 const& rgb) noexcept -> float {
  return rgb.x * float(0.212671) + rgb.y * float(0.715160) +
         rgb.z * float(0.072169);
}

auto invalid_game_object(GFX::GameObject* gameobject, GFX::Scene& scene,
                                SRenderer* srenderer,
                                SceneDataPackState& state) noexcept -> void {
  // fetch components
  GFX::TransformComponent* transform =
      gameobject->getEntity().getComponent<GFX::TransformComponent>();
  GFX::MeshReference* meshref =
      gameobject->getEntity().getComponent<GFX::MeshReference>();
  GFX::MeshRenderer* meshrenderer =
      gameobject->getEntity().getComponent<GFX::MeshRenderer>();
  GFX::LightComponent* lightComponenet =
      gameobject->getEntity().getComponent<GFX::LightComponent>();
  // we do not cares those game object has no relative components
  if (meshref == nullptr && meshrenderer == nullptr &&
      lightComponenet == nullptr)
    return;
  // get transform
  Math::mat4 objectTransform;
  float oddScaling = 1.f;
  Math::vec3 scaling = Math::vec3{1, 1, 1};
  {  // get mesh transform matrix
    GFX::GameObject* go = gameobject;
    GFX::TransformComponent* transform =
        go->getEntity().getComponent<GFX::TransformComponent>();
    objectTransform = transform->getTransform() * objectTransform;
    oddScaling *= transform->scale.x * transform->scale.y * transform->scale.z;
    scaling *= transform->scale;
    while (go->parent != Core::NULL_ENTITY) {
      go = scene.getGameObject(go->parent);
      GFX::TransformComponent* transform =
          go->getEntity().getComponent<GFX::TransformComponent>();
      objectTransform = transform->getTransform() * objectTransform;
      oddScaling *=
          transform->scale.x * transform->scale.y * transform->scale.z;
      scaling *= transform->scale;
    }
  }
  // geometry processing
  bool transformChanged = false;
  if (meshref && meshrenderer) {
    // mesh resource
    GFX::Mesh* mesh = meshref->mesh;
    Math::bounds3 bounds = Math::Transform(objectTransform) * mesh->aabb;
    srenderer->statisticsData.aabb =
        Math::unionBounds(srenderer->statisticsData.aabb, bounds);
    auto meshRecord = srenderer->sceneDataPack.mesh_record.find(mesh);
    {  // insert mesh to the index / vertex buffer if it does not exist now
      if (meshRecord == srenderer->sceneDataPack.mesh_record.end())
        meshRecord = invalid_mesh_record(mesh, srenderer, state);
    }
    // mesh reference
    auto meshRefRecord =
        srenderer->sceneDataPack.mesh_ref_record.find(gameobject->entity);
    {  // add mesh reference if not exist now
      if (meshRefRecord == srenderer->sceneDataPack.mesh_ref_record.end()) {
        // 
        state.invalidGeometryBuffer = true;
      } else {
        // has
        for (auto idx : meshRefRecord->second.geometry_indices) {
          if (srenderer->sceneDataPack.geometry_buffer_cpu[idx]
                  .geometryTransform != objectTransform) {
            transformChanged = true;
            state.dirtyGeometryBuffer = true;
            state.invalidAccumulation = true;
            srenderer->sceneDataPack.geometry_buffer_cpu[idx]
                .geometryTransform = objectTransform;
            srenderer->sceneDataPack.geometry_buffer_cpu[idx]
                .geometryTransformInverse = Math::inverse(objectTransform);
            srenderer->sceneDataPack.geometry_buffer_cpu[idx]
                .oddNegativeScaling = oddScaling >= 0 ? 1.f : -1.f;
          }
        }
        meshRefRecord->second.blasInstance.transform = objectTransform;
        srenderer->sceneDataPack.tlas_desc.instances.push_back(
            meshRefRecord->second.blasInstance);
      }
    }
  }
  // light processing
  if (lightComponenet) {
    // if light is diffuse area light
    if (lightComponenet->type ==
        GFX::LightComponent::LightType::MESH_PRIMITIVE) {
      auto meshRefRecord =
          srenderer->sceneDataPack.mesh_ref_record.find(gameobject->entity);
      if (meshRefRecord != srenderer->sceneDataPack.mesh_ref_record.end()) {
        // now only diffuse area light is supported, we must have mesh record
        auto entityLightCompRecord =
            srenderer->sceneDataPack.light_comp_record.find(gameobject->entity);
        if (entityLightCompRecord ==
            srenderer->sceneDataPack.light_comp_record.end()) {
        }
        for (size_t index : entityLightCompRecord->second.lightIndices) {
            // TODO
          //if (srenderer->sceneDataPack.light_buffer_cpu[index]
          //        .areaData.intensity != lightComponenet->intensity) {
          //  state.invalidAccumulation = true;
          //  state.invalidLightBuffer = true;
          //  srenderer->sceneDataPack.light_buffer_cpu[index]
          //      .areaData.intensity = lightComponenet->intensity;
          //}
          // if transform changed, we need to recompute surface area thing
          if (transformChanged &&
              entityLightCompRecord->second.scaling != scaling) {
            state.invalidLightDistBuffer = true;
            auto& lightData = srenderer->sceneDataPack.light_buffer_cpu[index];
            if (meshref->customPrimitiveFlag == 0 ||
                meshref->customPrimitiveFlag == 1) {
              float totalArea = 0;
              for (uint32_t geoidx : meshRefRecord->second.geometry_indices) {
                // GFX::Mesh* meshPtr =
                // meshRefRecord->second.meshReference->mesh;
                size_t primitiveNum = meshRefRecord->second.meshReference->mesh
                                          ->indexBuffer_host.size /
                                      (3 * sizeof(uint32_t));
                uint32_t* indexBuffer =
                    static_cast<uint32_t*>(meshRefRecord->second.meshReference
                                               ->mesh->indexBuffer_host.data);
                float* vertexBuffer =
                    static_cast<float*>(meshRefRecord->second.meshReference
                                            ->mesh->vertexBuffer_host.data);
                size_t vertexStride =
                    sizeof(SRenderer::InterleavedVertex) / sizeof(float);
                std::vector<float> areas;
                for (size_t i = 0; i < primitiveNum; ++i) {
                  uint32_t i0 = indexBuffer[3 * i + 0];
                  uint32_t i1 = indexBuffer[3 * i + 1];
                  uint32_t i2 = indexBuffer[3 * i + 2];
                  Math::vec3 const& pos0 =
                      *(Math::vec3*)(&(vertexBuffer[i0 * vertexStride]));
                  Math::vec3 const& pos1 =
                      *(Math::vec3*)(&(vertexBuffer[i1 * vertexStride]));
                  Math::vec3 const& pos2 =
                      *(Math::vec3*)(&(vertexBuffer[i2 * vertexStride]));
                  Math::vec3 v0 =
                      Math::vec3(objectTransform * Math::vec4(pos0, 0));
                  Math::vec3 v1 =
                      Math::vec3(objectTransform * Math::vec4(pos1, 0));
                  Math::vec3 v2 =
                      Math::vec3(objectTransform * Math::vec4(pos2, 0));
                  Math::vec3 const e1 = v1 - v0;
                  Math::vec3 const e2 = v2 - v0;
                  float area = Math::length(Math::cross(e1, e2)) / 2;
                  areas.push_back(area);
                  totalArea += area;
                }
                // create dist1D
                // entityLightCompRecord->second.tableDist = areas;
                entityLightCompRecord->second.tableDists[0] =
                    SRenderer::TableDist1D{areas};
                entityLightCompRecord->second.lightPowers[0] =
                    totalArea * 3.1415926 *
                    luminance(lightComponenet->intensity);
                srenderer->sceneDataPack.geometry_buffer_cpu[geoidx]
                    .surfaceArea = totalArea;
              }
            }
            // if mesh reference is pointing to a sphere mesh
            else if (meshref->customPrimitiveFlag == 2 ||
                     meshref->customPrimitiveFlag == 3) {
              float const radius = Math::length(
                  Math::vec3(objectTransform * Math::vec4(1, 0, 0, 1)) -
                  Math::vec3(objectTransform * Math::vec4(0, 0, 0, 1)));
              float const surfaceArea = 4 * 3.1415926 * radius * radius;
              entityLightCompRecord->second.lightPowers[0] =
                  surfaceArea * 3.1415926 *
                  luminance(lightComponenet->intensity);
              srenderer->sceneDataPack
                  .geometry_buffer_cpu[meshRefRecord->second
                                           .geometry_indices[0]]
                  .surfaceArea = surfaceArea;
            }
          }
        }
      }
    } 
    else if (lightComponenet->type ==
               GFX::LightComponent::LightType::ENVIRONMENT) {
      lightComponenet->texture;
    } else {
      auto entityLightCompRecord =
          srenderer->sceneDataPack.light_comp_record.find(gameobject->entity);
      if (entityLightCompRecord ==
          srenderer->sceneDataPack.light_comp_record.end()) {
        
      }
      for (size_t index : entityLightCompRecord->second.lightIndices) {
        if (lightComponenet->type ==
            GFX::LightComponent::LightType::DIRECTIONAL) {
          srenderer->raCommon.mainDirectionalLight =
              RACommon::DirectionalLightInfo{objectTransform, uint32_t(index)};
        }

        std::vector<SRenderer::PolymorphicLightInfo> polylights =
            Impl_Light::createPolymorphicLights(*lightComponenet,
                                                objectTransform);
        srenderer->sceneDataPack.light_buffer_cpu[index] = polylights[0];
      }
    }
  }
}

auto invalidSceneLightingSetting(SRenderer* srenderer) noexcept -> void {
  srenderer->sceneDataPack.sample_dist_buffer_cpu.clear();
  std::vector<float> lightPowerArray = {};
  for (auto& lightCompRecord : srenderer->sceneDataPack.light_comp_record) {
    uint32_t i = 0;
    for (uint32_t subLightIdx : lightCompRecord.second.lightIndices) {
      auto& light = srenderer->sceneDataPack.light_buffer_cpu[subLightIdx];
      // push light power
      lightPowerArray.push_back(lightCompRecord.second.lightPowers[i]);
      // push light dist array
      if (lightCompRecord.second.tableDists.size() == 0) {
        continue;
      } else {
        // TODO
        //light.areaData.sample_dist_size_0 =
        //    lightCompRecord.second.tableDists[i].pmf.size();
        //light.areaData.sample_dist_offset_pmf_0 =
        //    srenderer->sceneDataPack.sample_dist_buffer_cpu.size();
        //srenderer->sceneDataPack.sample_dist_buffer_cpu.insert(
        //    srenderer->sceneDataPack.sample_dist_buffer_cpu.end(),
        //    lightCompRecord.second.tableDists[i].pmf.begin(),
        //    lightCompRecord.second.tableDists[i].pmf.end());
        //light.areaData.sample_dist_offset_cdf_0 =
        //    srenderer->sceneDataPack.sample_dist_buffer_cpu.size();
        srenderer->sceneDataPack.sample_dist_buffer_cpu.insert(
            srenderer->sceneDataPack.sample_dist_buffer_cpu.end(),
            lightCompRecord.second.tableDists[i].cdf.begin(),
            lightCompRecord.second.tableDists[i].cdf.end());
        ++i;
      }
    }
  }
  srenderer->sceneDataPack.lightTableDist =
      SRenderer::TableDist1D{lightPowerArray};
  srenderer->sceneDataPack.sceneInfoUniform.light_num =
      srenderer->sceneDataPack.lightTableDist.pmf.size();
  srenderer->sceneDataPack.sceneInfoUniform.light_offset_pmf =
      srenderer->sceneDataPack.sample_dist_buffer_cpu.size();
  srenderer->sceneDataPack.sample_dist_buffer_cpu.insert(
      srenderer->sceneDataPack.sample_dist_buffer_cpu.end(),
      srenderer->sceneDataPack.lightTableDist.pmf.begin(),
      srenderer->sceneDataPack.lightTableDist.pmf.end());
  srenderer->sceneDataPack.sceneInfoUniform.light_offset_cdf =
      srenderer->sceneDataPack.sample_dist_buffer_cpu.size();
  srenderer->sceneDataPack.sample_dist_buffer_cpu.insert(
      srenderer->sceneDataPack.sample_dist_buffer_cpu.end(),
      srenderer->sceneDataPack.lightTableDist.cdf.begin(),
      srenderer->sceneDataPack.lightTableDist.cdf.end());
  // also set pmf
  size_t i = 0;
  for (auto& lightCompRecord : srenderer->sceneDataPack.light_comp_record) {
    for (uint32_t subLightIdx : lightCompRecord.second.lightIndices) {
      auto& light = srenderer->sceneDataPack.light_buffer_cpu[subLightIdx];
      //light.areaData.total_value =
      //    srenderer->sceneDataPack.lightTableDist.pmf[i++];
    }
  }
}

auto SRenderer::invalidScene(GFX::Scene& scene) noexcept -> void {
  //
  statisticsData.aabb = Math::bounds3{};
  raCommon.mainDirectionalLight = std::nullopt;
  //
  RHI::Device* device = GFX::GFXManager::get()->rhiLayer->getDevice();
  sceneDataPack.tlas_desc.instances.clear();
  // for all mesh refs
  SceneDataPackState packstate;
  for (auto go_handle : scene.gameObjects) {
    invalid_game_object(scene.getGameObject(go_handle.first), scene, this,
                        packstate);
  }
  // for all materials
  bool invalid_material_buffer = false;
  for (auto& iter : sceneDataPack.material_record) {
    if (iter.first->isDirty) {
      MaterialData& matData = sceneDataPack.material_buffer_cpu[iter.second];
      GFX::Material* mat = iter.first;
      matData.bsdf_id = mat->BxDF;
      matData.alphaCutoff = mat->alphaThreshold;
      matData.roughness = mat->roughness;
      matData.metalness = mat->metalness;
      matData.baseOrDiffuseColor = mat->baseOrDiffuseColor;
      matData.specularColor = mat->specularColor;
      matData.emissiveColor = mat->emissiveColor;
      matData.transmissionFactor = mat->eta;
      iter.first->isDirty = false;
      invalid_material_buffer = true;
    }
  }
  if (invalid_material_buffer) {
    sceneDataPack.back_material_buffer = std::move(sceneDataPack.material_buffer);
    sceneDataPack.material_buffer = device->createDeviceLocalBuffer(
        sceneDataPack.material_buffer_cpu.data(),
        sceneDataPack.material_buffer_cpu.size() * sizeof(MaterialData),
        (uint32_t)RHI::BufferUsage::STORAGE);
  }

  sceneDataPack.back_tlas = sceneDataPack.tlas;

  // if (packstate.invalidTLAS)
  if (sceneDataPack.tlas_desc.instances.size() != 0)
    if (device->getRayTracingExtension())
      sceneDataPack.tlas = device->createTLAS(sceneDataPack.tlas_desc);

  sceneDataPack.back_light_buffer = std::move(sceneDataPack.light_buffer);
  sceneDataPack.light_buffer = device->createDeviceLocalBuffer(
      sceneDataPack.light_buffer_cpu.data(),
      sceneDataPack.light_buffer_cpu.size() * sizeof(PolymorphicLightInfo),
      (uint32_t)RHI::BufferUsage::SHADER_DEVICE_ADDRESS |
          (uint32_t)RHI::BufferUsage::STORAGE);

  RHI::MultiFrameFlights* multiFrameFlights =
      GFX::GFXManager::get()->rhiLayer->getMultiFrameFlights();
  uint32_t fid = multiFrameFlights->getFlightIndex();

  // TODO
  commonDescData.set0_flights_resources[fid][5] = {
      5, RHI::BindingResource{{sceneDataPack.light_buffer.get(), 0,
                               sceneDataPack.light_buffer->size()}}};
  commonDescData.set1_flights_resources[fid][0] = {
      0, RHI::BindingResource{sceneDataPack.tlas.get()}};
  commonDescData.set0_flights_resources[fid][4] = {
      4, RHI::BindingResource{{sceneDataPack.material_buffer.get(), 0,
                               sceneDataPack.material_buffer->size()}}};

  sceneDataBuffers.scene_info_buffer.setStructure(
      sceneDataPack.sceneInfoUniform, fid);

  if (packstate.invalidLightDistBuffer) {
    invalidSceneLightingSetting(this);
    sceneDataPack.back_sample_dist_buffer =
        std::move(sceneDataPack.sample_dist_buffer);
    sceneDataPack.sample_dist_buffer = device->createDeviceLocalBuffer(
        sceneDataPack.sample_dist_buffer_cpu.data(),
        sceneDataPack.sample_dist_buffer_cpu.size() * sizeof(float),
        (uint32_t)RHI::BufferUsage::SHADER_DEVICE_ADDRESS |
            (uint32_t)RHI::BufferUsage::STORAGE);
  }

  commonDescData.set0_flights_resources[fid][6] = {
      6, RHI::BindingResource{{sceneDataPack.sample_dist_buffer.get(), 0,
                               sceneDataPack.sample_dist_buffer.get() != 0
                                   ? sceneDataPack.sample_dist_buffer->size()
                                   : 0}}};

  if (packstate.invalidAccumulation) {
    state.batchIdx = 0;
  }
  if (sceneDataPack.geometry_buffer_cpu.size() > 0) {
    sceneDataBuffers.geometry_buffer.setStructure(
        sceneDataPack.geometry_buffer_cpu.data(), fid);
  }

  raCommon.sceneAABB = statisticsData.aabb;
  rtCommon.accumIDX = state.batchIdx;
}

auto packTexture(SRenderer* srenderer, Core::GUID guid) -> uint32_t {
  RHI::TextureView* texView = Core::ResourceManager::get()
                                  ->getResource<GFX::Texture>(guid)
                                  ->originalView.get();
  auto findTex = srenderer->sceneDataPack.textureview_record.find(texView);
  if (findTex == srenderer->sceneDataPack.textureview_record.end()) {
    uint32_t texID = srenderer->sceneDataPack.unbinded_textures.size();
    srenderer->sceneDataPack.unbinded_textures.push_back(texView);
    srenderer->sceneDataPack.textureview_record[texView] = texID;
    findTex = srenderer->sceneDataPack.textureview_record.find(texView);
  }
  return findTex->second;
};

auto SRenderer::packScene(GFX::Scene& scene) noexcept -> void {
  RHI::Device* device = GFX::GFXManager::get()->rhiLayer->getDevice();
  for (auto go_handle : scene.gameObjects) {
    auto* go = scene.getGameObject(go_handle.first);
    GFX::MeshReference* meshref =
        go->getEntity().getComponent<GFX::MeshReference>();
    GFX::MeshRenderer* meshrenderer =
        go->getEntity().getComponent<GFX::MeshRenderer>();
    GFX::LightComponent* lightComponenet =
        go->getEntity().getComponent<GFX::LightComponent>();

    // get transform
    float oddScaling = 1;
    Math::mat4 objectMat;
    Math::vec3 scaling = Math::vec3{1, 1, 1};
    {  // get mesh transform matrix
      GFX::TransformComponent* transform =
          go->getEntity().getComponent<GFX::TransformComponent>();
      objectMat = transform->getTransform() * objectMat;
      oddScaling *=
          transform->scale.x * transform->scale.y * transform->scale.z;
      scaling *= transform->scale;
      auto* go_loop = go;
      while (go_loop->parent != Core::NULL_ENTITY) {
        go_loop = scene.getGameObject(go_loop->parent);
        GFX::TransformComponent* transform =
            go_loop->getEntity().getComponent<GFX::TransformComponent>();
        objectMat = transform->getTransform() * objectMat;
        oddScaling *=
            transform->scale.x * transform->scale.y * transform->scale.z;
        scaling *= transform->scale;
      }
      if (oddScaling != 0) oddScaling / std::abs(oddScaling);
    }

    if (lightComponenet && !meshref) {
      if (lightComponenet->type ==
          GFX::LightComponent::LightType::ENVIRONMENT) {
        sceneDataPack.sceneInfoUniform.env_map =
            packTexture(this, lightComponenet->texture->guid);
      }
      else if (lightComponenet->type ==
                 GFX::LightComponent::LightType::DIRECTIONAL) {
        sceneDataPack.light_comp_record[go->entity] =
            SceneDataPack::EntityLightRecord{};
        auto entityLightCompRecord =
            sceneDataPack.light_comp_record.find(go->entity);

        raCommon.mainDirectionalLight = RACommon::DirectionalLightInfo{
            objectMat, uint32_t(sceneDataPack.light_buffer_cpu.size())};
        entityLightCompRecord->second.lightIndices.push_back(
            sceneDataPack.light_buffer_cpu.size());

        std::vector<SRenderer::PolymorphicLightInfo> polylights =
            Impl_Light::createPolymorphicLights(*lightComponenet, objectMat);
        sceneDataPack.light_buffer_cpu.push_back(polylights[0]);
      } else if (lightComponenet->type ==
                 GFX::LightComponent::LightType::SPOT ||
                 lightComponenet->type ==
                     GFX::LightComponent::LightType::POINT) {
        sceneDataPack.light_comp_record[go->entity] =
            SceneDataPack::EntityLightRecord{};
        auto entityLightCompRecord =
            sceneDataPack.light_comp_record.find(go->entity);

        entityLightCompRecord->second.lightIndices.push_back(
            sceneDataPack.light_buffer_cpu.size());

        std::vector<SRenderer::PolymorphicLightInfo> polylights =
            Impl_Light::createPolymorphicLights(*lightComponenet, objectMat);
        sceneDataPack.light_buffer_cpu.push_back(polylights[0]);
      }
    }
    // if mesh reference is pointing to a triangle mesh
    if (!meshref) continue;
    GFX::Mesh* mesh = meshref->mesh;
    if (!mesh) continue;
    if (!meshrenderer) continue;
    // if do not have according mesh record, add the record
    auto meshRecord = sceneDataPack.mesh_record.find(mesh);
    if (meshRecord == sceneDataPack.mesh_record.end()) {
      sceneDataPack.geometryDirty = true;
      sceneDataPack.mesh_record[mesh] = {};
      meshRecord = sceneDataPack.mesh_record.find(mesh);
      {
        // create mesh record, add all data buffers
        uint32_t vertex_offset = sceneDataPack.vertex_buffer_cpu.size();
        uint32_t position_offset = sceneDataPack.position_buffer_cpu.size();
        sceneDataPack.vertex_buffer_cpu.resize(
            vertex_offset + mesh->vertexBuffer_host.size / sizeof(float));
        memcpy(&(sceneDataPack.vertex_buffer_cpu[vertex_offset]),
               mesh->vertexBuffer_host.data, mesh->vertexBuffer_host.size);
        if (config.enableRayTracing) {
          sceneDataPack.position_buffer_cpu.resize(
              position_offset + mesh->positionBuffer_host.size / sizeof(float));
          memcpy(&(sceneDataPack.position_buffer_cpu[position_offset]),
                 mesh->positionBuffer_host.data,
                 mesh->positionBuffer_host.size);
        }
        uint32_t index_offset = sceneDataPack.index_buffer_cpu.size();
        sceneDataPack.index_buffer_cpu.resize(
            index_offset + mesh->indexBuffer_host.size / sizeof(uint32_t));
        memcpy(&(sceneDataPack.index_buffer_cpu[index_offset]),
               mesh->indexBuffer_host.data, mesh->indexBuffer_host.size);
        // add all submeshes
        for (auto& submesh : mesh->submeshes) {
          GeometryDrawData geometry;
          geometry.vertexOffset = submesh.baseVertex +
                                  vertex_offset * sizeof(float) /
                                      SRenderer::vertexBufferLayout.arrayStride;
          geometry.indexOffset = submesh.offset + index_offset;
          geometry.materialID = 0;
          geometry.indexSize = submesh.size;
          geometry.geometryTransform = {};
          meshRecord->second.submesh_geometry.push_back(geometry);

          if (config.enableRayTracing) {
            RHI::BLASDescriptor& blasDesc = meshRecord->second.blas_desc;
            blasDesc.triangleGeometries.push_back(RHI::BLASTriangleGeometry{
                nullptr, nullptr, nullptr, RHI::IndexFormat::UINT32_T,
                uint32_t(mesh->positionBuffer_host.size / (sizeof(float) * 3)),
                geometry.vertexOffset, geometry.indexSize / 3,
                uint32_t(geometry.indexOffset * sizeof(uint32_t)),
                RHI::AffineTransformMatrix{},
                (uint32_t)
                    RHI::BLASGeometryFlagBits::NO_DUPLICATE_ANY_HIT_INVOCATION,
                geometry.materialID});
          }
        }
      }
    }
    // if do not have according mesh ref record, add the record
    auto meshRefRecord = sceneDataPack.mesh_ref_record.find(go->entity);
    if (meshRefRecord == sceneDataPack.mesh_ref_record.end()) {
      sceneDataPack.geometryDirty = true;
      sceneDataPack.mesh_ref_record[go->entity] = {};
      meshRefRecord = sceneDataPack.mesh_ref_record.find(go->entity);
      {  // create mesh record
        int mesh_primitive_type = meshref->customPrimitiveFlag;
        meshRefRecord->second.mesh = mesh;
        meshRefRecord->second.meshReference = meshref;
        uint32_t geometry_start = sceneDataPack.geometry_buffer_cpu.size();
        uint32_t offset = 0;
        for (auto& iter : meshRecord->second.submesh_geometry) {
          meshRefRecord->second.geometry_indices.push_back(
              sceneDataPack.geometry_buffer_cpu.size());
          GFX::Material* mat = meshrenderer->materials[offset++];
          // check whether material has been registered
          auto findMat = sceneDataPack.material_record.find(mat);
          if (findMat == sceneDataPack.material_record.end()) {
            uint32_t matID = sceneDataPack.material_buffer_cpu.size();
            MaterialData matData;

            Core::GUID baseTexGUID = mat->textures["base_color"].guid;
            Core::GUID normTexGUID = mat->textures["normal_bump"].guid;

            auto getTexID = [&](Core::GUID guid) -> uint32_t {
              RHI::TextureView* normTexView =
                  Core::ResourceManager::get()
                      ->getResource<GFX::Texture>(guid)
                      ->originalView.get();
              auto findTex = sceneDataPack.textureview_record.find(normTexView);
              if (findTex == sceneDataPack.textureview_record.end()) {
                uint32_t texID = sceneDataPack.unbinded_textures.size();
                sceneDataPack.unbinded_textures.push_back(normTexView);
                sceneDataPack.textureview_record[normTexView] = texID;
                findTex = sceneDataPack.textureview_record.find(normTexView);
              }
              return findTex->second;
            };
            matData.bsdf_id = mat->BxDF;
            if (baseTexGUID == 0) {
              baseTexGUID = GFX::GFXManager::get()->registerTextureResource("content/textures/white.png");
            }
            matData.baseOrDiffuseTextureIndex = getTexID(baseTexGUID);
            matData.normalTextureIndex =
                (normTexGUID == Core::INVALID_GUID || normTexGUID == 0)
                    ? -1
                    : getTexID(normTexGUID);
            matData.alphaCutoff = mat->alphaThreshold;
            matData.roughness = mat->roughness;
            matData.metalness = mat->metalness;
            matData.baseOrDiffuseColor = mat->baseOrDiffuseColor;
            matData.specularColor = mat->specularColor;
            matData.emissiveColor = mat->emissiveColor;
            matData.transmissionFactor = mat->eta;
            sceneDataPack.material_buffer_cpu.push_back(matData);
            sceneDataPack.material_record[mat] = matID;
            findMat = sceneDataPack.material_record.find(mat);
          }

          int primitiveType = meshref->customPrimitiveFlag;
          if (mat->alphaState != GFX::Material::AlphaState::Opaque) {
            primitiveType += 1;
            mesh_primitive_type = primitiveType;
          }

          GeometryDrawData geometrydata = iter;
          geometrydata.geometryTransform = objectMat;
          geometrydata.geometryTransformInverse = Math::inverse(objectMat);
          geometrydata.oddNegativeScaling = oddScaling >= 0 ? 1.f : -1.f;
          geometrydata.materialID = findMat->second;
          geometrydata.primitiveType = primitiveType;
          geometrydata.lightID = lightComponenet ? 0 : 4294967295;
          uint32_t geometry_id = sceneDataPack.geometry_buffer_cpu.size();
          sceneDataPack.geometry_buffer_cpu.push_back(geometrydata);

          // Add indirect draw info
          {
            RACommon::DrawIndexedIndirectEX drawcall = {
                geometrydata.indexSize,    1, geometrydata.indexOffset,
                geometrydata.vertexOffset, 0, geometry_id};
            if (geometrydata.primitiveType == 0 ||
                geometrydata.primitiveType == 2) {
              raCommon.structured_drawcalls.opaque_drawcalls_host.push_back(
                  drawcall);
            } else if (geometrydata.primitiveType == 1 ||
                       geometrydata.primitiveType == 3) {
              raCommon.structured_drawcalls.alphacut_drawcalls_host.push_back(
                  drawcall);
            }
            raCommon.structured_drawcalls.bsdf_drawcalls_host[mat->BxDF]
                .push_back(drawcall);
          }
        }
        meshRefRecord->second.blasInstance.instanceCustomIndex = geometry_start;
        meshRefRecord->second.primitiveType = mesh_primitive_type;
      }
    }
    // if do not have according light record, add the record
    if (lightComponenet) {
      auto entityLightCompRecord =
          sceneDataPack.light_comp_record.find(go->entity);
      if (entityLightCompRecord == sceneDataPack.light_comp_record.end()) {
        if (lightComponenet->type ==
            GFX::LightComponent::LightType::ENVIRONMENT) {
          sceneDataPack.sceneInfoUniform.env_map =
              packTexture(this, lightComponenet->texture->guid);
        } else {
          sceneDataPack.light_comp_record[go->entity] =
              SceneDataPack::EntityLightRecord{};
          entityLightCompRecord =
              sceneDataPack.light_comp_record.find(go->entity);
          entityLightCompRecord->second.scaling = scaling;
          // if mesh reference is pointing to a triangle mesh
          if (meshref->customPrimitiveFlag == 0 ||
              meshref->customPrimitiveFlag == 1) {
            float totalArea = 0;
            for (uint32_t geoidx : meshRefRecord->second.geometry_indices) {
              sceneDataPack.geometry_buffer_cpu[geoidx].lightID =
                  sceneDataPack.light_buffer_cpu.size();
              entityLightCompRecord->second.lightIndices.push_back(
                  sceneDataPack.light_buffer_cpu.size());
              // TODO
              //sceneDataPack.light_buffer_cpu.push_back(LightData{
              //    1,  // type 1, sphere area light
              //    lightComponenet->intensity,
              //    geoidx,
              //    0,
              //});
              // GFX::Mesh* meshPtr = meshRefRecord->second.meshReference->mesh;
              size_t primitiveNum = meshRefRecord->second.meshReference->mesh
                                        ->indexBuffer_host.size /
                                    (3 * sizeof(uint32_t));
              uint32_t* indexBuffer =
                  static_cast<uint32_t*>(meshRefRecord->second.meshReference
                                             ->mesh->indexBuffer_host.data);
              float* vertexBuffer =
                  static_cast<float*>(meshRefRecord->second.meshReference->mesh
                                          ->vertexBuffer_host.data);
              size_t vertexStride = sizeof(InterleavedVertex) / sizeof(float);
              std::vector<float> areas;
              for (size_t i = 0; i < primitiveNum; ++i) {
                uint32_t i0 = indexBuffer[3 * i + 0];
                uint32_t i1 = indexBuffer[3 * i + 1];
                uint32_t i2 = indexBuffer[3 * i + 2];
                Math::vec3 const& pos0 =
                    *(Math::vec3*)(&(vertexBuffer[i0 * vertexStride]));
                Math::vec3 const& pos1 =
                    *(Math::vec3*)(&(vertexBuffer[i1 * vertexStride]));
                Math::vec3 const& pos2 =
                    *(Math::vec3*)(&(vertexBuffer[i2 * vertexStride]));
                Math::vec3 v0 = Math::vec3(objectMat * Math::vec4(pos0, 0));
                Math::vec3 v1 = Math::vec3(objectMat * Math::vec4(pos1, 0));
                Math::vec3 v2 = Math::vec3(objectMat * Math::vec4(pos2, 0));
                Math::vec3 const e1 = v1 - v0;
                Math::vec3 const e2 = v2 - v0;
                float area = Math::length(Math::cross(e1, e2)) / 2;
                areas.push_back(area);
                totalArea += area;
              }
              // create dist1D
              // entityLightCompRecord->second.tableDist = areas;
              entityLightCompRecord->second.tableDists.emplace_back(
                  TableDist1D{areas});
              entityLightCompRecord->second.lightPowers.emplace_back(
                  totalArea * 3.1415926 *
                  luminance(lightComponenet->intensity));
              sceneDataPack.geometry_buffer_cpu[geoidx].surfaceArea = totalArea;
            }
          }
          // if mesh reference is pointing to a sphere mesh
          else if (meshref->customPrimitiveFlag == 2 ||
                   meshref->customPrimitiveFlag == 3) {
            entityLightCompRecord->second.lightIndices.push_back(
                sceneDataPack.light_buffer_cpu.size());
            // TODO
            //sceneDataPack.light_buffer_cpu.push_back(LightData{
            //    0,  // type 0, sphere area light
            //    lightComponenet->intensity,
            //    meshRefRecord->second.geometry_indices[0],
            //    0,
            //});
            float const radius =
                Math::length(Math::vec3(objectMat * Math::vec4(1, 0, 0, 1)) -
                             Math::vec3(objectMat * Math::vec4(0, 0, 0, 1)));
            float const surfaceArea = 4 * 3.1415926 * radius * radius;
            entityLightCompRecord->second.lightPowers.emplace_back(
                surfaceArea * 3.1415926 *
                luminance(lightComponenet->intensity));
            sceneDataPack
                .geometry_buffer_cpu[meshRefRecord->second.geometry_indices[0]]
                .surfaceArea = surfaceArea;
          }
        }
      }
    }
  }
  // TODO :: do light sampling precomputation
  {
    std::vector<float> lightPowerArray = {};
    for (auto& lightCompRecord : sceneDataPack.light_comp_record) {
      uint32_t i = 0;
      for (uint32_t subLightIdx : lightCompRecord.second.lightIndices) {
        auto& light = sceneDataPack.light_buffer_cpu[subLightIdx];
        // push light power
        if (lightCompRecord.second.lightPowers.size() == 0)
          // TODO
          continue;
        lightPowerArray.push_back(lightCompRecord.second.lightPowers[i]);
        // push light dist array
        if (lightCompRecord.second.tableDists.size() == 0) {
          continue;
        } else {
            // TODO
          //light.areaData.sample_dist_size_0 =
          //    lightCompRecord.second.tableDists[i].pmf.size();
          //light.areaData.sample_dist_offset_pmf_0 =
          //    sceneDataPack.sample_dist_buffer_cpu.size();
          //sceneDataPack.sample_dist_buffer_cpu.insert(
          //    sceneDataPack.sample_dist_buffer_cpu.end(),
          //    lightCompRecord.second.tableDists[i].pmf.begin(),
          //    lightCompRecord.second.tableDists[i].pmf.end());
          //light.areaData.sample_dist_offset_cdf_0 =
          //    sceneDataPack.sample_dist_buffer_cpu.size();
          sceneDataPack.sample_dist_buffer_cpu.insert(
              sceneDataPack.sample_dist_buffer_cpu.end(),
              lightCompRecord.second.tableDists[i].cdf.begin(),
              lightCompRecord.second.tableDists[i].cdf.end());
          ++i;
        }
      }
    }
    sceneDataPack.lightTableDist = TableDist1D{lightPowerArray};
    sceneDataPack.sceneInfoUniform.light_num =
        sceneDataPack.lightTableDist.pmf.size();
    sceneDataPack.sceneInfoUniform.light_offset_pmf =
        sceneDataPack.sample_dist_buffer_cpu.size();
    sceneDataPack.sample_dist_buffer_cpu.insert(
        sceneDataPack.sample_dist_buffer_cpu.end(),
        sceneDataPack.lightTableDist.pmf.begin(),
        sceneDataPack.lightTableDist.pmf.end());
    sceneDataPack.sceneInfoUniform.light_offset_cdf =
        sceneDataPack.sample_dist_buffer_cpu.size();
    sceneDataPack.sample_dist_buffer_cpu.insert(
        sceneDataPack.sample_dist_buffer_cpu.end(),
        sceneDataPack.lightTableDist.cdf.begin(),
        sceneDataPack.lightTableDist.cdf.end());
    // also set pmf
    size_t i = 0;
    for (auto& lightCompRecord : sceneDataPack.light_comp_record) {
      for (uint32_t subLightIdx : lightCompRecord.second.lightIndices) {
        auto& light = sceneDataPack.light_buffer_cpu[subLightIdx];
        // TODO
        //if (light.areaData.lightType == 0)
        //  light.areaData.total_value = sceneDataPack.lightTableDist.pmf[i++];
      }
    }
  }
  // if geometry is dirty, rebuild all the buffers
  if (sceneDataPack.geometryDirty) {
    sceneDataPack.vertex_buffer = device->createDeviceLocalBuffer(
        sceneDataPack.vertex_buffer_cpu.data(),
        sceneDataPack.vertex_buffer_cpu.size() * sizeof(float),
        (uint32_t)RHI::BufferUsage::VERTEX |
            (uint32_t)RHI::BufferUsage::SHADER_DEVICE_ADDRESS |
            (uint32_t)RHI::BufferUsage::STORAGE);
    sceneDataPack.index_buffer = device->createDeviceLocalBuffer(
        sceneDataPack.index_buffer_cpu.data(),
        sceneDataPack.index_buffer_cpu.size() * sizeof(uint32_t),
        (uint32_t)RHI::BufferUsage::INDEX |
            (uint32_t)RHI::BufferUsage::SHADER_DEVICE_ADDRESS |
            (uint32_t)
                RHI::BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY |
            (uint32_t)RHI::BufferUsage::STORAGE);
    sceneDataPack.material_buffer = device->createDeviceLocalBuffer(
        sceneDataPack.material_buffer_cpu.data(),
        sceneDataPack.material_buffer_cpu.size() * sizeof(MaterialData),
        (uint32_t)RHI::BufferUsage::SHADER_DEVICE_ADDRESS |
            (uint32_t)RHI::BufferUsage::STORAGE);
    sceneDataPack.light_buffer = device->createDeviceLocalBuffer(
        sceneDataPack.light_buffer_cpu.data(),
        sceneDataPack.light_buffer_cpu.size() * sizeof(PolymorphicLightInfo),
        (uint32_t)RHI::BufferUsage::SHADER_DEVICE_ADDRESS |
            (uint32_t)RHI::BufferUsage::STORAGE);
    sceneDataPack.sample_dist_buffer = device->createDeviceLocalBuffer(
        sceneDataPack.sample_dist_buffer_cpu.data(),
        sceneDataPack.sample_dist_buffer_cpu.size() * sizeof(float),
        (uint32_t)RHI::BufferUsage::SHADER_DEVICE_ADDRESS |
            (uint32_t)RHI::BufferUsage::STORAGE);
    if (config.enableRayTracing) {
      sceneDataPack.position_buffer = device->createDeviceLocalBuffer(
          sceneDataPack.position_buffer_cpu.data(),
          sceneDataPack.position_buffer_cpu.size() * sizeof(float),
          (uint32_t)RHI::BufferUsage::VERTEX |
              (uint32_t)RHI::BufferUsage::SHADER_DEVICE_ADDRESS |
              (uint32_t)RHI::BufferUsage::
                  ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY |
              (uint32_t)RHI::BufferUsage::STORAGE);
    }
    sceneDataBuffers.geometry_buffer =
        GFX::GFXManager::get()
            ->createStructuredArrayMultiStorageBuffer<GeometryDrawData>(
                sceneDataPack.geometry_buffer_cpu.size());
    // set scene data pack not dirty
    sceneDataPack.geometryDirty = false;
  }

  // create BLAS / TLAS
  for (auto& iter : sceneDataPack.mesh_record) {
    if (iter.second.blases == nullptr) {
      iter.second.blas_desc.allowRefitting = true;
      iter.second.blas_desc.allowCompaction = true;
      for (auto& mesh : iter.second.blas_desc.triangleGeometries) {
        mesh.positionBuffer = sceneDataPack.position_buffer.get();
        mesh.indexBuffer = sceneDataPack.index_buffer.get();
      }
      if (device->getRayTracingExtension())
        iter.second.blases = device->createBLAS(iter.second.blas_desc);
    }
  }
  for (auto& iter : sceneDataPack.mesh_ref_record) {
    // if BLAS is not created
    if (iter.second.primitiveType == 0 || iter.second.primitiveType == 1) {
      iter.second.blasInstance.blas =
          sceneDataPack.mesh_record[iter.second.mesh].blases.get();
      iter.second.blasInstance.instanceShaderBindingTableRecordOffset =
          iter.second.primitiveType;
    } else if (iter.second.primitiveType == 2 ||
               iter.second.primitiveType == 3) {
      iter.second.blasInstance.blas =
          customPrimitive.sphere.meshRecord.blases.get();
      iter.second.blasInstance.instanceShaderBindingTableRecordOffset =
          iter.second.primitiveType;
    }
    sceneDataPack.tlas_desc.instances.push_back(iter.second.blasInstance);
  }
  sceneDataPack.tlas_desc.allowRefitting = true;

  if (device->getRayTracingExtension())
    sceneDataPack.tlas = device->createTLAS(sceneDataPack.tlas_desc);

  raCommon.structured_drawcalls.buildIndirectDrawcalls();
  // for (int i = 0; i < 2; ++i) {
  //	commonDescData.set1_flights_rt[i] =
  // device->createBindGroup(RHI::BindGroupDescriptor{
  //		commonDescData.set1_layout_rt.get(),
  //		std::vector<RHI::BindGroupEntry>{
  //			{0,RHI::BindingResource{sceneDataPack.tlas.get()}},
  //			{1,RHI::BindingResource{rdgraph->getTexture("TracerTarget_Color")->texture->originalView.get()}},
  //	} });
  //	commonDescData.set1_flights_rt_array[i] =
  // commonDescData.set1_flights_rt[i].get();
  // }

  // rebuild all bind groups
  for (int i = 0; i < 2; ++i) {
    commonDescData.set0_flights_resources[i] =
        std::vector<RHI::BindGroupEntry> {
      {0,
       RHI::BindingResource{
           sceneDataBuffers.global_uniform_buffer.getBufferBinding(i)}},
          {1, RHI::BindingResource{{sceneDataPack.vertex_buffer.get(), 0,
                                    sceneDataPack.vertex_buffer.get() != 0
                                        ? sceneDataPack.vertex_buffer->size()
                                        : 0}}},
        {2, RHI::BindingResource{{sceneDataPack.index_buffer.get(), 0,
                                  sceneDataPack.index_buffer.get() != 0
                                      ? sceneDataPack.index_buffer->size()
                                      : 0}}},
        {3,
         RHI::BindingResource{
             sceneDataBuffers.geometry_buffer.getBufferBinding(i)}},
        {4, RHI::BindingResource{{sceneDataPack.material_buffer.get(), 0,
                                  sceneDataPack.material_buffer.get() != 0
                                      ? sceneDataPack.material_buffer->size()
                                      : 0}}},
        {5, RHI::BindingResource{{sceneDataPack.light_buffer.get(), 0,
                                  sceneDataPack.light_buffer.get() != 0
                                      ? sceneDataPack.light_buffer->size()
                                      : 0}}},
        {6, RHI::BindingResource{{sceneDataPack.sample_dist_buffer.get(), 0,
                                  sceneDataPack.sample_dist_buffer.get() != 0
                                      ? sceneDataPack.sample_dist_buffer->size()
                                      : 0}}},
        {7,
         RHI::BindingResource{
             sceneDataBuffers.scene_info_buffer.getBufferBinding(i)}},
        {8,
         RHI::BindingResource{
             sceneDataPack.unbinded_textures,
             Core::ResourceManager::get()
                 ->getResource<GFX::Sampler>(
                     GFX::GFXManager::get()->commonSampler.defaultSampler)
                 ->sampler.get()}},
    };

    commonDescData.set1_flights_resources[i] = std::vector<RHI::BindGroupEntry>{
        {0, RHI::BindingResource{sceneDataPack.tlas.get()}},
    };
  }
}

auto SRenderer::updateCamera(GFX::TransformComponent const& transform,
                             GFX::CameraComponent const& camera,
                             Math::ivec2 const& viewport) noexcept
    -> void {
  RHI::MultiFrameFlights* multiFrameFlights =
      GFX::GFXManager::get()->rhiLayer->getMultiFrameFlights();
  GlobalUniforms globalUni;

  raCommon.mainCamera = &camera;
  {
    CameraData& camData = globalUni.cameraData;

    camData.nearZ = camera.near;
    camData.farZ = camera.far;

    camData.posW = transform.translation;
    camData.target = transform.translation + transform.getRotatedForward();

    camData.viewMat = Math::transpose(
        Math::lookAt(camData.posW, camData.target, Math::vec3(0, 1, 0)).m);
    camData.invViewMat = Math::inverse(camData.viewMat);
    camData.projMat = Math::transpose(camera.getProjectionMat());
    camData.invProjMat = Math::inverse(camData.projMat);
    raCommon.mainCameraInfo.view = Math::transpose(camData.viewMat);

    camData.viewProjMat = camData.viewMat * camData.projMat;
    camData.invViewProj = Math::inverse(camData.viewProjMat);

    // Ray tracing related vectors
    camData.focalDistance = 1;
    camData.aspectRatio = camera.aspect;
    camData.up = Math::vec3(0, 1, 0);
    camData.cameraW =
        Math::normalize(camData.target - camData.posW) * camData.focalDistance;
    camData.cameraU = Math::normalize(Math::cross(camData.cameraW, camData.up));
    camData.cameraV =
        Math::normalize(Math::cross(camData.cameraU, camData.cameraW));
    const float ulen = camData.focalDistance *
                       std::tan(Math::radians(camera.fovy) * 0.5f) *
                       camData.aspectRatio;
    camData.cameraU *= ulen;
    const float vlen =
        camData.focalDistance * std::tan(Math::radians(camera.fovy) * 0.5f);
    camData.cameraV *= vlen;

    camData.jitterX = 0;
    camData.jitterY = 0;

    camData.clipToWindowScale = float2(0.5f * viewport.x, -0.5f * viewport.y);
    camData.clipToWindowBias = float2(0.f) + float2(viewport) * 0.5f;

    camData.rectArea =
        4 * ulen * vlen / (camData.focalDistance * camData.focalDistance);
  }
  if (globalUniRecord.cameraData.viewMat != globalUni.cameraData.viewMat ||
      globalUniRecord.cameraData.projMat != globalUni.cameraData.projMat)
    state.batchIdx = 0;
  globalUniRecord = globalUni;
  sceneDataBuffers.global_uniform_buffer.setStructure(
      globalUni, multiFrameFlights->getFlightIndex());

  state.allBatch++;
}

auto SRenderer::updateRDGData(RDG::Graph* graph) noexcept -> void {
  uint32_t flightIdx = GFX::GFXManager::get()
                           ->rhiLayer->getMultiFrameFlights()
                           ->getFlightIndex();
  uint32_t prevFlightIdx = (flightIdx + 1) % 2;
  graph->renderData.setBindingResource(
      "GlobalUniforms",
      sceneDataBuffers.global_uniform_buffer.getBufferBinding(flightIdx));
  graph->renderData.setBindingResource(
      "PrevGlobalUniforms",
      sceneDataBuffers.global_uniform_buffer.getBufferBinding(prevFlightIdx));
  graph->renderData.setBindingResource(
      "PrevGeometryBuffer",
      sceneDataBuffers.geometry_buffer.getBufferBinding(prevFlightIdx));

  graph->renderData.setBindGroupEntries(
      "CommonScene", &(commonDescData.set0_flights_resources[flightIdx]));
  graph->renderData.setBindGroupEntries(
      "CommonRT", &(commonDescData.set1_flights_resources[flightIdx]));
  graph->renderData.setUInt("AccumIdx", state.batchIdx++);
  graph->renderData.setUInt("FrameIdx", state.allBatch);
  graph->renderData.setUVec2("TargetSize", {state.width, state.height});
  graph->renderData.setPtr("CameraData", &(globalUniRecord.cameraData));
  graph->renderData.setMat4("ViewProj", globalUniRecord.cameraData.viewProjMat);
  graph->renderData.setPtr("SceneAABB", &(statisticsData.aabb));
  graph->renderData.setDelegate(
      "IssueAllDrawcalls",
      [&, flightIdx = flightIdx](RDG::RenderData::DelegateData const& data) {
        if (sceneDataPack.geometry_buffer_cpu.size() > 0) {
          data.passEncoder.render->setIndexBuffer(
              sceneDataPack.index_buffer.get(), RHI::IndexFormat::UINT32_T, 0,
              sceneDataPack.index_buffer->size());
          data.passEncoder.render->setBindGroup(
              0, data.pipelinePass->bindgroups[0][flightIdx].get(), 0, 0);
          uint32_t geometry_idx = 0;
          for (auto& geometry : sceneDataPack.geometry_buffer_cpu) {
            data.passEncoder.render->pushConstants(
                &geometry_idx, (uint32_t)RHI::ShaderStages::VERTEX, 0,
                sizeof(uint32_t));
            data.passEncoder.render->drawIndexed(geometry.indexSize, 1,
                                                 geometry.indexOffset,
                                                 geometry.vertexOffset, 0);
            geometry_idx++;
          }
        }
      });
  graph->renderData.setDelegate(
      "IssueDrawcalls_LightOnly",
      [&, flightIdx = flightIdx](RDG::RenderData::DelegateData const& data) {
        if (sceneDataPack.geometry_buffer_cpu.size() > 0) {
          data.passEncoder.render->setIndexBuffer(
              sceneDataPack.index_buffer.get(), RHI::IndexFormat::UINT32_T, 0,
              sceneDataPack.index_buffer->size());
          data.passEncoder.render->setBindGroup(
              0, data.pipelinePass->bindgroups[0][flightIdx].get(), 0, 0);
          uint32_t geometry_idx = 0;
          for (auto& geometry : sceneDataPack.geometry_buffer_cpu) {
            if (geometry.lightID != 4294967295) {
              data.passEncoder.render->pushConstants(
                  &geometry_idx, (uint32_t)RHI::ShaderStages::VERTEX, 0,
                  sizeof(uint32_t));
              data.passEncoder.render->drawIndexed(geometry.indexSize, 1,
                                                   geometry.indexOffset,
                                                   geometry.vertexOffset, 0);
            }
            geometry_idx++;
          }
        }
      });
  graph->renderData.setDelegate(
      "PrepareDrawcalls",
      [&, flightIdx = flightIdx](RDG::RenderData::DelegateData const& data) {
        data.passEncoder.render->setIndexBuffer(
            sceneDataPack.index_buffer.get(), RHI::IndexFormat::UINT32_T, 0,
            sceneDataPack.index_buffer->size());
        data.passEncoder.render->setBindGroup(
            0, data.pipelinePass->bindgroups[0][flightIdx].get(), 0, 0);
      });
}

SRenderer::TableDist1D::TableDist1D(std::vector<float> const& f) {
  pmf = f;
  cdf = std::vector<float>(f.size() + 1);
  cdf[0] = 0;
  for (int i = 0; i < (int)f.size(); i++) cdf[i + 1] = cdf[i] + pmf[i];
  float total = cdf.back();
  if (total > 0)
    for (int i = 0; i < (int)pmf.size(); i++) {
      pmf[i] /= total;
      cdf[i] /= total;
    }
  else {
    for (int i = 0; i < (int)pmf.size(); i++) {
      pmf[i] = float(1) / float(pmf.size());
      cdf[i] = float(i) / float(pmf.size());
    }
    cdf.back() = 1;
  }
  cdf.back() = 1;
}

auto SRenderer::init() noexcept -> void {
  GFX::GFXManager::get()
      ->registerMaterialTemplate(0, "Lambertian")
      .addConstantData("albedo_tint", RHI::DataFormat::FLOAT32X4)
      .addConstantData("uv_tiling", RHI::DataFormat::FLOAT32X2)
      .addConstantData("uv_scaling", RHI::DataFormat::FLOAT32X2)
      .addConstantData("mat_type", RHI::DataFormat::UINT32)
      .addTexture("basecolor_opacity_tex")
      .addTexture("normal_bump_tex")
      .addTexture("emission_tex");
}

#pragma endregion
}