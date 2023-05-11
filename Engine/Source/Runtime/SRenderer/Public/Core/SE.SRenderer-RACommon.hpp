#pragma once
#include <array>
#include <compare>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <typeinfo>
#include <unordered_map>
#include <vector>
#include <Resource/SE.Core.Resource.hpp>
#include <SE.Math.Geometric.hpp>
#include <SE.RHI.hpp>
#include <SE.GFX.hpp>

namespace SIByL {
SE_EXPORT struct RACommon {
  RACommon() { singleton = this; }
  static auto get() noexcept -> RACommon* { return singleton; }

  struct DirectionalLightInfo {
    Math::mat4 transform;
    uint32_t lightID;
  };
  std::optional<DirectionalLightInfo> mainDirectionalLight = std::nullopt;

  struct ShadowmapInfo {
    Math::mat4 viewProj;
  };
  std::vector<ShadowmapInfo> shadowmapData;

  struct alignas(64) CascadeShadowmapData {
    Math::mat4 cascade_transform_0;
    Math::mat4 cascade_transform_1;
    Math::mat4 cascade_transform_2;
    Math::mat4 cascade_transform_3;
    Math::vec4 cascade_depths;
  } mainLightCSM;
  GFX::StructuredUniformBufferView<CascadeShadowmapData> csm_info_device;

  std::vector<Math::mat4> cascade_views;
  Math::vec4 cascade_distances;

  Math::bounds3 sceneAABB;

  // indexed
  struct DrawIndexedIndirectCmd {
    uint32_t indexCount;
    uint32_t instanceCount;
    uint32_t firstIndex;
    int32_t vertexOffset;
    uint32_t firstInstance;
  };

  struct DrawIndexedIndirectEX {
    uint32_t indexCount;
    uint32_t instanceCount;
    uint32_t firstIndex;
    int32_t vertexOffset;
    uint32_t firstInstance;
    uint32_t geometryID;
    uint32_t padding0;
    uint32_t padding1;
  };

  // non indexed
  struct DrawIndirectCmd {
    uint32_t vertexCount;
    uint32_t instanceCount;
    uint32_t firstVertex;
    uint32_t firstInstance;
  };

  struct IndirectDrawcall {
    uint64_t offset;
    uint32_t drawCount;
  };

  struct DrawcallData {
    std::vector<DrawIndexedIndirectEX> opaque_drawcalls_host;
    std::vector<DrawIndexedIndirectEX> alphacut_drawcalls_host;
    std::unordered_map<uint32_t, std::vector<DrawIndexedIndirectEX>>
        bsdf_drawcalls_host;

    IndirectDrawcall opaque_drawcall;
    IndirectDrawcall alphacut_drawcall;
    std::unordered_map<uint32_t, IndirectDrawcall> bsdfs_drawcalls;

    std::vector<DrawIndexedIndirectEX> all_drawcall_host;
    GFX::Buffer* all_drawcall_device = nullptr;

    auto buildIndirectDrawcalls() noexcept -> void;

  } structured_drawcalls;

  GFX::CameraComponent const* mainCamera;
  struct {
    Math::mat4 view;
  } mainCameraInfo;

 private:
  static RACommon* singleton;
};
}  // namespace SIByL