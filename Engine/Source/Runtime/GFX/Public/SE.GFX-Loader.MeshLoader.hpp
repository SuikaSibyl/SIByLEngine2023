#pragma once
#include <common_config.hpp>
#include <filesystem>
#include <SE.GFX-GFXConfig.hpp>
#include <Resource/SE.Core.Resource.hpp>

namespace SIByL::GFX {
SE_EXPORT auto getVertexBufferLayout(MeshDataLayout const& mdl) noexcept
    -> RHI::VertexBufferLayout;

SE_EXPORT struct MeshLoader_OBJ {
  static auto loadOBJ(std::filesystem::path const& path,
                      MeshDataLayout const& layout, Core::Buffer* vertexBuffer,
                      Core::Buffer* indexBuffer,
                      Core::Buffer* vertexPosOnlyBuffer = nullptr) noexcept
      -> void;

  static auto loadMeshResource(std::filesystem::path const& path,
                               MeshDataLayout const& layout,
                               bool usePosOnlyBuffer) noexcept -> Core::GUID;
};
}  // namespace SIByL::GFX