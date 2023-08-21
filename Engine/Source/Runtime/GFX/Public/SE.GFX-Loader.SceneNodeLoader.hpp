#pragma once

#include <filesystem>
#include <SE.GFX-Main.hpp>
#include <SE.GFX-GFXConfig.hpp>

namespace SIByL::GFX {
SE_EXPORT struct SceneNodeLoader_obj {
  /** Load obj file */
  static auto loadSceneNode(std::filesystem::path const& path,
                            GFX::Scene& gfxscene,
                            MeshLoaderConfig meshConfig = {}) noexcept -> void;
  static auto loadSceneNode(std::filesystem::path const& path,
                            GFX::Scene& gfxscene, GameObjectHandle parent,
                            MeshLoaderConfig meshConfig = {}) noexcept
      -> GameObjectHandle;
};

SE_EXPORT struct SceneNodeLoader_glTF {
  /** Load glTF file */
  static auto loadSceneNode(std::filesystem::path const& path,
                            GFX::Scene& gfxscene,
                            MeshLoaderConfig meshConfig = {}) noexcept -> void;
};

SE_EXPORT struct SceneNodeLoader_assimp {
  /** Load obj file */
  static auto loadSceneNode(std::filesystem::path const& path,
                            GFX::Scene& gfxscene,
                            MeshLoaderConfig meshConfig = {}) noexcept -> void;
};

SE_EXPORT struct SceneNodeLoader_mitsuba {
  /** Load obj file */
  static auto loadSceneNode(std::filesystem::path const& path,
                            GFX::Scene& gfxscene,
                            MeshLoaderConfig meshConfig = {}) noexcept -> void;
};
}  // namespace SIByL::GFX