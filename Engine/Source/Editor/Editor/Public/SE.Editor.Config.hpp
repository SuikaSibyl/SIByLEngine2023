#pragma once
#include <compare>
#include <filesystem>
#include <functional>
#include <Resource/SE.Core.Resource.hpp>
#include <SE.GFX.hpp>
#include <SE.GFX-Script.hpp>
#include <SE.Video.hpp>
#include <SE.Editor.Core.hpp>
#include <SE.Editor.GFX.hpp>
#include <SE.Editor.RDG.hpp>

namespace SIByL::Editor {
SE_EXPORT struct Config {
  /** build editor layer up in a pre-defined way */
  static inline auto buildEditorLayer(Editor::EditorLayer* layer) noexcept
      -> void {
    layer->registerWidget<Editor::SceneWidget>();
    layer->registerWidget<Editor::InspectorWidget>();
    layer->registerWidget<Editor::ContentWidget>();
    layer->registerWidget<Editor::LogWidget>();
    layer->registerWidget<Editor::ViewportWidget>();
    layer->registerWidget<Editor::StatusWidget>();
    layer->registerWidget<Editor::RDGViewerWidget>();

    layer->getWidget<Editor::SceneWidget>()->bindScene(nullptr);
    layer->getWidget<Editor::SceneWidget>()->inspectorWidget =
        layer->getWidget<Editor::InspectorWidget>();
    layer->getWidget<Editor::SceneWidget>()->viewportWidget =
        layer->getWidget<Editor::ViewportWidget>();

    GameObjectInspector& gameobjectInspector =
        layer->getWidget<Editor::SceneWidget>()->gameobjectInspector;
    // Register component fragment to show component ui on the panel
    gameobjectInspector.registerFragment<Editor::TagComponentFragment>();
    gameobjectInspector.registerFragment<Editor::TransformComponentFragment>();
    gameobjectInspector.registerFragment<Editor::MeshReferenceComponentFragment>();
    gameobjectInspector.registerFragment<Editor::MeshRendererComponentFragment>();
    gameobjectInspector.registerFragment<Editor::CameraComponentFragment>();
    gameobjectInspector.registerFragment<Editor::LightComponentFragment>();
    gameobjectInspector.registerFragment<Editor::NativeScriptComponentFragment>();
    // Register component can be added through editor pannel
    gameobjectInspector.registerComponent<GFX::MeshReference>("Mesh Reference");
    gameobjectInspector.registerComponent<GFX::MeshRenderer>("Mesh Renderer");
    gameobjectInspector.registerComponent<GFX::LightComponent>("Light Component");
    gameobjectInspector.registerComponent<NativeScriptComponent>("Native Script");

    Editor::ResourceViewer& resource_viewer =
        layer->getWidget<Editor::InspectorWidget>()->resourceViewer;
    resource_viewer.registerElucidator<Editor::TextureElucidator>(
        "struct SIByL::GFX::Texture");
    resource_viewer.registerElucidator<Editor::MaterialElucidator>(
        "struct SIByL::GFX::Material");
    resource_viewer.registerElucidator<Editor::MeshElucidator>(
        "struct SIByL::GFX::Mesh");
    resource_viewer.registerElucidator<Editor::VideoClipElucidator>(
        "struct SIByL::GFX::VideoClip");

    layer->getWidget<Editor::ContentWidget>()->inspectorWidget =
        layer->getWidget<Editor::InspectorWidget>();
    layer->getWidget<Editor::ContentWidget>()->reigsterIconResources();

    Editor::ContentWidget* contentWidget =
        layer->getWidget<Editor::ContentWidget>();
    contentWidget->registerResource<GFX::Scene>(contentWidget->icons.scene,
                                                {".scene"}, nullptr);
    contentWidget->registerResource<GFX::Mesh>(
        contentWidget->icons.mesh, {".obj", ".fbx", ".gltf"}, nullptr);
    contentWidget->registerResource<GFX::Texture>(
        contentWidget->icons.image,
        {".jpg", ".jpeg", ".png", ".PNG", ".tga", ".TGA"},
        std::bind(&GFX::GFXManager::requestOfflineTextureResource,
                  GFX::GFXManager::get(), std::placeholders::_1),
        [](char const* path) {
          return GFX::GFXManager::get()->registerTextureResource(path);
        });
    contentWidget->registerResource<GFX::ShaderModule>(
        contentWidget->icons.shader, {".glsl", "spv"}, nullptr);
    contentWidget->registerResource<GFX::Material>(
        contentWidget->icons.material, {".mat"},
        std::bind(&GFX::GFXManager::requestOfflineMaterialResource,
                  GFX::GFXManager::get(), std::placeholders::_1),
        [](char const* path) {
          return GFX::GFXManager::get()->registerMaterialResource(path);
        });
    contentWidget->registerResource<GFX::VideoClip>(
        contentWidget->icons.video, {".mkv"},
        std::bind(&GFX::VideExtension::requestOfflineVideoClipResource,
                  GFX::GFXManager::get()->getExt<GFX::VideExtension>(
                      GFX::Ext::VideoClip),
                  std::placeholders::_1),
        std::bind(&GFX::VideExtension::registerVideoClipResource,
                  GFX::GFXManager::get()->getExt<GFX::VideExtension>(
                      GFX::Ext::VideoClip),
                  std::placeholders::_1));
  }
};
}  // namespace SIByL::Editor