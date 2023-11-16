#pragma once
#include <imgui.h>
#include <imgui_internal.h>
#include <filesystem>
#include <functional>
#include <SE.Editor.Core.hpp>
#include <SE.RHI.hpp>
#include <Resource/SE.Core.Resource.hpp>
#include <SE.Editor.GFX.hpp>
#include <SE.Video.hpp>
#include <SE.Math.Spline.hpp>
#include <Misc/SE.Core.Misc.hpp>
#include <ImGuizmo.h>

namespace SIByL::Editor {
SE_EXPORT auto drawBoolControl(std::string const& label, bool& value,
                               float labelWidth = 50) noexcept -> void;
SE_EXPORT auto drawFloatControl(std::string const& label, float& value,
                                float resetValue = 0,
                                float columeWidth = 100) noexcept -> void;
SE_EXPORT auto drawVec3Control(const std::string& label, Math::vec3& values,
                               float resetValue = 0,
                               float columeWidth = 100) noexcept -> void;
SE_EXPORT auto drawCustomColume(const std::string& label, float columeWidth,
                      std::function<void()> const& func) noexcept -> void;
SE_EXPORT auto to_string(RHI::VertexFormat vertexFormat) noexcept
    -> std::string;
SE_EXPORT auto to_string(RHI::PrimitiveTopology topology) noexcept
    -> std::string;
SE_EXPORT auto to_string(RHI::VertexStepMode stepMode) noexcept -> std::string;
}  // namespace SIByL::Editor

namespace SIByL::Editor {
SE_EXPORT struct TextureUtils {
  static auto getImGuiTexture(Core::GUID guid) noexcept -> ImGuiTexture*;
};
SE_EXPORT struct TextureFragment : public Fragment {
  /** virtual draw gui*/
  virtual auto onDrawGui(uint32_t flags, void* data) noexcept -> void {}
};
}  // namespace SIByL::Editor

namespace SIByL::Editor {
SE_EXPORT struct ResourceElucidator {
  /** override draw gui */
  virtual auto onDrawGui(Core::GUID guid) noexcept -> void = 0;
};

SE_EXPORT struct ResourceViewer {
  /** override draw gui */
  auto onDrawGui(char const* type, Core::GUID guid) noexcept -> void;

   template <class T>
   requires std::derived_from<T, ResourceElucidator>
   auto registerElucidator(char const* type) noexcept -> void {
     elucidatorMaps[type] = std::make_unique<T>();
   }

 private:
  std::unordered_map<char const*, std::unique_ptr<ResourceElucidator>>
      elucidatorMaps;
};

SE_EXPORT struct MeshElucidator : public ResourceElucidator {
  /** override draw gui */
  virtual auto onDrawGui(Core::GUID guid) noexcept -> void override;
  static auto onDrawGui_GUID(Core::GUID guid) noexcept -> void;
  static auto onDrawGui_PTR(GFX::Mesh* mesh) noexcept -> void;
};

SE_EXPORT struct TextureElucidator : public ResourceElucidator {
  /** override draw gui */
  virtual auto onDrawGui(Core::GUID guid) noexcept -> void override;
  static auto onDrawGui_GUID(Core::GUID guid, bool draw_tree = true) noexcept -> void;
  static auto onDrawGui_PTR(GFX::Texture* tex, bool draw_tree = true) noexcept -> void;
};

SE_EXPORT struct MaterialElucidator : public ResourceElucidator {
  /** override draw gui */
  virtual auto onDrawGui(Core::GUID guid) noexcept -> void override;
  static auto onDrawGui_GUID(Core::GUID guid, bool draw_tree = true) noexcept -> void;
  /** override draw gui */
  static auto onDrawGui_PTR(GFX::Material* material, bool draw_tree = true) noexcept -> void;
};

SE_EXPORT struct VideoClipElucidator : public ResourceElucidator {
  /** override draw gui */
  virtual auto onDrawGui(Core::GUID guid) noexcept -> void override;
  static auto onDrawGui_GUID(Core::GUID guid) noexcept -> void;
  /** override draw gui */
  static auto onDrawGui_PTR(GFX::VideoClip* vc) noexcept -> void;
};
}  // namespace SIByL::Editor

namespace SIByL::Editor {
SE_EXPORT struct InspectorWidget : public Widget {
  /** draw gui*/
  virtual auto onDrawGui() noexcept -> void override;
  /** set custom draw */
  auto setCustomDraw(std::function<void()> func) noexcept -> void;
  /** set empty */
  auto setEmpty() noexcept -> void;
  /** custom draw on inspector widget */
  std::function<void()> customDraw;
  /** resource viewer */
  ResourceViewer resourceViewer;
};

SE_EXPORT struct CustomInspector {
  /** draw gui*/
  virtual auto onDrawGui() noexcept -> void = 0;
  /** register a widget to editor */
  template <class T>
  auto registerFragment() noexcept -> void {
    fragments[typeid(T).name()] = std::make_unique<T>();
    fragmentSequence.push_back(fragments[typeid(T).name()].get());
  }
  /** set insepctor widget to show this cutom one */
  auto setInspectorWidget(InspectorWidget* widget) noexcept -> void;
  /** get a widget registered in editor */
  template <class T>
  auto getFragment() noexcept -> T* {
    auto iter = fragments.find(typeid(T).name());
    if (iter == fragments.end())
      return nullptr;
    else
      return static_cast<T*>(iter->second.get());
  }

 protected:
  /** fragment sequences to be drawn */
  std::vector<Fragment*> fragmentSequence = {};
  /** all the widgets registered */
  std::unordered_map<char const*, std::unique_ptr<Fragment>> fragments = {};
};
}  // namespace SIByL::Editor

namespace SIByL::Editor {
SE_EXPORT struct StatusWidget : public Widget {
  Core::Timer* timer;
  /** draw gui*/
  virtual auto onDrawGui() noexcept -> void override;
};
}  // namespace SIByL::Editor

namespace SIByL::Editor {
SE_EXPORT struct LogWidget : public Widget {
  /** constructor */
  LogWidget();
  /** destructor */
  ~LogWidget();
  /** draw gui*/
  virtual auto onDrawGui() noexcept -> void override;
  /** set custom draw */
  auto setCustomDraw(std::function<void()> func) noexcept -> void;
  /** set empty */
  auto clear() noexcept -> void;
  /** add line */
  auto addline(std::string const& str) -> void;
  /** custom draw on inspector widget */
  std::function<void()> customDraw;

  ImGuiTextBuffer _buf;
  ImGuiTextFilter filter;
  ImVector<int> lineOffsets;  // Index to lines offset. We maintain this with
                              // AddLog() calls.
  bool autoScroll;            // Keep scrolling if already at the bottom.
};
}  // namespace SIByL::Editor

namespace SIByL::Editor {
auto captureImage(Core::GUID src) noexcept -> void;

SE_EXPORT struct ImGuizmoState {
  ImGuizmo::OPERATION mCurrentGizmoOperation = ImGuizmo::TRANSLATE;
  ImGuizmo::MODE mCurrentGizmoMode = ImGuizmo::LOCAL;
  Math::mat4 objMatrix;
  bool useSnap = false;
  float snap[3] = {1.f, 1.f, 1.f};
  float width, height;
};

SE_EXPORT struct ViewportWidget : public Widget {
  auto setTarget(std::string const& name, GFX::Texture* tex) noexcept -> void;

  GFX::Texture* texture;
  GFX::CameraComponent* camera;
  GFX::TransformComponent camera_transform;
  GFX::TransformComponent* camera_transform_ref;
  std::string name = "Viewport";
  ImGuizmoState gizmoState;
  bool* forceReset;
  std::optional<GFX::GameObjectHandle> selectedGO = std::nullopt;
  GFX::Scene* selectedScene;

  /** draw gui*/
  virtual auto onDrawGui() noexcept -> void override;
};
}  // namespace SIByL::Editor

namespace SIByL::Editor {
SE_EXPORT struct GameObjectInspector : public CustomInspector {
  /** custom data to be parsed to each fragment */
  struct GameObjectData {
    GFX::GameObjectHandle handle;
    GFX::Scene* scene;
  } data;

  struct IComponentOperator {
    virtual auto addComponent(Core::Entity&) -> void = 0;
    virtual auto getComponent(Core::Entity&) -> void* = 0;
    virtual auto removeComponent(Core::Entity&) -> void = 0;
    virtual auto initComponent(Core::Entity& entity) -> bool = 0;
  };
  template <class T>
  struct ComponentOperator : IComponentOperator {
    virtual auto addComponent(Core::Entity& entity) -> void override {
      entity.addComponent<T>();
    }
    virtual auto getComponent(Core::Entity& entity) -> void* override {
      return (void*)entity.getComponent<T>();
    }
    virtual auto removeComponent(Core::Entity& entity) -> void override {
      entity.removeComponent<T>();
    }
    virtual auto initComponent(Core::Entity& entity) -> bool override {
      return initComponentOnRegister<T>(entity, *(T*)getComponent(entity));
    }
  };
  std::vector<std::pair<std::string, std::unique_ptr<IComponentOperator>>>
      componentsRegister = {};
  template <class T>
  inline auto registerComponent(std::string const& name) -> void {
    componentsRegister.emplace_back(
        std::pair<std::string, std::unique_ptr<IComponentOperator>>(
            name, std::make_unique<ComponentOperator<T>>()));
  }
  /** draw each fragments */
  virtual auto onDrawGui() noexcept -> void;
};

SE_EXPORT struct SceneWidget : public Widget {
  /** virtual draw gui*/
  virtual auto onDrawGui() noexcept -> void override;
  /** draw a scene gameobject and its descents, return whether is deleted */
  auto drawNode(GFX::GameObjectHandle const& node) -> bool;
  /** bind scene to the widget */
  auto bindScene(GFX::Scene* scene) noexcept -> void { this->scene = scene; }
  /** scene binded to be shown on widget */
  GFX::Scene* scene = nullptr;
  /** the scene node that should be opened */
  GFX::GameObjectHandle forceNodeOpen = GFX::NULL_GO;
  /** the scene node that is inspected */
  GFX::GameObjectHandle inspected = GFX::NULL_GO;
  /** inspector widget to show gameobject detail */
  InspectorWidget* inspectorWidget = nullptr;
  /** viewport widget to show gameobject gizmos */
  ViewportWidget* viewportWidget = nullptr;
  /** game object inspector */
  GameObjectInspector gameobjectInspector = {};
};
}  // namespace SIByL::Editor

namespace SIByL::Editor {
SE_EXPORT struct ContentWidget : public Widget {
  /** draw gui*/
  virtual auto onDrawGui() noexcept -> void override;
  /** contents source */
  enum struct Source {
    FileSystem,
    RuntimeResources,
  } source = Source::FileSystem;
  /** resources registry */
  struct ResourceRegistry {
    std::string resourceName;
    Core::GUID resourceIcon;
    std::vector<std::string> possibleExtensions;
    using GUIDFinderFn = std::function<Core::GUID(Core::ORID)>;
    GUIDFinderFn guidFinder = nullptr;
    using ResourceLoadFn = std::function<Core::GUID(char const*)>;
    ResourceLoadFn resourceLoader = nullptr;
    inline auto matchExtensions(std::string const& ext) const noexcept -> bool {
      bool match = false;
      for (auto const& e : possibleExtensions)
        if (e == ext) match = true;
      return match;
    }
  };
  std::vector<ResourceRegistry> resourceRegistries;
  /** register a resource type */
  template <Core::StructResource T>
  auto registerResource(
      Core::GUID icon, std::initializer_list<std::string> extensions,
      ResourceRegistry::GUIDFinderFn guidFinder = nullptr,
      ResourceRegistry::ResourceLoadFn resourceLoader = nullptr) noexcept
      -> void;
  /** icon resources */
  struct IconResource {
    Core::GUID back;
    Core::GUID folder;
    Core::GUID file;
    Core::GUID mesh;
    Core::GUID scene;
    Core::GUID material;
    Core::GUID shader;
    Core::GUID image;
    Core::GUID video;
  } icons;
  /* register icon resources*/
  auto reigsterIconResources() noexcept -> void;
  /** current directory */
  std::filesystem::path currentDirectory = "./content";
  /** current resource */
  struct {
    char const* type = nullptr;
    std::vector<Core::GUID> const* GUIDs = nullptr;
  } runtimeResourceInfo;
  /** inspector widget to show gameobject detail */
  InspectorWidget* inspectorWidget = nullptr;

  struct modalState {
    bool showModal = false;
    std::string path;
    ResourceRegistry const* entry_ptr;
  } modal_state;
};

template <Core::StructResource T>
auto ContentWidget::registerResource(
    Core::GUID icon, std::initializer_list<std::string> extensions,
    ResourceRegistry::GUIDFinderFn guidFinder,
    ResourceRegistry::ResourceLoadFn resourceLoader) noexcept -> void {
  resourceRegistries.emplace_back(ResourceRegistry{
      typeid(T).name(), icon, std::vector<std::string>(extensions), guidFinder,
      resourceLoader});
}
}  // namespace SIByL::Editor

namespace SIByL::Editor {
SE_EXPORT struct CameraState {
  float yaw = 0;
  float pitch = 0;
  float roll = 0;

  float x = 0;
  float y = 0;
  float z = 0;

  auto setFromTransform(GFX::TransformComponent const& transform) noexcept
      -> void;
  auto lerpTowards(CameraState const& target, float positionLerpPct,
                   float rotationLerpPct) noexcept -> void;
  auto updateTransform(GFX::TransformComponent& transform) noexcept -> void;
};

SE_EXPORT struct SimpleCameraController {
  SimpleCameraController() : input(nullptr), timer(nullptr) {}
  SimpleCameraController(Platform::Input* input, Core::Timer* timer,
                         ViewportWidget* viewport)
      : input(input), timer(timer), viewport(viewport) {}

  auto inline init(Platform::Input* input, Core::Timer* timer,
                   ViewportWidget* viewport) {
    this->input = input;
    this->timer = timer;
    this->viewport = viewport;
  }

  float mouseSensitivityMultiplier = 0.01f;
  CameraState targetCameraState;
  CameraState interpolatingCameraState;
  float boost = 3.5f;
  float positionLerpTime = 0.2f;
  float mouseSensitivity = 60.0f;
  Math::AnimationCurve mouseSensitivityCurve = {{0, 0.5, 0, 5}, {1, 2.5, 0, 0}};
  float rotationLerpTime = 0.01f;
  bool invertY = true;
  bool forceReset = false;

  ViewportWidget* viewport = nullptr;

  auto onEnable(GFX::TransformComponent const& transform) noexcept -> void;
  auto getInputTranslationDirection() noexcept -> Math::vec3;
  auto bindTransform(GFX::TransformComponent* transform) noexcept -> void;
  auto onUpdate() noexcept -> void;

 private:
  GFX::TransformComponent* transform = nullptr;
  Platform::Input* input;
  Core::Timer* timer;
};

SE_EXPORT struct SimpleCameraController2D {
  SimpleCameraController2D() : input(nullptr), timer(nullptr) {}
  SimpleCameraController2D(Platform::Input* input, Core::Timer* timer,
                         ViewportWidget* viewport)
      : input(input), timer(timer), viewport(viewport) {}

  auto inline init(Platform::Input* input, Core::Timer* timer,
                   ViewportWidget* viewport) {
    this->input = input;
    this->timer = timer;
    this->viewport = viewport;
  }

  float mouseSensitivityMultiplier = 0.01f;
  CameraState targetCameraState;
  CameraState interpolatingCameraState;
  float boost = 3.5f;
  float positionLerpTime = 0.2f;
  float mouseSensitivity = 60.0f;
  Math::AnimationCurve mouseSensitivityCurve = {{0, 0.5, 0, 5}, {1, 2.5, 0, 0}};
  float rotationLerpTime = 0.01f;
  bool invertY = true;
  bool forceReset = false;
  float scaling = 1.f;

  ViewportWidget* viewport = nullptr;

  auto onEnable(GFX::TransformComponent const& transform) noexcept -> void;
  auto getInputTranslationDirection() noexcept -> Math::vec3;
  auto bindTransform(GFX::TransformComponent* transform) noexcept -> void;
  auto onUpdate() noexcept -> void;

 private:
  GFX::TransformComponent* transform = nullptr;
  Platform::Input* input;
  Core::Timer* timer;
};
}  // namespace SIByL::Editor

namespace SIByL::Editor {
SE_EXPORT struct ComponentElucidator : public Fragment {
  /** override draw gui */
  virtual auto onDrawGui(uint32_t flags, void* data) noexcept -> void override;
  /** elucidate component */
  virtual auto elucidateComponent(
      GameObjectInspector::GameObjectData* data) noexcept -> void = 0;
};

SE_EXPORT template <class T, class UIFunction>
auto drawComponent(GFX::GameObject* gameObject, std::string const& name,
                   UIFunction uiFunction, bool couldRemove = true) noexcept
    -> void {
  const ImGuiTreeNodeFlags treeNodeFlags =
      ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_Framed |
      ImGuiTreeNodeFlags_FramePadding | ImGuiTreeNodeFlags_SpanAvailWidth |
      ImGuiTreeNodeFlags_AllowItemOverlap;
  T* component = gameObject->getEntity().getComponent<T>();
  if (component) {
    ImVec2 contentRegionAvailable = ImGui::GetContentRegionAvail();
    float lineHeight = GImGui->Font->FontSize + GImGui->Style.FramePadding.y * 2.0f;
    ImGui::Separator();
    bool open = ImGui::TreeNodeEx((void*)typeid(T).hash_code(), treeNodeFlags,
                                  name.c_str());
    bool removeComponent = false;
    if (couldRemove) {
      ImGui::SameLine(contentRegionAvailable.x - lineHeight * 0.5);
      if (ImGui::Button("+", ImVec2{lineHeight, lineHeight})) {
        ImGui::OpenPopup("ComponentSettings");
      }
      if (ImGui::BeginPopup("ComponentSettings")) {
        if (ImGui::MenuItem("Remove Component")) removeComponent = true;
        ImGui::EndPopup();
      }
    }
    if (open) {
      T* component = gameObject->getEntity().getComponent<T>();
      uiFunction(component);
      ImGui::Dummy(ImVec2(0.0f, 20.0f));
      ImGui::TreePop();
    }
    if (couldRemove && removeComponent)
      gameObject->getEntity().removeComponent<T>();
  }
}

SE_EXPORT struct TagComponentFragment : public ComponentElucidator {
  virtual auto elucidateComponent(
      GameObjectInspector::GameObjectData* data) noexcept -> void;
};

SE_EXPORT struct TransformComponentFragment : public ComponentElucidator {
  virtual auto elucidateComponent(
      GameObjectInspector::GameObjectData* data) noexcept -> void;
};

SE_EXPORT struct MeshReferenceComponentFragment : public ComponentElucidator {
  virtual auto elucidateComponent(
      GameObjectInspector::GameObjectData* data) noexcept -> void;
};

SE_EXPORT struct MeshRendererComponentFragment : public ComponentElucidator {
  virtual auto elucidateComponent(
      GameObjectInspector::GameObjectData* data) noexcept -> void;
};

SE_EXPORT struct LightComponentFragment : public ComponentElucidator {
  virtual auto elucidateComponent(
      GameObjectInspector::GameObjectData* data) noexcept -> void;
};

SE_EXPORT struct CameraComponentFragment : public ComponentElucidator {
  virtual auto elucidateComponent(
      GameObjectInspector::GameObjectData* data) noexcept -> void;
};

SE_EXPORT struct NativeScriptComponentFragment : public ComponentElucidator {
  virtual auto elucidateComponent(
      GameObjectInspector::GameObjectData* data) noexcept -> void;
};
}