#pragma once
#include <memory>
#include <unordered_map>
#include <Resource/SE.Core.Resource.hpp>
#include <SE.RHI.hpp>
#include <SE.Editor.Core-ImGuiBackend.hpp>

namespace SIByL::Editor {
SE_EXPORT struct ImGuiLayer : public Core::Layer {
 public:
  /** initialzier */
  ImGuiLayer(RHI::RHILayer* rhiLayer);
  /** virtual destructor*/
  virtual ~ImGuiLayer();
  /* get singleton */
  static auto get() noexcept -> ImGuiLayer* { return singleton; }
  // auto onEvent(Event& e) -> void;
  auto onWindowResize(size_t x, size_t y) -> void;

  auto startNewFrame() -> void;
  auto startGuiRecording() -> void;
  auto render() -> void;

  auto getDPI() noexcept -> float { return dpi; }
  auto createImGuiTexture(RHI::Sampler* sampler, RHI::TextureView* view,
                          RHI::TextureLayout layout) noexcept
      -> std::unique_ptr<ImGuiTexture>;

  /** rhi layer */
  RHI::RHILayer* rhiLayer = nullptr;
  /** imgui backend */
  std::unique_ptr<ImGuiBackend> imguiBackend = nullptr;
  /** imgui texture pool */
  std::unordered_map<Core::GUID, std::unique_ptr<ImGuiTexture>>
      ImGuiTexturePool = {};

 private:
  float dpi;
  static ImGuiLayer* singleton;
};
}  // namespace SIByL::Editor