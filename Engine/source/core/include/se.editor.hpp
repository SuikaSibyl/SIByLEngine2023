#pragma once
#include <se.core.hpp>
#include <se.rhi.hpp>
#include <se.gfx.hpp>
#include <imgui.h>
#include <imgui_internal.h>

namespace se::editor {
struct SIByL_API ImGuiTexture {
  /** virtual destructor */
  virtual ~ImGuiTexture() = default;
  /** get imgui image handle */
  virtual auto getTextureID() noexcept -> ImTextureID = 0;
};

struct SIByL_API ImGuiBackend {
  /** virtual destructor */
  virtual ~ImGuiBackend() = default;
  /** setup the backend for the platform */
  virtual auto setupPlatformBackend() noexcept -> void = 0;
  /** upload the fonts for the imgui */
  virtual auto uploadFonts() noexcept -> void = 0;
  /** get the window DPI */
  virtual auto getWindowDPI() noexcept -> float = 0;
  /** response to the window resize */
  virtual auto onWindowResize(size_t, size_t) -> void = 0;
  /** start a new frame */
  virtual auto startNewFrame() -> void = 0;
  /** render the editor frame */
  virtual auto render(ImDrawData* draw_data,
    rhi::Semaphore* waitSemaphore = nullptr) -> void = 0;
  /** present the current frame */
  virtual auto present() -> void = 0;
  /** create the ImGui Texture */
  virtual auto createImGuiTexture(rhi::Sampler* sampler, rhi::TextureView* view,
    rhi::TextureLayout layout) noexcept -> std::unique_ptr<ImGuiTexture> = 0;
};

using RawImGuiCtx = ::ImGuiContext;

struct SIByL_API ImGuiContext {
  /** initialzier */
  static auto initialize(rhi::Device* device) -> void;
  static auto finalize() -> void;
  static auto getRawCtx() noexcept -> RawImGuiCtx*;
  // auto onEvent(Event& e) -> void;
  static auto onWindowResize(size_t x, size_t y) -> void;

  static auto startNewFrame() -> void;
  static auto startGuiRecording() -> void;
  static auto render(rhi::Semaphore* waitSemaphore = nullptr) -> void;

  static auto getDPI() noexcept -> float { return dpi; }
  static auto createImGuiTexture(rhi::Sampler* sampler, rhi::TextureView* view,
    rhi::TextureLayout layout) noexcept -> std::unique_ptr<ImGuiTexture>;

  static auto getImGuiTexture(gfx::TextureHandle texture) noexcept -> ImGuiTexture*;

  /** imgui backend */
  static std::unique_ptr<ImGuiBackend> imguiBackend;
  /** imgui texture pool */
  static std::unordered_map<rhi::Texture*, std::unique_ptr<ImGuiTexture>> ImGuiTexturePool;
  static float dpi;
  static ::ImGuiContext* imContext;
};

struct SIByL_API Widget {
  /** virtual destructor */
  virtual ~Widget() = default;
  /** virtual draw gui*/
  virtual auto onDrawGui() noexcept -> void = 0;
  /** fetch common infomation */
  auto commonOnDrawGui() noexcept -> void;
  /** widget info */
  struct WidgetInfo {
      ImVec2 windowPos;
      ImVec2 mousePos;
      bool isHovered;
      bool isFocused;
  } info;
};

struct SIByL_API Fragment {
  /** virtual destructor */
  virtual ~Fragment() = default;
  /** virtual draw gui*/
  virtual auto onDrawGui(uint32_t flags, void* data) noexcept -> void = 0;
};
}