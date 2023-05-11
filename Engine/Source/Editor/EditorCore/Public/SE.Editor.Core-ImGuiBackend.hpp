#pragma once
#include <imgui.h>
#include <memory>
#include <SE.RHI.hpp>
#include <System/SE.Core.System.hpp>

namespace SIByL::Editor {
SE_EXPORT struct ImGuiTexture {
  /** virtual destructor */
  virtual ~ImGuiTexture() = default;
  /* get imgui image handle */
  virtual auto getTextureID() noexcept -> ImTextureID = 0;
};

SE_EXPORT struct ImGuiBackend {
  /** virtual destructor */
  virtual ~ImGuiBackend() = default;

  virtual auto setupPlatformBackend() noexcept -> void = 0;
  virtual auto uploadFonts() noexcept -> void = 0;
  virtual auto getWindowDPI() noexcept -> float = 0;
  virtual auto onWindowResize(size_t, size_t) -> void = 0;

  virtual auto startNewFrame() -> void = 0;
  virtual auto render(ImDrawData* draw_data) -> void = 0;
  virtual auto present() -> void = 0;

  virtual auto createImGuiTexture(RHI::Sampler* sampler, RHI::TextureView* view,
                                  RHI::TextureLayout layout) noexcept
      -> std::unique_ptr<ImGuiTexture> = 0;
};
}  // namespace SIByL::Editor