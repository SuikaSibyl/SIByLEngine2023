#pragma once
#include <se.editor.hpp>
#include <../source/rhi/se.rhi.vk.hpp>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>

namespace se::editor {
struct SIByL_API ImGuiTexture_VK : public ImGuiTexture {
  /** initialzie */
  ImGuiTexture_VK(rhi::Sampler* sampler, rhi::TextureView* view, rhi::TextureLayout layout);
  /** virtual destructor */
  virtual ~ImGuiTexture_VK();
  /** get imgui image handle */
  virtual auto getTextureID() noexcept -> ImTextureID override;
  /** vulkan descriptor set */
  VkDescriptorSet descriptorSet = {};
};

struct SIByL_API ImGuiBackend_VK : public ImGuiBackend {
  /** initializer */
  ImGuiBackend_VK(rhi::Device* device);
  virtual ~ImGuiBackend_VK();
  virtual auto setupPlatformBackend() noexcept -> void override;
  virtual auto uploadFonts() noexcept -> void override;
  virtual auto getWindowDPI() noexcept -> float override;
  virtual auto onWindowResize(size_t x, size_t y) -> void override;
  virtual auto startNewFrame() -> void override;
  virtual auto render(ImDrawData* draw_data,
    rhi::Semaphore* waitSemaphore = nullptr) -> void override;
  virtual auto present() -> void override;
  virtual auto createImGuiTexture(rhi::Sampler* sampler, rhi::TextureView* view,
    rhi::TextureLayout layout) noexcept -> std::unique_ptr<ImGuiTexture> override;
  ImGui_ImplVulkanH_Window mainWindowData;
  VkPipelineCache pipelineCache = VK_NULL_HANDLE;
  VkDescriptorPool descriptorPool;
  se::window* bindedWindow = nullptr;
  rhi::Device_VK* device;
  bool swapChainRebuild = false;
};
}