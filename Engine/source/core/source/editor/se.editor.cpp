#define DLIB_EXPORT
#include <se.editor.hpp>
#include "se.editor.vk.hpp"
#undef DLIB_EXPORT
#define SIByL_API __declspec(dllimport)
#include <../source/rhi/se.rhi.vk.hpp>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>

namespace se::editor {
ImGuiTexture_VK::ImGuiTexture_VK(rhi::Sampler* sampler, rhi::TextureView* view, rhi::TextureLayout layout) {
  descriptorSet = ImGui_ImplVulkan_AddTexture(
    static_cast<rhi::Sampler_VK*>(sampler)->textureSampler,
    static_cast<rhi::TextureView_VK*>(view)->imageView,
    getVkImageLayout(layout));
}

ImGuiTexture_VK::~ImGuiTexture_VK() {}

auto ImGuiTexture_VK::getTextureID() noexcept -> ImTextureID {
  return (ImTextureID)descriptorSet;
}

inline int g_MinImageCount = 2;

ImGuiBackend_VK::ImGuiBackend_VK(rhi::Device* device)
  : device((rhi::Device_VK*)device)
  , bindedWindow(device->fromAdapter()->fromContext()->getBindedWindow()) {
  rhi::Adapter_VK* adapter = this->device->getAdapterVk();
  rhi::Context_VK* context = this->device->getAdapterVk()->getContext();
  // fill main window data
  mainWindowData.Surface = context->getVkSurfaceKHR();
  // select Surface Format
  VkFormat const requestSurfaceImageFormat[] = {
      VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM,
      VK_FORMAT_B8G8R8_UNORM, VK_FORMAT_R8G8B8_UNORM};
  VkColorSpaceKHR const requestSurfaceColorSpace =
      VK_COLORSPACE_SRGB_NONLINEAR_KHR;
  mainWindowData.SurfaceFormat = ImGui_ImplVulkanH_SelectSurfaceFormat(
    static_cast<rhi::Adapter_VK*>(this->device->getAdapterVk())
    ->getVkPhysicalDevice(), mainWindowData.Surface, requestSurfaceImageFormat,
    (size_t)IM_ARRAYSIZE(requestSurfaceImageFormat), requestSurfaceColorSpace);
  // select Surface present mode
  VkPresentModeKHR present_modes[] = {VK_PRESENT_MODE_FIFO_KHR};
  mainWindowData.PresentMode = ImGui_ImplVulkanH_SelectPresentMode(
    static_cast<rhi::Adapter_VK*>(this->device->getAdapterVk())
    ->getVkPhysicalDevice(), mainWindowData.Surface, &present_modes[0], IM_ARRAYSIZE(present_modes));
  // Create SwapChain, RenderPass, Framebuffer, etc.
  rhi::QueueFamilyIndices_VK indices =
    static_cast<rhi::Adapter_VK*>(this->device->getAdapterVk())
    ->getQueueFamilyIndices();
  IM_ASSERT(g_MinImageCount >= 2);
  int width, height;
  context->getBindedWindow()->getFramebufferSize(&width, &height);
  ImGui_ImplVulkanH_CreateOrResizeWindow(
    context->getVkInstance(),
    adapter->getVkPhysicalDevice(),
    this->device->getVkDevice(),
    &mainWindowData, indices.graphicsFamily.value(), nullptr, width, height,
    g_MinImageCount);
  // Create Descriptor Pool
  VkDescriptorPoolSize pool_sizes[] = {
      {VK_DESCRIPTOR_TYPE_SAMPLER, 1000},
      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000},
      {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000},
      {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000},
      {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000},
      {VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000},
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000},
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000},
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000},
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000},
      {VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000}};
  VkDescriptorPoolCreateInfo pool_info = {};
  pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
  pool_info.maxSets = 1000 * IM_ARRAYSIZE(pool_sizes);
  pool_info.poolSizeCount = (uint32_t)IM_ARRAYSIZE(pool_sizes);
  pool_info.pPoolSizes = pool_sizes;
  vkCreateDescriptorPool(
    this->device->getVkDevice(),
    &pool_info, nullptr, &descriptorPool);
  // bind resize event
  bindedWindow->connectResizeEvent(std::bind(&ImGuiBackend_VK::onWindowResize,
    this, std::placeholders::_1, std::placeholders::_2));
}

ImGuiBackend_VK::~ImGuiBackend_VK() {
  vkDeviceWaitIdle(device->getVkDevice());
  ImGui_ImplVulkanH_DestroyWindow(
    device->getAdapterVk()->getContext()->getVkInstance(),
    device->getVkDevice(),
    &mainWindowData, nullptr);
  vkDestroyDescriptorPool(
    device->getVkDevice(),
    descriptorPool, nullptr);
  ImGui_ImplGlfw_Shutdown();
  ImGui_ImplVulkan_Shutdown();

  rhi::Context_VK* context = device->getAdapterVk()->getContext();
  context->getVkSurfaceKHR() = {};
}

auto ImGuiBackend_VK::getWindowDPI() noexcept -> float {
  return bindedWindow->getHighDPI();
}

auto ImGuiBackend_VK::setupPlatformBackend() noexcept -> void {
  auto adapter = device->getAdapterVk();
  auto context = adapter->getContext();
  rhi::QueueFamilyIndices_VK indices = adapter->getQueueFamilyIndices();
  ImGui_ImplGlfw_InitForVulkan((GLFWwindow*)bindedWindow->getHandle(), true);
  ImGui_ImplVulkan_InitInfo init_info = {};
  init_info.Instance = adapter->getContext()->getVkInstance();
  init_info.PhysicalDevice = adapter->getVkPhysicalDevice();
  init_info.Device = device->getVkDevice();
  init_info.QueueFamily = indices.graphicsFamily.value();
  init_info.Queue = static_cast<rhi::Queue_VK*>(device->getGraphicsQueue())->queue;
  init_info.PipelineCache = pipelineCache;
  init_info.DescriptorPool = descriptorPool;
  init_info.Subpass = 0;
  init_info.MinImageCount = g_MinImageCount;
  init_info.ImageCount = mainWindowData.ImageCount;
  init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
  init_info.Allocator = nullptr;
  init_info.CheckVkResultFn = nullptr;
  ImGui_ImplVulkan_Init(&init_info, mainWindowData.RenderPass);
}

VkResult err;
void check_vk_result(VkResult err) {
  if (err == 0) return;
  root::print::error(std::format("ImGui Vulkan Error: VkResult = {0}", (unsigned int)err));
  if (err < 0) return;
}

// Upload Fonts
auto ImGuiBackend_VK::uploadFonts() noexcept -> void {
  // Use any command queue
  VkCommandPool command_pool =
    mainWindowData.Frames[mainWindowData.FrameIndex].CommandPool;
  VkCommandBuffer command_buffer =
    mainWindowData.Frames[mainWindowData.FrameIndex].CommandBuffer;
  VkResult err = vkResetCommandPool(device->getVkDevice(), command_pool, 0);
  check_vk_result(err);
  VkCommandBufferBeginInfo begin_info = {};
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin_info.flags |= VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  err = vkBeginCommandBuffer(command_buffer, &begin_info);
  check_vk_result(err);
  ImGui_ImplVulkan_CreateFontsTexture(command_buffer);
  VkSubmitInfo end_info = {};
  end_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  end_info.commandBufferCount = 1;
  end_info.pCommandBuffers = &command_buffer;
  err = vkEndCommandBuffer(command_buffer);
  check_vk_result(err);
  err = vkQueueSubmit(
    static_cast<rhi::Queue_VK*>(device->getGraphicsQueue())->queue, 1, &end_info, VK_NULL_HANDLE);
  check_vk_result(err);
  err = vkDeviceWaitIdle(device->getVkDevice());
  check_vk_result(err);
  ImGui_ImplVulkan_DestroyFontUploadObjects();
}

auto ImGuiBackend_VK::onWindowResize(size_t x, size_t y) -> void {
  auto adapter = device->getAdapterVk();
  auto context = adapter->getContext();
  rhi::QueueFamilyIndices_VK indices = adapter->getQueueFamilyIndices();
  ImGui_ImplVulkan_SetMinImageCount(g_MinImageCount);
  ImGui_ImplVulkanH_CreateOrResizeWindow(
    context->getVkInstance(),
    adapter->getVkPhysicalDevice(),
    device->getVkDevice(),
    &mainWindowData, indices.graphicsFamily.value(), nullptr, x, y,
    g_MinImageCount);
  mainWindowData.FrameIndex = 0;
}

auto ImGuiBackend_VK::startNewFrame() -> void {
  // Start the Dear ImGui frame
  ImGui_ImplVulkan_NewFrame();
  ImGui_ImplGlfw_NewFrame();
}

ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

auto ImGuiBackend_VK::render(ImDrawData* draw_data,
    rhi::Semaphore* waitSemaphore) -> void {
  mainWindowData.ClearValue.color.float32[0] = clear_color.x * clear_color.w;
  mainWindowData.ClearValue.color.float32[1] = clear_color.y * clear_color.w;
  mainWindowData.ClearValue.color.float32[2] = clear_color.z * clear_color.w;
  mainWindowData.ClearValue.color.float32[3] = clear_color.w;

  VkResult err;
  ImGui_ImplVulkanH_Window* wd = &mainWindowData;

  VkSemaphore image_acquired_semaphore =
      wd->FrameSemaphores[wd->SemaphoreIndex].ImageAcquiredSemaphore;
  VkSemaphore render_complete_semaphore =
      wd->FrameSemaphores[wd->SemaphoreIndex].RenderCompleteSemaphore;
  err = vkAcquireNextImageKHR(device->getVkDevice(),
      wd->Swapchain, UINT64_MAX, image_acquired_semaphore, VK_NULL_HANDLE,
      &wd->FrameIndex);
  check_vk_result(err);

  ImGui_ImplVulkanH_Frame* fd = &wd->Frames[wd->FrameIndex];
  {
    err = vkWaitForFences(device->getVkDevice(), 1,
    &fd->Fence, VK_TRUE,
    UINT64_MAX);  // wait indefinitely instead of periodically checking
    check_vk_result(err);
    err = vkResetFences(device->getVkDevice(), 1,
        &fd->Fence);
    check_vk_result(err);
  }
  {
    err = vkResetCommandPool(device->getVkDevice(),
        fd->CommandPool, 0);
    check_vk_result(err);
    VkCommandBufferBeginInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    info.flags |= VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    err = vkBeginCommandBuffer(fd->CommandBuffer, &info);
    check_vk_result(err);
  }
  {
    VkRenderPassBeginInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    info.renderPass = wd->RenderPass;
    info.framebuffer = fd->Framebuffer;
    info.renderArea.extent.width = wd->Width;
    info.renderArea.extent.height = wd->Height;
    info.clearValueCount = 1;
    info.pClearValues = &wd->ClearValue;
    vkCmdBeginRenderPass(fd->CommandBuffer, &info, VK_SUBPASS_CONTENTS_INLINE);
  }
  // Record dear imgui primitives into command buffer
  ImGui_ImplVulkan_RenderDrawData(draw_data, fd->CommandBuffer);
  // Submit command buffer
  vkCmdEndRenderPass(fd->CommandBuffer);
  {
    std::vector<VkSemaphore> waitSeams = {image_acquired_semaphore};
    std::vector<VkPipelineStageFlags> wait_stages = {
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    if (waitSemaphore) {
      waitSeams.push_back(
        static_cast<rhi::Semaphore_VK*>(waitSemaphore)->semaphore);
      wait_stages.push_back(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
    }

    VkSubmitInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    info.waitSemaphoreCount = waitSeams.size();
    info.pWaitSemaphores = waitSeams.data();
    info.pWaitDstStageMask = wait_stages.data();
    info.commandBufferCount = 1;
    info.pCommandBuffers = &fd->CommandBuffer;
    info.signalSemaphoreCount = 1;
    info.pSignalSemaphores = &render_complete_semaphore;

    err = vkEndCommandBuffer(fd->CommandBuffer);
    check_vk_result(err);
    err = vkQueueSubmit(
      static_cast<rhi::Queue_VK*>(device->getGraphicsQueue())->queue,
      1, &info, fd->Fence);
    check_vk_result(err);
  }
}

auto ImGuiBackend_VK::present() -> void {
  VkSemaphore render_complete_semaphore =
      mainWindowData.FrameSemaphores[mainWindowData.SemaphoreIndex]
          .RenderCompleteSemaphore;
  VkPresentInfoKHR info = {};
  info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
  info.waitSemaphoreCount = 1;
  info.pWaitSemaphores = &render_complete_semaphore;
  info.swapchainCount = 1;
  info.pSwapchains = &mainWindowData.Swapchain;
  info.pImageIndices = &mainWindowData.FrameIndex;
  VkResult err = vkQueuePresentKHR(
    static_cast<rhi::Queue_VK*>(device->getGraphicsQueue())->queue, &info);
  check_vk_result(err);
  mainWindowData.SemaphoreIndex =
      (mainWindowData.SemaphoreIndex + 1) %
      mainWindowData.ImageCount;  // Now we can use the next set of semaphores
}

auto ImGuiBackend_VK::createImGuiTexture(
  rhi::Sampler* sampler,
  rhi::TextureView* view,
  rhi::TextureLayout layout) noexcept
  -> std::unique_ptr<ImGuiTexture> {
  return std::make_unique<ImGuiTexture_VK>(sampler, view, layout);
}

auto ImGuiContext::getImGuiTexture(gfx::TextureHandle texture) noexcept -> ImGuiTexture* {
  auto& pool = ImGuiTexturePool;
  auto iter = pool.find(texture->texture.get());
  if (iter == pool.end()) {
    gfx::SamplerHandle sampler = gfx::GFXContext::create_sampler_desc(
      rhi::AddressMode::CLAMP_TO_EDGE, rhi::FilterMode::NEAREST,
      rhi::MipmapFilterMode::NEAREST);
    pool.insert({ texture->texture.get(), 
      imguiBackend->createImGuiTexture(
        sampler.get(), texture->getSRV(0, 1, 0, 1),
        rhi::TextureLayout::SHADER_READ_ONLY_OPTIMAL)});
    return pool[texture->texture.get()].get();
  } else {
    return iter->second.get();
  }
}

auto ImGuiContext::initialize(rhi::Device* device) -> void {
  imguiBackend = std::make_unique<ImGuiBackend_VK>(device);
  // Setup Dear ImGui context
  IMGUI_CHECKVERSION();
  imContext = ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO();
  (void)io;
  io.BackendFlags |= ImGuiBackendFlags_HasMouseCursors;
  io.BackendFlags |= ImGuiBackendFlags_HasSetMousePos;
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard; // Enable Keyboard Controls
  io.ConfigFlags |= ImGuiConfigFlags_DockingEnable; // Enable Docking
  io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable; // Enable Multi-Viewport
  // Setup Dear ImGui style
  ImGui::StyleColorsDark();
  dpi = imguiBackend->getWindowDPI();

  std::string engine_path = se::RuntimeConfig::get()->string_property("engine_path");
  io.Fonts->AddFontFromFileTTF(
    (engine_path + "binary/resources/fonts/opensans/OpenSans-Bold.ttf").c_str(), dpi * 15.0f);
  io.FontDefault = io.Fonts->AddFontFromFileTTF(
    (engine_path + "binary/resources/fonts/opensans/OpenSans-Bold.ttf").c_str(), dpi * 15.0f);

  // set dark theme
  {
      auto& colors = ImGui::GetStyle().Colors;
      // Back Grounds
      colors[ImGuiCol_WindowBg] = ImVec4{ 0.121568f, 0.121568f, 0.121568f, 1.0f };
      colors[ImGuiCol_DockingEmptyBg] = ImVec4{ 0.117647f, 0.117647f, 0.117647f, 1.0f };
      // Headers
      colors[ImGuiCol_Header] = ImVec4{ 0.121568f, 0.121568f, 0.121568f, 1.0f };
      colors[ImGuiCol_HeaderHovered] = ImVec4{ 0.2392f, 0.2392f, 0.2392f, 1.0f };
      colors[ImGuiCol_HeaderActive] = ImVec4{ 0.2392f, 0.2392f, 0.2392f, 1.0f };
      // Buttons
      colors[ImGuiCol_Button] = ImVec4{ 0.2f, 0.205f, 0.21f, 1.0f };
      colors[ImGuiCol_ButtonHovered] = ImVec4{ 0.3f, 0.305f, 0.31f, 1.0f };
      colors[ImGuiCol_ButtonActive] = ImVec4{ 0.15f, 0.1505f, 0.151f, 1.0f };
      // Frame BG
      colors[ImGuiCol_FrameBg] = ImVec4{ 0.2f, 0.205f, 0.21f, 1.0f };
      colors[ImGuiCol_FrameBgHovered] = ImVec4{ 0.3f, 0.305f, 0.31f, 1.0f };
      colors[ImGuiCol_FrameBgActive] = ImVec4{ 0.15f, 0.1505f, 0.151f, 1.0f };
      // Tabs
      colors[ImGuiCol_Tab] = ImVec4{ 0.15f, 0.1505f, 0.151f, 1.0f };
      colors[ImGuiCol_TabHovered] = ImVec4{ 0.38f, 0.3805f, 0.381f, 1.0f };
      colors[ImGuiCol_TabActive] = ImVec4{ 0.23922f, 0.23922f, 0.23922f, 1.0f };
      colors[ImGuiCol_TabUnfocused] = ImVec4{ 0.15f, 0.1505f, 0.151f, 1.0f };
      colors[ImGuiCol_TabUnfocusedActive] = ImVec4{ 0.2f, 0.205f, 0.21f, 1.0f };
      // Title
      colors[ImGuiCol_TitleBg] = ImVec4{ 0.15f, 0.1505f, 0.151f, 1.0f };
      colors[ImGuiCol_TitleBgActive] = ImVec4{ 0.121568f, 0.121568f, 0.121568f, 1.0f };
      colors[ImGuiCol_TitleBgCollapsed] = ImVec4{ 0.15f, 0.1505f, 0.151f, 1.0f };
  }
  // When viewports are enabled we tweak WindowRounding/WindowBg so platform
  // windows can look identical to regular ones.
  ImGuiStyle& style = ImGui::GetStyle();
  if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
    style.WindowRounding = 0.0f;
    style.Colors[ImGuiCol_WindowBg].w = 1.0f;
  }

  imguiBackend->setupPlatformBackend();
  imguiBackend->uploadFonts();
}

auto ImGuiContext::finalize() -> void {
  ImGuiTexturePool.clear();
  imguiBackend = nullptr;
  ImGui::DestroyContext();
}

auto ImGuiContext::getRawCtx() noexcept -> RawImGuiCtx* {
  return imContext;
}

auto ImGuiContext::startNewFrame() -> void {
  imguiBackend->startNewFrame();
}

auto ImGuiContext::startGuiRecording() -> void {
  ImGui::NewFrame();
  // Using Docking space
  {
    static bool dockspaceOpen = true;
    static bool opt_fullscreen = true;
    static bool opt_padding = false;
    static ImGuiDockNodeFlags dockspace_flags = ImGuiDockNodeFlags_None;

    // We are using the ImGuiWindowFlags_NoDocking flag to make the parent
    // window not dockable into, because it would be confusing to have two
    // docking targets within each others.
    ImGuiWindowFlags window_flags =
        ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoDocking;
    if (opt_fullscreen) {
      const ImGuiViewport* viewport = ImGui::GetMainViewport();
      ImGui::SetNextWindowPos(viewport->WorkPos);
      ImGui::SetNextWindowSize(viewport->WorkSize);
      ImGui::SetNextWindowViewport(viewport->ID);
      ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
      ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
      window_flags |= ImGuiWindowFlags_NoTitleBar |
                      ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize |
                      ImGuiWindowFlags_NoMove;
      window_flags |=
          ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;
    } else {
      dockspace_flags &= ~ImGuiDockNodeFlags_PassthruCentralNode;
    }
    // When using ImGuiDockNodeFlags_PassthruCentralNode, DockSpace() will
    // render our background and handle the pass-thru hole, so we ask Begin() to
    // not render a background.
    if (dockspace_flags & ImGuiDockNodeFlags_PassthruCentralNode)
      window_flags |= ImGuiWindowFlags_NoBackground;
    // Important: note that we proceed even if Begin() returns false (aka window
    // is collapsed). This is because we want to keep our DockSpace() active. If
    // a DockSpace() is inactive, all active windows docked into it will lose
    // their parent and become undocked. We cannot preserve the docking
    // relationship between an active window and an inactive docking, otherwise
    // any change of dockspace/settings would lead to windows being stuck in
    // limbo and never being visible.
    if (!opt_padding)
      ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::Begin("DockSpace Demo", &dockspaceOpen, window_flags);
    if (!opt_padding) ImGui::PopStyleVar();
    if (opt_fullscreen) ImGui::PopStyleVar(2);
    // Submit the DockSpace
    ImGuiIO& io = ImGui::GetIO();
    ImGuiStyle& style = ImGui::GetStyle();
    // style.WindowMinSize.x = 350.0f;
    if (io.ConfigFlags & ImGuiConfigFlags_DockingEnable) {
      ImGuiID dockspace_id = ImGui::GetID("MyDockSpace");
      ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), dockspace_flags);
    }
  }
}

auto ImGuiContext::render(rhi::Semaphore* waitSemaphore) -> void {
  // End docking space
  ImGui::End();
  // Do render ImGui stuffs
  ImGui::Render();
  ImDrawData* main_draw_data = ImGui::GetDrawData();
  const bool main_is_minimized = (main_draw_data->DisplaySize.x <= 0.0f ||
                                  main_draw_data->DisplaySize.y <= 0.0f);
  if (!main_is_minimized) imguiBackend->render(main_draw_data, waitSemaphore);
  ImGuiIO& io = ImGui::GetIO();
  (void)io;
  // Update and Render additional Platform Windows
  if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
    ImGui::UpdatePlatformWindows();
    ImGui::RenderPlatformWindowsDefault();
  }
  // Present Main Platform Window
  if (!main_is_minimized) imguiBackend->present();
}

float ImGuiContext::dpi;
::ImGuiContext* ImGuiContext::imContext = nullptr;
std::unique_ptr<ImGuiBackend> ImGuiContext::imguiBackend = nullptr;
std::unordered_map<rhi::Texture*, std::unique_ptr<ImGuiTexture>> ImGuiContext::ImGuiTexturePool = {};

auto Widget::commonOnDrawGui() noexcept -> void {
  // get the screen pos
  info.windowPos = ImGui::GetWindowPos();
  // see whether it is hovered
  if (ImGui::IsWindowHovered()) info.isHovered = true;
  else info.isHovered = false;
  if (ImGui::IsWindowFocused()) info.isFocused = true;
  else info.isFocused = false;
}
}