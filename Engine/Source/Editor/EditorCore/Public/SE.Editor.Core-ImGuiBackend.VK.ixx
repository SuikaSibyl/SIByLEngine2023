module;
#include <imgui.h>
#include <memory>
#include <cstdint>
#include <format>
#include <functional>
#include <vulkan/vulkan.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>
export module SE.Editor.Core:ImGuiBackend.VK;
import SE.Core.System;
import SE.Core.Log;
import SE.Platform.Window;
import SE.RHI;
import :ImGuiBackend;

namespace SIByL::Editor
{
	export struct ImGuiTexture_VK :public ImGuiTexture {
		/** initialzie */
		ImGuiTexture_VK(RHI::Sampler* sampler, RHI::TextureView* view, RHI::TextureLayout layout);
		/** virtual destructor */
		virtual ~ImGuiTexture_VK();
		/** get imgui image handle */
		virtual auto getTextureID() noexcept -> ImTextureID override;
		/** vulkan descriptor set */
		VkDescriptorSet descriptorSet = {};
	};

	ImGuiTexture_VK::ImGuiTexture_VK(RHI::Sampler* sampler, RHI::TextureView* view, RHI::TextureLayout layout) {
		descriptorSet = ImGui_ImplVulkan_AddTexture(
			static_cast<RHI::Sampler_VK*>(sampler)->textureSampler,
			static_cast<RHI::TextureView_VK*>(view)->imageView,
			getVkImageLayout(layout));
	}
	
	ImGuiTexture_VK::~ImGuiTexture_VK() {}

	auto ImGuiTexture_VK::getTextureID() noexcept -> ImTextureID {
		return (ImTextureID)descriptorSet;
	}

	export struct ImGuiBackend_VK :public ImGuiBackend {
		/** initializer */
		ImGuiBackend_VK(RHI::RHILayer* rhiLayer);
		virtual ~ImGuiBackend_VK();

		virtual auto setupPlatformBackend() noexcept -> void override;
		virtual auto uploadFonts() noexcept -> void override;
		virtual auto getWindowDPI() noexcept -> float override;
		virtual auto onWindowResize(size_t x, size_t y) -> void override;

		virtual auto startNewFrame() -> void override;
		virtual auto render(ImDrawData* draw_data) -> void override;
		virtual auto present() -> void override;

		virtual auto createImGuiTexture(RHI::Sampler* sampler, RHI::TextureView* view,
			RHI::TextureLayout layout) noexcept -> std::unique_ptr<ImGuiTexture> override;

	private:
		ImGui_ImplVulkanH_Window mainWindowData;
		VkPipelineCache pipelineCache = VK_NULL_HANDLE;
		VkDescriptorPool descriptorPool;
		RHI::RHILayer* rhiLayer = nullptr;
		Platform::Window* bindedWindow = nullptr;
	};

	inline int g_MinImageCount = 2;

	ImGuiBackend_VK::ImGuiBackend_VK(RHI::RHILayer* rhiLayer)
		: rhiLayer(rhiLayer)
		, bindedWindow(rhiLayer->getContext()->getBindedWindow())
	{
		// fill main window data
		mainWindowData.Surface = static_cast<RHI::Context_VK*>(rhiLayer->getContext())->getVkSurfaceKHR();
		// select Surface Format
		VkFormat const requestSurfaceImageFormat[] = { VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_B8G8R8_UNORM, VK_FORMAT_R8G8B8_UNORM };
		VkColorSpaceKHR const requestSurfaceColorSpace = VK_COLORSPACE_SRGB_NONLINEAR_KHR;
		mainWindowData.SurfaceFormat = ImGui_ImplVulkanH_SelectSurfaceFormat(
			static_cast<RHI::Adapter_VK*>(rhiLayer->getAdapter())->getVkPhysicalDevice(),
			mainWindowData.Surface,
			requestSurfaceImageFormat,
			(size_t)IM_ARRAYSIZE(requestSurfaceImageFormat),
			requestSurfaceColorSpace);
		// select Surface present mode
		VkPresentModeKHR present_modes[] = { VK_PRESENT_MODE_FIFO_KHR };
		mainWindowData.PresentMode = ImGui_ImplVulkanH_SelectPresentMode(
			static_cast<RHI::Adapter_VK*>(rhiLayer->getAdapter())->getVkPhysicalDevice(),
			mainWindowData.Surface,
			&present_modes[0],
			IM_ARRAYSIZE(present_modes));
		// Create SwapChain, RenderPass, Framebuffer, etc.
		RHI::QueueFamilyIndices_VK indices = static_cast<RHI::Adapter_VK*>(rhiLayer->getAdapter())->getQueueFamilyIndices();
		IM_ASSERT(g_MinImageCount >= 2);
		int width, height;
		rhiLayer->getContext()->getBindedWindow()->getFramebufferSize(&width, &height);
		ImGui_ImplVulkanH_CreateOrResizeWindow(
			static_cast<RHI::Context_VK*>(rhiLayer->getContext())->getVkInstance(),
			static_cast<RHI::Adapter_VK*>(rhiLayer->getAdapter())->getVkPhysicalDevice(),
			static_cast<RHI::Device_VK*>(rhiLayer->getDevice())->getVkDevice(),
			&mainWindowData,
			indices.graphicsFamily.value(),
			nullptr,
			width,
			height,
			g_MinImageCount);
		// Create Descriptor Pool
		VkDescriptorPoolSize pool_sizes[] = {
			{ VK_DESCRIPTOR_TYPE_SAMPLER, 1000 },
			{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000 },
			{ VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000 },
			{ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000 },
			{ VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000 },
			{ VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000 },
			{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000 },
			{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000 },
			{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000 },
			{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000 },
			{ VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000 }
		};
		VkDescriptorPoolCreateInfo pool_info = {};
		pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
		pool_info.maxSets = 1000 * IM_ARRAYSIZE(pool_sizes);
		pool_info.poolSizeCount = (uint32_t)IM_ARRAYSIZE(pool_sizes);
		pool_info.pPoolSizes = pool_sizes;
		vkCreateDescriptorPool(static_cast<RHI::Device_VK*>(rhiLayer->getDevice())->getVkDevice(), &pool_info, nullptr, &descriptorPool);
		// bind resize event
		bindedWindow->connectResizeEvent(std::bind(&ImGuiBackend_VK::onWindowResize, this, std::placeholders::_1, std::placeholders::_2));
	}

	ImGuiBackend_VK::~ImGuiBackend_VK() {
		vkDeviceWaitIdle(static_cast<RHI::Device_VK*>(rhiLayer->getDevice())->getVkDevice());
		ImGui_ImplVulkanH_DestroyWindow(
			static_cast<RHI::Context_VK*>(rhiLayer->getContext())->getVkInstance(),
			static_cast<RHI::Device_VK*>(rhiLayer->getDevice())->getVkDevice(),
			&mainWindowData,
			nullptr);
		vkDestroyDescriptorPool(static_cast<RHI::Device_VK*>(rhiLayer->getDevice())->getVkDevice(), descriptorPool, nullptr);
		ImGui_ImplGlfw_Shutdown();
		ImGui_ImplVulkan_Shutdown();
	}

	auto ImGuiBackend_VK::getWindowDPI() noexcept -> float {
		return bindedWindow->getHighDPI();
	}

	auto ImGuiBackend_VK::setupPlatformBackend() noexcept -> void {
		RHI::QueueFamilyIndices_VK indices = static_cast<RHI::Adapter_VK*>(rhiLayer->getAdapter())->getQueueFamilyIndices();
		ImGui_ImplGlfw_InitForVulkan((GLFWwindow*)bindedWindow->getHandle(), true);
		ImGui_ImplVulkan_InitInfo init_info = {};
		init_info.Instance = static_cast<RHI::Context_VK*>(rhiLayer->getContext())->getVkInstance();
		init_info.PhysicalDevice = static_cast<RHI::Adapter_VK*>(rhiLayer->getAdapter())->getVkPhysicalDevice();
		init_info.Device = static_cast<RHI::Device_VK*>(rhiLayer->getDevice())->getVkDevice();
		init_info.QueueFamily = indices.graphicsFamily.value();
		init_info.Queue = static_cast<RHI::Queue_VK*>(rhiLayer->getDevice()->getGraphicsQueue())->queue;
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
		Core::LogManager::Error(std::format("ImGui Vulkan Error: VkResult = {0}", (unsigned int)err));
		if (err < 0) return;
	}

	// Upload Fonts
	auto ImGuiBackend_VK::uploadFonts() noexcept -> void {
		// Use any command queue
		VkCommandPool command_pool = mainWindowData.Frames[mainWindowData.FrameIndex].CommandPool;
		VkCommandBuffer command_buffer = mainWindowData.Frames[mainWindowData.FrameIndex].CommandBuffer;
		VkResult err = vkResetCommandPool(static_cast<RHI::Device_VK*>(rhiLayer->getDevice())->getVkDevice(), command_pool, 0);
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
		err = vkQueueSubmit(static_cast<RHI::Queue_VK*>(rhiLayer->getDevice()->getGraphicsQueue())->queue, 1, &end_info, VK_NULL_HANDLE);
		check_vk_result(err);
		err = vkDeviceWaitIdle(static_cast<RHI::Device_VK*>(rhiLayer->getDevice())->getVkDevice());
		check_vk_result(err);
		ImGui_ImplVulkan_DestroyFontUploadObjects();
	}
	
	auto ImGuiBackend_VK::onWindowResize(size_t x, size_t y) -> void {
		RHI::QueueFamilyIndices_VK indices = static_cast<RHI::Adapter_VK*>(rhiLayer->getAdapter())->getQueueFamilyIndices();
		ImGui_ImplVulkan_SetMinImageCount(g_MinImageCount);
		ImGui_ImplVulkanH_CreateOrResizeWindow(
			static_cast<RHI::Context_VK*>(rhiLayer->getContext())->getVkInstance(),
			static_cast<RHI::Adapter_VK*>(rhiLayer->getAdapter())->getVkPhysicalDevice(),
			static_cast<RHI::Device_VK*>(rhiLayer->getDevice())->getVkDevice(),
			&mainWindowData,
			indices.graphicsFamily.value(),
			nullptr,
			x, y,
			g_MinImageCount);
		mainWindowData.FrameIndex = 0;
	}

	auto ImGuiBackend_VK::startNewFrame() -> void {
		// Start the Dear ImGui frame
		ImGui_ImplVulkan_NewFrame();
		ImGui_ImplGlfw_NewFrame();
	}

	ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

	auto ImGuiBackend_VK::render(ImDrawData* draw_data) -> void {
		mainWindowData.ClearValue.color.float32[0] = clear_color.x * clear_color.w;
		mainWindowData.ClearValue.color.float32[1] = clear_color.y * clear_color.w;
		mainWindowData.ClearValue.color.float32[2] = clear_color.z * clear_color.w;
		mainWindowData.ClearValue.color.float32[3] = clear_color.w;

		VkResult err;
		ImGui_ImplVulkanH_Window* wd = &mainWindowData;

		VkSemaphore image_acquired_semaphore = wd->FrameSemaphores[wd->SemaphoreIndex].ImageAcquiredSemaphore;
		VkSemaphore render_complete_semaphore = wd->FrameSemaphores[wd->SemaphoreIndex].RenderCompleteSemaphore;
		err = vkAcquireNextImageKHR(static_cast<RHI::Device_VK*>(rhiLayer->getDevice())->getVkDevice(), wd->Swapchain, UINT64_MAX, image_acquired_semaphore, VK_NULL_HANDLE, &wd->FrameIndex);
		check_vk_result(err);

		ImGui_ImplVulkanH_Frame* fd = &wd->Frames[wd->FrameIndex];
		{
			err = vkWaitForFences(static_cast<RHI::Device_VK*>(rhiLayer->getDevice())->getVkDevice(), 1, &fd->Fence, VK_TRUE, UINT64_MAX);    // wait indefinitely instead of periodically checking
			check_vk_result(err);
			err = vkResetFences(static_cast<RHI::Device_VK*>(rhiLayer->getDevice())->getVkDevice(), 1, &fd->Fence);
			check_vk_result(err);
		}
		{
			err = vkResetCommandPool(static_cast<RHI::Device_VK*>(rhiLayer->getDevice())->getVkDevice(), fd->CommandPool, 0);
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
			VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
			VkSubmitInfo info = {};
			info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
			info.waitSemaphoreCount = 1;
			info.pWaitSemaphores = &image_acquired_semaphore;
			info.pWaitDstStageMask = &wait_stage;
			info.commandBufferCount = 1;
			info.pCommandBuffers = &fd->CommandBuffer;
			info.signalSemaphoreCount = 1;
			info.pSignalSemaphores = &render_complete_semaphore;

			err = vkEndCommandBuffer(fd->CommandBuffer);
			check_vk_result(err);
			err = vkQueueSubmit(static_cast<RHI::Queue_VK*>(rhiLayer->getDevice()->getGraphicsQueue())->queue, 1, &info, fd->Fence);
			check_vk_result(err);
		}
	}

	auto ImGuiBackend_VK::present() -> void {
		VkSemaphore render_complete_semaphore = mainWindowData.FrameSemaphores[mainWindowData.SemaphoreIndex].RenderCompleteSemaphore;
		VkPresentInfoKHR info = {};
		info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
		info.waitSemaphoreCount = 1;
		info.pWaitSemaphores = &render_complete_semaphore;
		info.swapchainCount = 1;
		info.pSwapchains = &mainWindowData.Swapchain;
		info.pImageIndices = &mainWindowData.FrameIndex;
		VkResult err = vkQueuePresentKHR(static_cast<RHI::Queue_VK*>(rhiLayer->getDevice()->getGraphicsQueue())->queue, &info);
		check_vk_result(err);
		mainWindowData.SemaphoreIndex = (mainWindowData.SemaphoreIndex + 1) % mainWindowData.ImageCount; // Now we can use the next set of semaphores
	}

	auto ImGuiBackend_VK::createImGuiTexture(RHI::Sampler* sampler, RHI::TextureView* view,
		RHI::TextureLayout layout) noexcept -> std::unique_ptr<ImGuiTexture> {
		return std::make_unique<ImGuiTexture_VK>(sampler, view, layout);
	}

}