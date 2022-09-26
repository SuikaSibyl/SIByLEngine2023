module;
#include <set>
#include <vector>
#include <format>
#include <vulkan/vulkan.h>
#include <memory>
#include <optional>
#include <limits>
#include <cstdint>
#include <algorithm>
#include <future>
#define VK_USE_PLATFORM_WIN32_KHR
#define GLFW_INCLUDE_VULKAN
#include <glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <glfw3native.h>
#include <vulkan/vulkan_win32.h>
export module RHI:VK;
import :Interface;
import Core.Log;
import Math.Limits;
import Platform.Window;

namespace SIByL::RHI
{
	// **************************
	// Initialization			|
	struct Context_VK;
	struct Adapter_VK;
	struct Device_VK;
	struct QueueFamilyIndices_VK;
	// **************************
	// Buffers					|
	struct Buffer_VK;			
	// **************************
	// Textures & Views			|
	struct Texture_VK;			
	struct TextureView_VK;		
	struct ExternalTexture_VK;	
	// **************************
	// Samplers				    |
	struct Sampler_VK;
	// **************************
	// SwapChain				|
	struct SwapChain_VK;
	// **************************
	// Resource Binding		    |
	struct BindGroupPool_VK;
	struct BindGroupLayout_VK;	
	struct BindGroup_VK;		
	// **************************
	// Command				    |
	struct CommandBuffer_VK;
	struct CommandPool_VK;
	struct MultiFrameFlights_VK;
	// **************************
	// Queue				    |
	struct Queue_VK;			
	// **************************
	// Queries				    |
	struct QuerySet_VK;			
	// **************************

	// 
	// ===========================================================================
	// Initialization Interface
	
	////////////////////////////////////
	//
	// Context
	//

	export struct Context_VK final :public Context {
		/** virtual destructor */
		virtual ~Context_VK() { destroy(); }
		/** initialize the context */
		virtual auto init(Platform::Window* window = nullptr, ContextExtensionsFlags ext = 0) noexcept -> bool override;
		/** Request an adapter */
		virtual auto requestAdapter(RequestAdapterOptions const& options = {}) noexcept -> std::unique_ptr<Adapter> override;
		/** Get the binded window */
		virtual auto getBindedWindow() const noexcept -> Platform::Window* override;
		/** clean up context resources */
		virtual auto destroy() noexcept -> void override;
	public:
		/** get VkInstance */
		auto getVkInstance() noexcept -> VkInstance& { return instance; }
		/** get VkSurfaceKHR */
		auto getVkSurfaceKHR() noexcept -> VkSurfaceKHR& { return surface; }
		/** get DebugMessageFunc */
		auto getDebugMessenger() noexcept -> VkDebugUtilsMessengerEXT& { return debugMessenger; }
		/** get All VkPhysicalDevices available */
		auto getVkPhysicalDevices() noexcept -> std::vector<VkPhysicalDevice>& { return devices; }
		/** get All Device Extensions required */
		auto getDeviceExtensions() noexcept -> std::vector<const char*>& { return deviceExtensions; }
		/** get Context Extensions Flags */
		auto getContextExtensionsFlags() const noexcept -> ContextExtensionsFlags { return extensions; }
		// Debug Ext Func Pointers
		typedef void (VKAPI_PTR* PFN_vkCmdBeginDebugUtilsLabelEXT)(VkCommandBuffer commandBuffer, const VkDebugUtilsLabelEXT* pLabelInfo);
		PFN_vkCmdBeginDebugUtilsLabelEXT vkCmdBeginDebugUtilsLabelEXT;
		typedef void (VKAPI_PTR* PFN_vkCmdEndDebugUtilsLabelEXT)(VkCommandBuffer commandBuffer);
		PFN_vkCmdEndDebugUtilsLabelEXT vkCmdEndDebugUtilsLabelEXT;
		// Mesh Shader Ext Func Pointers
		typedef void (VKAPI_PTR* PFN_vkCmdDrawMeshTasksNV)(VkCommandBuffer commandBuffer, uint32_t taskCount, uint32_t firstTask);
		PFN_vkCmdDrawMeshTasksNV vkCmdDrawMeshTasksNV;
	private:
		VkInstance instance;
		VkSurfaceKHR surface;
		VkDebugUtilsMessengerEXT debugMessenger;
		Platform::Window* bindedWindow = nullptr;
		ContextExtensionsFlags extensions = 0;
		std::vector<VkPhysicalDevice> devices;
		std::vector<const char*> deviceExtensions = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };
	};

	////////////////////////////////////
	//
	// Adapter
	//

	export struct QueueFamilyIndices_VK {
		std::optional<uint32_t> graphicsFamily;
		std::optional<uint32_t> presentFamily;
		std::optional<uint32_t> computeFamily;
		/** check whether queue families are complete */
		auto isComplete() noexcept -> bool {
			return graphicsFamily.has_value() && presentFamily.has_value() && computeFamily.has_value();
		}
	};

	export struct Adapter_VK final :public Adapter {
		/** constructor */
		Adapter_VK(VkPhysicalDevice device, Context_VK* context, VkPhysicalDeviceProperties const& properties);
		/** Requests a device from the adapter. */
		virtual auto requestDevice() noexcept -> std::unique_ptr<Device> override;
		/** Requests the AdapterInfo for this Adapter. */
		virtual auto requestAdapterInfo() const noexcept -> AdapterInfo override;
	public:
		/** get context the adapter is on */
		auto getContext() noexcept -> Context_VK* { return context; }
		/** get VkPhysicalDevice */
		auto getVkPhysicalDevice() noexcept -> VkPhysicalDevice& { return physicalDevice; }
		/** get TimestampPeriod */
		auto getTimestampPeriod() const noexcept -> float { return timestampPeriod; }
		/** get QueueFamilyIndices_VK */
		auto getQueueFamilyIndices() const noexcept -> QueueFamilyIndices_VK const& { return queueFamilyIndices; }
		/** get All Device Extensions required */
		auto getDeviceExtensions() noexcept -> std::vector<const char*>& { return context->getDeviceExtensions(); }
		/** get All Device Extensions required */
		auto findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) noexcept -> uint32_t;
	private:
		/** the context the adapter is requested from */
		Context_VK* context = nullptr;
		/** adapter information */
		AdapterInfo const adapterInfo;
		/** the graphics card selected */
		VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
		/** the timestamp period for timestemp query */
		float timestampPeriod = 0.0f;
		/** QueueFamilyIndices_VK */
		QueueFamilyIndices_VK queueFamilyIndices;
	};

	////////////////////////////////////
	//
	// Device
	//

	export struct Queue_VK :public Queue {
		/** virtual destructor */
		virtual ~Queue_VK() = default;
		/** Schedules the execution of the command buffers by the GPU on this queue. */
		virtual auto submit(std::vector<CommandBuffer*> const& commandBuffers) noexcept -> void override;
		/** Schedules the execution of the command buffers by the GPU on this queue. With sync objects */
		virtual auto submit(std::vector<CommandBuffer*> const& commandBuffers,
			Semaphore* wait, Semaphore* signal, Fence* fence) noexcept -> void override;
		/** Returns a Promise that resolves once this queue finishes
		* processing all the work submitted up to this moment. */
		virtual auto onSubmittedWorkDone() noexcept -> std::future<bool> override;
		/** Issues a write operation of the provided data into a Buffer. */
		virtual auto writeBuffer(
			Buffer* buffer,
			uint64_t bufferOffset,
			ArrayBuffer* data,
			uint64_t dataOffset,
			Extend3D const& size) noexcept -> void override;
		/** Issues a write operation of the provided data into a Texture. */
		virtual auto writeTexture(
			ImageCopyTexture const& destination,
			ArrayBuffer* data,
			ImageDataLayout const& layout,
			Extend3D const& size) noexcept -> void override;
		/** Issues a copy operation of the contents of a platform
		* image/canvas into the destination texture. */
		virtual auto copyExternalImageToTexture(
			ImageCopyExternalImage const& source,
			ImageCopyExternalImage const& destination,
			Extend3D const& copySize) noexcept -> void override;
		/** Present swap chain. */
		virtual auto presentSwapChain(
			SwapChain* swapchain,
			uint32_t imageIndex,
			Semaphore* semaphore) noexcept -> void override;
		/** wait until idle */
		virtual auto waitIdle() noexcept -> void override;
		/** Vulkan queue handle */
		VkQueue queue;
		/* the device this buffer is created on */
		Device_VK* device = nullptr;
	};

#pragma region VK_QUEUE_IMPL

	auto Queue_VK::onSubmittedWorkDone() noexcept -> std::future<bool> {
		return std::future<bool>{};
	}

	auto Queue_VK::writeBuffer(
		Buffer* buffer,
		uint64_t bufferOffset,
		ArrayBuffer* data,
		uint64_t dataOffset,
		Extend3D const& size) noexcept -> void {

	}

	auto Queue_VK::writeTexture(
		ImageCopyTexture const& destination,
		ArrayBuffer* data,
		ImageDataLayout const& layout,
		Extend3D const& size) noexcept -> void {

	}

	auto Queue_VK::copyExternalImageToTexture(
		ImageCopyExternalImage const& source,
		ImageCopyExternalImage const& destination,
		Extend3D const& copySize) noexcept -> void {

	}
	
	auto Queue_VK::waitIdle() noexcept -> void {
		vkQueueWaitIdle(queue);
	}

#pragma endregion

	export struct Device_VK final :public Device {
		/** virtual destructor */
		virtual ~Device_VK();
		/** destroy the device */
		virtual auto destroy() noexcept -> void override;
		/** wait until idle */
		virtual auto waitIdle() noexcept -> void { vkDeviceWaitIdle(device); }
		// Read-only fields
		// ---------------------------
		/** the graphics queue for this device */
		virtual auto getGraphicsQueue() noexcept -> Queue* { return &graphicsQueue; }
		/** the compute queue for this device */
		virtual auto getComputeQueue() noexcept -> Queue* { return &computeQueue; }
		/** the present queue for this device */
		virtual auto getPresentQueue() noexcept -> Queue* { return &presentQueue; }
		// Create resources on device
		// ---------------------------
		/** create a buffer on the device */
		virtual auto createBuffer(BufferDescriptor const& desc) noexcept -> std::unique_ptr<Buffer> override;
		/** create a texture on the device */
		virtual auto createTexture(TextureDescriptor const& desc) noexcept -> std::unique_ptr<Texture> override;
		/** create a sampler on the device */
		virtual auto createSampler(SamplerDescriptor const& desc) noexcept -> std::unique_ptr<Sampler> override;
		/** create a external texture on the device */
		virtual auto importExternalTexture(ExternalTextureDescriptor const& desc) noexcept -> std::unique_ptr<ExternalTexture> override;
		/* create a swapchain on the device */
		virtual auto createSwapChain(SwapChainDescriptor const& desc) noexcept -> std::unique_ptr<SwapChain> override;
		// Create resources binding objects
		// ---------------------------
		/** create a bind group layout on the device */
		virtual auto createBindGroupLayout(BindGroupLayoutDescriptor const& desc) noexcept -> std::unique_ptr<BindGroupLayout> override;
		/** create a pipeline layout on the device */
		virtual auto createPipelineLayout(PipelineLayoutDescriptor const& desc) noexcept -> std::unique_ptr<PipelineLayout> override;
		/** create a bind group on the device */
		virtual auto createBindGroup(BindGroupDescriptor const& desc) noexcept -> std::unique_ptr<BindGroup> override;
		// Create pipeline objects
		// ---------------------------
		/** create a shader module on the device */
		virtual auto createShaderModule(ShaderModuleDescriptor const& desc) noexcept -> std::unique_ptr<ShaderModule> override;
		/** create a compute pipeline on the device */
		virtual auto createComputePipeline(ComputePipelineDescriptor const& desc) noexcept -> std::unique_ptr<ComputePipeline> override;
		/** create a render pipeline on the device */
		virtual auto createRenderPipeline(RenderPipelineDescriptor const& desc) noexcept -> std::unique_ptr<RenderPipeline> override;
		/** create a compute pipeline on the device in async way */
		virtual auto createComputePipelineAsync(ComputePipelineDescriptor const& desc) noexcept
			-> std::future<std::unique_ptr<ComputePipeline>> override;
		/** create a render pipeline on the device in async way */
		virtual auto createRenderPipelineAsync(RenderPipelineDescriptor const& desc) noexcept
			-> std::future<std::unique_ptr<RenderPipeline>> override;
		// Create command encoders
		// ---------------------------
		/** create a multi frame flights */
		virtual auto createMultiFrameFlights(MultiFrameFlightsDescriptor const& desc) noexcept
			-> std::unique_ptr<MultiFrameFlights> override;
		/** create a command encoder */
		virtual auto createCommandEncoder(CommandEncoderDescriptor const& desc) noexcept
			-> std::unique_ptr<CommandEncoder> override;
		/** create a render bundle encoder */
		virtual auto createRenderBundleEncoder(CommandEncoderDescriptor const& desc) noexcept
			-> std::unique_ptr<RenderBundleEncoder> override;
		// Create query sets
		// ---------------------------
		virtual auto createQuerySet(QuerySetDescriptor const& desc) noexcept -> std::unique_ptr<QuerySet> override;
	public:
		/** get vulkan logical device handle */
		auto getVkDevice() noexcept -> VkDevice& { return device; }
		/** get graphics queue handle */
		auto getVkGraphicsQueue() noexcept -> Queue_VK& { return graphicsQueue; }
		/** get compute queue handle */
		auto getVkComputeQueue() noexcept -> Queue_VK& { return computeQueue; }
		/** get present queue handle */
		auto getVkPresentQueue() noexcept -> Queue_VK& { return presentQueue; }
		/** get the adapter from which this device was created */
		auto getAdapterVk() noexcept -> Adapter_VK*& { return adapter; }
		/** get bind group pool */
		auto getBindGroupPool() noexcept -> BindGroupPool_VK* { return bindGroupPool.get(); }
		/** create command pools */
		auto createCommandPools() noexcept -> void;
		/** create bind group pool */
		auto createBindGroupPool() noexcept -> void;
		/** create command pools */
		auto allocateCommandBuffer() noexcept -> std::unique_ptr<CommandBuffer_VK>;
	private:
		/** vulkan logical device handle */
		VkDevice device;
		/** various queue handles */
		Queue_VK graphicsQueue, computeQueue, presentQueue;
		/** various queue command pools */
		std::unique_ptr<CommandPool_VK> graphicPool = nullptr, computePool = nullptr, presentPool = nullptr;
		/** the adapter from which this device was created */
		Adapter_VK* adapter = nullptr;
		/** bind group pool */
		std::unique_ptr<BindGroupPool_VK> bindGroupPool = nullptr;
		/** multiframe flights */
		std::unique_ptr<MultiFrameFlights_VK> multiFrameFlights = nullptr;
	};

#pragma region VK_CONTEXT_IMPL

	/** Whether enable validation layer */
	constexpr bool const enableValidationLayers = true;
	/** Whether enable validation layer verbose output */
	constexpr bool const enableValidationLayerVerboseOutput = false;
	/** Possible names of validation layer */
	std::vector<const char*> const validationLayerNames = {
		"VK_LAYER_KHRONOS_validation",
	};

	/** Debug callback of vulkan validation layer */
	static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
		VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
		VkDebugUtilsMessageTypeFlagsEXT messageType,
		const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
		void* pUserData)
	{
		switch (messageSeverity)
		{
		case VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT:
			if (enableValidationLayerVerboseOutput)
				Core::LogManager::Log(std::format("VULKAN :: VALIDATION :: {}", pCallbackData->pMessage));
			break;
		case VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT:
			Core::LogManager::Log(std::format("VULKAN :: VALIDATION :: {}", pCallbackData->pMessage));
			break;
		case VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT:
			Core::LogManager::Warning(std::format("VULKAN :: VALIDATION :: {}", pCallbackData->pMessage));
			break;
		case VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT:
			Core::LogManager::Error(std::format("VULKAN :: VALIDATION :: {}", pCallbackData->pMessage));
			break;
		case VK_DEBUG_UTILS_MESSAGE_SEVERITY_FLAG_BITS_MAX_ENUM_EXT:
			Core::LogManager::Error(std::format("VULKAN :: VALIDATION :: {}", pCallbackData->pMessage));
			break;
		default:
			break;
		}
		return VK_FALSE;
	}

	inline auto checkValidationLayerSupport() noexcept -> bool {
		// get extension count
		uint32_t layerCount;
		vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
		// get extension details
		std::vector<VkLayerProperties> availableLayers(layerCount);
		vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());
		// for each possible name
		for (char const* layerName : validationLayerNames) {
			bool layerFound = false;
			// compare with every abailable layer name
			for (auto const& layerProperties : availableLayers) {
				if (strcmp(layerName, layerProperties.layerName) == 0) {
					layerFound = true;
					break;
				}
			}
			// layer not found
			if (!layerFound) {
				return false;
			}
		}
		// find validation layer
		return true;
	}

#define VK_KHR_WIN32_SURFACE_EXTENSION_NAME "VK_KHR_win32_surface"

	auto getRequiredExtensions(Context_VK* context, ContextExtensionsFlags& ext) noexcept -> std::vector<const char*> {
		// extensions needed
		std::vector<const char*> extensions;
		// add glfw extension
		if (context->getBindedWindow()->getVendor() == Platform::WindowVendor::GLFW) {
			uint32_t glfwExtensionCount = 0;
			char const** glfwExtensions;
			glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
			std::vector<const char*> glfwExtensionNames(glfwExtensions, glfwExtensions + glfwExtensionCount);
			extensions.insert(extensions.end(), glfwExtensionNames.begin(), glfwExtensionNames.end());
		}
		// add extensions that glfw needs
		else if (context->getBindedWindow()->getVendor() == Platform::WindowVendor::WIN_64) {
			extensions.emplace_back(VK_KHR_SURFACE_EXTENSION_NAME);
			extensions.emplace_back(VK_KHR_WIN32_SURFACE_EXTENSION_NAME);
		}
		// add other extensions according to ext bits
		if (ext & (ContextExtensionsFlags)ContextExtension::MESH_SHADER) {
			extensions.emplace_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
		}
		// add extensions that validation layer needs
		if (enableValidationLayers) {
			ext = ContextExtensionsFlags((uint32_t)ext | (uint32_t)ContextExtension::DEBUG_UTILS);
			extensions.emplace_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
		}
		// finialize collection
		return extensions;
	}

	void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo) {
		createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
		createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
		createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
		createInfo.pfnUserCallback = debugCallback;
	}

	auto createInstance(Context_VK* context, ContextExtensionsFlags& ext) noexcept -> void {
		// Check we could enable validation layers
		if (enableValidationLayers && !checkValidationLayerSupport())
			Core::LogManager::Error("Vulkan :: validation layers requested, but not available!");
		// Optional, but it may provide some useful information to 
		// the driver in order to optimize our specific application
		VkApplicationInfo appInfo{};
		appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
		appInfo.pApplicationName = "SIByLEngine";
		appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.pEngineName = "No Engine";
		appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.apiVersion = VK_API_VERSION_1_2;
		// Not optional, specify the desired global extensions
		auto extensions = getRequiredExtensions(context, ext);
		// Not optional,
		// Tells the Vulkan driver which global extensions and validation layers we want to use.
		// Global here means that they apply to the entire program and not a specific device,
		VkInstanceCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		createInfo.pApplicationInfo = &appInfo;
		createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
		createInfo.ppEnabledExtensionNames = extensions.data();
		// determine the global validation layers to enable
		VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
		if (enableValidationLayers) {
			createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayerNames.size());
			createInfo.ppEnabledLayerNames = validationLayerNames.data();
			// add debug messenger for init
			populateDebugMessengerCreateInfo(debugCreateInfo);
			createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;
		}
		else {
			createInfo.enabledLayerCount = 0;
			createInfo.pNext = nullptr;
		}
		// Create Vk Instance
		if (vkCreateInstance(&createInfo, nullptr, &(context->getVkInstance())) != VK_SUCCESS) {
			Core::LogManager::Error("Vulkan :: Failed to create instance!");
		}
	}

	VkResult CreateDebugUtilsMessengerEXT(
		VkInstance instance,
		VkDebugUtilsMessengerCreateInfoEXT const* pCreateInfo,
		VkAllocationCallbacks const* pAllocator,
		VkDebugUtilsMessengerEXT* pDebugMessenger)
	{
		auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
		if (func != nullptr) return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
		else return VK_ERROR_EXTENSION_NOT_PRESENT;
	}

	auto setupDebugMessenger(Context_VK* context) noexcept -> void {
		if (!enableValidationLayers) return;
		VkDebugUtilsMessengerCreateInfoEXT createInfo;
		populateDebugMessengerCreateInfo(createInfo);
		// load function from extern
		if (CreateDebugUtilsMessengerEXT(context->getVkInstance(), &createInfo, nullptr, &context->getDebugMessenger()) != VK_SUCCESS)
			Core::LogManager::Error("Vulkan :: failed to set up debug messenger!");
	}

	PFN_vkVoidFunction vkGetInstanceProcAddrStub(void* context, const char* name) {
		return vkGetInstanceProcAddr((VkInstance)context, name);
	}

	auto setupExtensions(Context_VK* context, ContextExtensionsFlags& ext) -> void {
		if (ext & (ContextExtensionsFlags)ContextExtension::DEBUG_UTILS) {
			context->vkCmdBeginDebugUtilsLabelEXT = (PFN_vkCmdBeginDebugUtilsLabelEXT)vkGetInstanceProcAddrStub(context->getVkInstance(), "vkCmdBeginDebugUtilsLabelEXT");
			context->vkCmdEndDebugUtilsLabelEXT = (PFN_vkCmdEndDebugUtilsLabelEXT)vkGetInstanceProcAddrStub(context->getVkInstance(), "vkCmdEndDebugUtilsLabelEXT");
		}
		if (ext & (ContextExtensionsFlags)ContextExtension::MESH_SHADER) {
			context->vkCmdDrawMeshTasksNV = (PFN_vkCmdDrawMeshTasksNV)vkGetInstanceProcAddrStub(context->getVkInstance(), "vkCmdDrawMeshTasksNV");
			context->getDeviceExtensions().emplace_back(VK_NV_MESH_SHADER_EXTENSION_NAME);
		}
	}

	auto attachWindow(Context_VK* contexVk) noexcept -> void {
		if (contexVk->getBindedWindow()->getVendor() == Platform::WindowVendor::GLFW) {
			if (glfwCreateWindowSurface(contexVk->getVkInstance(), (GLFWwindow*)contexVk->getBindedWindow()->getHandle(), 
				nullptr, &contexVk->getVkSurfaceKHR()) != VK_SUCCESS) {
				Core::LogManager::Error("Vulkan :: glfwCreateWindowSurface failed!");
			}
		}
		else if (contexVk->getBindedWindow()->getVendor() == Platform::WindowVendor::WIN_64) {
			VkWin32SurfaceCreateInfoKHR createInfo{};
			createInfo.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
			createInfo.hwnd = (HWND)contexVk->getBindedWindow()->getHandle();
			createInfo.hinstance = GetModuleHandle(nullptr);
			if (vkCreateWin32SurfaceKHR(contexVk->getVkInstance(), &createInfo, 
				nullptr, &contexVk->getVkSurfaceKHR()) != VK_SUCCESS) {
				Core::LogManager::Error("Vulkan :: failed to create WIN_64 window surface!");
			}
		}
	}

	auto Context_VK::init(Platform::Window* window, ContextExtensionsFlags ext) noexcept -> bool {
		bindedWindow = window;
		// create VkInstance
		createInstance(this, ext);
		setupDebugMessenger(this);
		setupExtensions(this, ext);
		attachWindow(this);
		// set extensions
		extensions = ext;
		return true;
	}

	auto findQueueFamilies(Context_VK* contextVk, VkPhysicalDevice& device) noexcept -> QueueFamilyIndices_VK {
		QueueFamilyIndices_VK indices;
		// Logic to find queue family indices to populate struct with
		uint32_t queueFamilyCount = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);
		std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());
		// find at least one queue family that supports VK_QUEUE_GRAPHICS_BIT
		int i = 0;
		for (auto const& queueFamily : queueFamilies) {
			// check graphic support
			if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
				indices.graphicsFamily = i;
				if (queueFamily.timestampValidBits <= 0)
					Core::LogManager::Error("VULKAN :: Graphics Family not support timestamp ValidBits");
			}
			// check queue support
			if (queueFamily.queueFlags & VK_QUEUE_COMPUTE_BIT) {
				indices.computeFamily = i;
			}
			// check present support
			VkBool32 presentSupport = false;
			vkGetPhysicalDeviceSurfaceSupportKHR(device, i, contextVk->getVkSurfaceKHR(), &presentSupport);
			if (presentSupport) indices.presentFamily = i;
			// check support completeness
			if (indices.isComplete()) break;
			i++;
		}
		return indices;
	}

	auto checkDeviceExtensionSupport(Context_VK* contextVk, VkPhysicalDevice& device, std::string& device_diagnosis) noexcept -> bool {
		// find all available extensions
		uint32_t extensionCount;
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);
		std::vector<VkExtensionProperties> availableExtensions(extensionCount);
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());
		// find all required extensions
		std::set<std::string> requiredExtensions(contextVk->getDeviceExtensions().begin(), contextVk->getDeviceExtensions().end());
		// find all required-but-not-available extensions
		for (const auto& extension : availableExtensions) 
			requiredExtensions.erase(extension.extensionName);
		// create device diagnosis error codes if there are invalid requirement
		if (!requiredExtensions.empty()) {
			device_diagnosis.append("Required Extension not supported: ");
			for (const auto& extension : requiredExtensions) {
				device_diagnosis.append(extension);
				device_diagnosis.append(" | ");
			}
		}
		return requiredExtensions.empty();
	}

	struct SwapChainSupportDetails {
		VkSurfaceCapabilitiesKHR		capabilities;
		std::vector<VkSurfaceFormatKHR> formats;
		std::vector<VkPresentModeKHR>	presentModes;
	};

	auto querySwapChainSupport(Context_VK* contextVk, VkPhysicalDevice& device)->SwapChainSupportDetails {
		SwapChainSupportDetails details;
		// query basic surface capabilities
		vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, contextVk->getVkSurfaceKHR(), &details.capabilities);
		// query the supported surface formats
		uint32_t formatCount;
		vkGetPhysicalDeviceSurfaceFormatsKHR(device, contextVk->getVkSurfaceKHR(), &formatCount, nullptr);
		if (formatCount != 0) {
			details.formats.resize(formatCount);
			vkGetPhysicalDeviceSurfaceFormatsKHR(device, contextVk->getVkSurfaceKHR(), &formatCount, details.formats.data());
		}
		// query the supported presentation modes
		uint32_t presentModeCount;
		vkGetPhysicalDeviceSurfacePresentModesKHR(device, contextVk->getVkSurfaceKHR(), &presentModeCount, nullptr);
		if (presentModeCount != 0) {
			details.presentModes.resize(presentModeCount);
			vkGetPhysicalDeviceSurfacePresentModesKHR(device, contextVk->getVkSurfaceKHR(), &presentModeCount, details.presentModes.data());
		}
		return details;
	}

	auto isDeviceSuitable(Context_VK* contextVk, VkPhysicalDevice& device, std::string& device_diagnosis) noexcept -> bool {
		// check queue family supports
		QueueFamilyIndices_VK indices = findQueueFamilies(contextVk, device);
		// check extension supports
		bool const extensionsSupported = checkDeviceExtensionSupport(contextVk, device, device_diagnosis);
		// check swapchain support
		bool swapChainAdequate = false;
		if (extensionsSupported) {
			SwapChainSupportDetails swapChainSupport = querySwapChainSupport(contextVk, device);
			swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
		}
		// check physical device feature supported
		VkPhysicalDeviceFeatures supportedFeatures;
		vkGetPhysicalDeviceFeatures(device, &supportedFeatures);
		bool physicalDeviceFeatureSupported = supportedFeatures.samplerAnisotropy;
		return indices.isComplete() && extensionsSupported && swapChainAdequate && physicalDeviceFeatureSupported;
	}

	auto rateDeviceSuitability(VkPhysicalDevice& device) noexcept -> int {
		VkPhysicalDeviceProperties deviceProperties;
		VkPhysicalDeviceFeatures deviceFeatures;
		vkGetPhysicalDeviceProperties(device, &deviceProperties);
		vkGetPhysicalDeviceFeatures(device, &deviceFeatures);\
		int score = 0;
		// Discrete GPUs have a significant performance advantage
		if (deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
			score += 1000;
		}
		// Maximum possible size of textures affects graphics quality
		score += deviceProperties.limits.maxImageDimension2D;
		// Application can't function without geometry shaders
		if (!deviceFeatures.geometryShader) return 0;
		return score;
	}

	auto queryAllPhysicalDevice(Context_VK* contextVk) noexcept -> void {
		uint32_t deviceCount = 0;
		vkEnumeratePhysicalDevices(contextVk->getVkInstance(), &deviceCount, nullptr);
		// If there are 0 devices with Vulkan support
		if (deviceCount == 0)
			Core::LogManager::Error("VULKAN :: Failed to find GPUs with Vulkan support!");
		// get all of the VkPhysicalDevice handles
		std::vector<VkPhysicalDevice>& devices = contextVk->getVkPhysicalDevices();
		devices.resize(deviceCount);
		vkEnumeratePhysicalDevices(contextVk->getVkInstance(), &deviceCount, devices.data());
		// check if any of the physical devices meet the requirements
		int i = 0;
		for (const auto& device : devices) {
			VkPhysicalDeviceProperties deviceProperties;
			vkGetPhysicalDeviceProperties(device, &deviceProperties);
			Core::LogManager::Log(std::format("VULKAN :: Physical Device [{}] Found, {}", i, deviceProperties.deviceName));
			++i;
		}
		// Find the best
		std::vector<std::string> diagnosis;
		std::vector<int> scores;
		for (auto& device : devices) {
			std::string device_diagnosis;
			if (isDeviceSuitable(contextVk, device, device_diagnosis)) {
				int rate = rateDeviceSuitability(device);
				scores.emplace_back(rate);
			}
			else {
				diagnosis.emplace_back(std::move(device_diagnosis));
				scores.emplace_back(0);
			}
		}
		for (int i = 0; i < devices.size(); ++i) 
			for (int j = i + 1; j < devices.size(); ++j) {
				if (scores[i] < scores[j]) {
					std::swap(scores[i], scores[j]);
					std::swap(devices[i], devices[j]);
				}
			}
	}

	auto Context_VK::requestAdapter(RequestAdapterOptions const& options) noexcept -> std::unique_ptr<Adapter> {
		if (devices.size() == 0)
			queryAllPhysicalDevice(this);
		
		if (devices.size() == 0)
			return nullptr;
		else {
			VkPhysicalDeviceProperties deviceProperties;
			vkGetPhysicalDeviceProperties(devices[0], &deviceProperties);
			Core::LogManager::Debug(std::format("VULKAN :: Adapter selected, Name: {}", deviceProperties.deviceName));
			return std::make_unique<Adapter_VK>(devices[0], this, deviceProperties);
		}
	}

	auto Context_VK::getBindedWindow() const noexcept -> Platform::Window* {
		return bindedWindow;
	}

	auto inline destroyDebugUtilsMessengerEXT(
		VkInstance instance, 
		VkDebugUtilsMessengerEXT debugMessenger, 
		VkAllocationCallbacks const* pAllocator) noexcept -> void
	{
		auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
		if (func != nullptr) func(instance, debugMessenger, pAllocator);
	}

	auto Context_VK::destroy() noexcept -> void {
		if (enableValidationLayers)
			destroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
		if (surface) vkDestroySurfaceKHR(instance, surface, nullptr);
		vkDestroyInstance(instance, nullptr);
	}

#pragma endregion

#pragma region VK_ADAPTER_IMPL

	Adapter_VK::Adapter_VK(VkPhysicalDevice device, Context_VK* context, VkPhysicalDeviceProperties const& properties)
		: physicalDevice(device), context(context), adapterInfo(
			[&]()->AdapterInfo {
				AdapterInfo info;
				info.device = properties.deviceName;;
				info.vendor = properties.vendorID;
				info.architecture = properties.deviceType;
				info.description = properties.deviceID;
				return info; }())
		, timestampPeriod(properties.limits.timestampPeriod)
					, queueFamilyIndices(findQueueFamilies(context, physicalDevice))
	{}

	auto Adapter_VK::requestDevice() noexcept -> std::unique_ptr<Device> {
		// get queues
		std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
		std::set<uint32_t> uniqueQueueFamilies = {
			queueFamilyIndices.graphicsFamily.value(),
			queueFamilyIndices.presentFamily.value(),
			queueFamilyIndices.computeFamily.value() };
		// Desc VkDeviceQueueCreateInfo
		VkDeviceQueueCreateInfo queueCreateInfo{};
		// the number of queues we want for a single queue family
		float queuePriority = 1.0f;	// a queue with graphics capabilities
		for (uint32_t queueFamily : uniqueQueueFamilies) {
			VkDeviceQueueCreateInfo queueCreateInfo{};
			queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
			queueCreateInfo.queueFamilyIndex = queueFamily;
			queueCreateInfo.queueCount = 1;
			queueCreateInfo.pQueuePriorities = &queuePriority;
			queueCreateInfos.push_back(queueCreateInfo);
		}
		// Desc Vk Physical Device Features
		// - the set of device features that we'll be using
		VkPhysicalDeviceFeatures deviceFeatures{};
		deviceFeatures.samplerAnisotropy = VK_TRUE;
		// Desc Vk Device Create Info
		VkDeviceCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
		createInfo.pEnabledFeatures = &deviceFeatures;
		createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
		createInfo.pQueueCreateInfos = queueCreateInfos.data();
		// enable extesions
		createInfo.enabledExtensionCount = static_cast<uint32_t>(getDeviceExtensions().size());
		createInfo.ppEnabledExtensionNames = getDeviceExtensions().data();
		// get all physical device features chain
		void const** pNextChainHead = &(createInfo.pNext);
		void** pNextChainTail = nullptr;
		VkPhysicalDeviceHostQueryResetFeatures resetFeatures;
		// Add various features
		VkPhysicalDeviceMeshShaderFeaturesNV mesh_shader_feature{};
		if (context->getContextExtensionsFlags() & (ContextExtensionsFlags)ContextExtension::MESH_SHADER) {
			mesh_shader_feature.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_NV;
			mesh_shader_feature.pNext = nullptr;
			mesh_shader_feature.taskShader = VK_TRUE;
			mesh_shader_feature.meshShader = VK_TRUE;
			if (pNextChainTail == nullptr)
				*pNextChainHead = &mesh_shader_feature;
			else
				*pNextChainTail = &mesh_shader_feature;
			pNextChainTail = &(mesh_shader_feature.pNext);
		}
		VkPhysicalDeviceFragmentShaderBarycentricFeaturesNV  shader_fragment_barycentric{};
		if (context->getContextExtensionsFlags() & (ContextExtensionsFlags)ContextExtension::FRAGMENT_BARYCENTRIC) {
			shader_fragment_barycentric.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_BARYCENTRIC_FEATURES_NV;
			shader_fragment_barycentric.pNext = nullptr;
			shader_fragment_barycentric.fragmentShaderBarycentric = VK_TRUE;
			if (pNextChainTail == nullptr)
				*pNextChainHead = &shader_fragment_barycentric;
			else
				*pNextChainTail = &shader_fragment_barycentric;
			pNextChainTail = &(shader_fragment_barycentric.pNext);
		}
		VkPhysicalDeviceVulkan12Features sampler_filter_min_max_properties{};
		if (context->getContextExtensionsFlags() & (ContextExtensionsFlags)ContextExtension::SAMPLER_FILTER_MIN_MAX) {
			sampler_filter_min_max_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
			sampler_filter_min_max_properties.pNext = nullptr;
			sampler_filter_min_max_properties.samplerFilterMinmax = VK_TRUE;
			sampler_filter_min_max_properties.shaderInt8 = VK_TRUE;
			sampler_filter_min_max_properties.hostQueryReset = VK_TRUE;
			if (pNextChainTail == nullptr)
				*pNextChainHead = &sampler_filter_min_max_properties;
			else
				*pNextChainTail = &sampler_filter_min_max_properties;
			pNextChainTail = &(sampler_filter_min_max_properties.pNext);
		}
		// create logical device
		std::unique_ptr<Device_VK> device = std::make_unique<Device_VK>();
		device->getAdapterVk() = this;
		if (vkCreateDevice(getVkPhysicalDevice(), &createInfo, nullptr, &device->getVkDevice()) != VK_SUCCESS) {
			Core::LogManager::Log("VULKAN :: failed to create logical device!");
		}
		// get queue handlevul		
		vkGetDeviceQueue(device->getVkDevice(), queueFamilyIndices.graphicsFamily.value(), 0, &device->getVkGraphicsQueue().queue);
		vkGetDeviceQueue(device->getVkDevice(), queueFamilyIndices.presentFamily.value(), 0, &device->getVkPresentQueue().queue);
		vkGetDeviceQueue(device->getVkDevice(), queueFamilyIndices.computeFamily.value(), 0, &device->getVkComputeQueue().queue);
		device->getVkGraphicsQueue().device = device.get();
		device->getVkPresentQueue().device = device.get();
		device->getVkComputeQueue().device = device.get();
		device->createCommandPools();
		device->createBindGroupPool();
		return std::move(device);
	}

	auto Adapter_VK::requestAdapterInfo() const noexcept -> AdapterInfo {
		return adapterInfo;
	}

	auto Adapter_VK::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) noexcept -> uint32_t {
		VkPhysicalDeviceMemoryProperties memProperties;
		vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
		for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
			if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
				return i;
		Core::LogManager::Log("VULKAN :: failed to find suitable memory type!");
	}

#pragma endregion

#pragma region VK_DEVICE_IMPL

	Device_VK::~Device_VK() { destroy(); }

	auto Device_VK::destroy() noexcept -> void {
		graphicPool = nullptr, computePool = nullptr, presentPool = nullptr;
		bindGroupPool = nullptr;
		if (device) vkDestroyDevice(device, nullptr);
	}

	inline auto getVkBufferUsageFlags(BufferUsagesFlags usage) noexcept -> VkBufferUsageFlags {
		VkBufferUsageFlags flags = 0;
		if (usage & (uint32_t)BufferUsage::COPY_SRC		)	flags |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
		if (usage & (uint32_t)BufferUsage::COPY_DST		)	flags |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
		if (usage & (uint32_t)BufferUsage::INDEX		)	flags |= VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
		if (usage & (uint32_t)BufferUsage::VERTEX		)	flags |= VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
		if (usage & (uint32_t)BufferUsage::UNIFORM		)	flags |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
		if (usage & (uint32_t)BufferUsage::STORAGE		)	flags |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
		if (usage & (uint32_t)BufferUsage::INDIRECT		)	flags |= VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
		if (usage & (uint32_t)BufferUsage::QUERY_RESOLVE)	flags |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
		return flags;
	}

	auto Device_VK::createTexture(TextureDescriptor const& desc) noexcept -> std::unique_ptr<Texture> {
		return nullptr;
	}

	auto Device_VK::createSampler(SamplerDescriptor const& desc) noexcept -> std::unique_ptr<Sampler> {
		return nullptr;
	}

	auto Device_VK::importExternalTexture(ExternalTextureDescriptor const& desc) noexcept -> std::unique_ptr<ExternalTexture> {
		return nullptr;
	}
	
	auto Device_VK::createComputePipeline(ComputePipelineDescriptor const& desc) noexcept -> std::unique_ptr<ComputePipeline> {
		return nullptr;
	}

	auto Device_VK::createComputePipelineAsync(ComputePipelineDescriptor const& desc) noexcept
		-> std::future<std::unique_ptr<ComputePipeline>> {
		std::future<std::unique_ptr<ComputePipeline>> r;
		return r;
	}

	auto Device_VK::createRenderPipelineAsync(RenderPipelineDescriptor const& desc) noexcept
		-> std::future<std::unique_ptr<RenderPipeline>> {
		std::future<std::unique_ptr<RenderPipeline>> r;
		return r;
	}

	auto Device_VK::createRenderBundleEncoder(CommandEncoderDescriptor const& desc) noexcept
		-> std::unique_ptr<RenderBundleEncoder> {
		return nullptr;
	}

	auto Device_VK::createQuerySet(QuerySetDescriptor const& desc) noexcept -> std::unique_ptr<QuerySet> {
		return nullptr;
	}

	auto Device_VK::createCommandPools() noexcept -> void {
		graphicPool = std::make_unique<CommandPool_VK>(this);
	}

#pragma endregion

	// Initialization Interface
	// ===========================================================================
	// Buffers Interface

	export struct Buffer_VK :public Buffer {
		/** constructor */
		Buffer_VK(Device_VK* device) :device(device) {}
		/** virtual destructor */
		virtual ~Buffer_VK();
		/** copy functions */
		Buffer_VK(Buffer_VK const& buffer) = delete;
		Buffer_VK(Buffer_VK&& buffer);
		auto operator=(Buffer_VK const& buffer) -> Buffer_VK & = delete;
		auto operator=(Buffer_VK&& buffer) -> Buffer_VK&;
		// Readonly Attributes
		// ---------------------------
		/** readonly get buffer size on GPU */
		virtual auto size() const noexcept -> size_t override { return _size; }
		/** readonly get buffer usage flags on GPU */
		virtual auto bufferUsageFlags() const noexcept -> BufferUsagesFlags override { return descriptor.usage; }
		/** readonly get map state on GPU */
		virtual auto bufferMapState() const noexcept -> BufferMapState override { return mapState; }
		// Map methods
		// ---------------------------
		/** Maps the given range of the GPUBuffer */
		virtual auto mapAsync(MapModeFlags mode, size_t offset = 0, size_t size = 0) noexcept -> std::future<bool> override;
		/** Returns an ArrayBuffer with the contents of the GPUBuffer in the given mapped range */
		virtual auto getMappedRange(size_t offset = 0, size_t size = 0) noexcept -> ArrayBuffer override;
		/** Unmaps the mapped range of the GPUBuffer and makes it’s contents available for use by the GPU again. */
		virtual auto unmap() noexcept -> void override;
		// Lifecycle methods
		// ---------------------------
		/** destroy the buffer */
		virtual auto destroy() const noexcept -> void override;
	public:
		/** initialize the buffer */
		auto init(Device_VK* device, size_t size, BufferDescriptor const& desc) noexcept -> void;
		/** get vulkan buffer */
		auto getVkBuffer() noexcept -> VkBuffer& { return buffer; }
		/** get vulkan buffer device memory */
		auto getVkDeviceMemory() noexcept -> VkDeviceMemory& { return bufferMemory; }
		/** set buffer state */
		auto setBufferMapState(BufferMapState const& state) noexcept -> void { mapState = state; }
	protected:
		/** vulkan buffer */
		VkBuffer buffer = {};
		/** vulkan buffer device memory */
		VkDeviceMemory bufferMemory = {};
		/** buffer creation desc */
		BufferDescriptor descriptor = {};
		/** buffer creation desc */
		BufferMapState mapState = BufferMapState::UNMAPPED;
		/** mapped address of the buffer */
		void* mappedData = nullptr;
		/** size of the buffer */
		size_t _size = 0;
		/* the device this buffer is created on */
		Device_VK* device = nullptr;
	};

#pragma region VK_BUFFER_IMPL

	void createBuffer(
		VkDeviceSize size,
		VkBufferUsageFlags usage,
		VkSharingMode shareMode,
		VkMemoryPropertyFlags properties,
		VkBuffer& buffer,
		VkDeviceMemory& bufferMemory,
		Device_VK* device)
	{
		// create buffer
		VkBufferCreateInfo bufferInfo{};
		bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bufferInfo.size = size;
		bufferInfo.usage = usage;
		bufferInfo.sharingMode = shareMode;
		if (vkCreateBuffer(device->getVkDevice(), &bufferInfo, nullptr, &buffer) != VK_SUCCESS)
			Core::LogManager::Log("VULKAN :: failed to create buffer!");
		// alloc memory
		VkMemoryRequirements memRequirements;
		vkGetBufferMemoryRequirements(device->getVkDevice(), buffer, &memRequirements);
		VkMemoryAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memRequirements.size;
		allocInfo.memoryTypeIndex = device->getAdapterVk()->findMemoryType(memRequirements.memoryTypeBits, properties);
		if (vkAllocateMemory(device->getVkDevice(), &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS)
			Core::LogManager::Log("VULKAN :: failed to allocate buffer memory!");
		vkBindBufferMemory(device->getVkDevice(), buffer, bufferMemory, 0);
	}

#define MACRO_BUFFER_USAGE_BITMAP(USAGE) \
		if ((uint32_t)usage & (uint32_t)SIByL::RHI::BufferUsageFlagBits::USAGE)\
		{\
			flags |= VK_BUFFER_USAGE_##USAGE;\
		}\

	inline auto getVkBufferShareMode(BufferShareMode shareMode) noexcept -> VkSharingMode {
		if (shareMode == SIByL::RHI::BufferShareMode::CONCURRENT) return VK_SHARING_MODE_CONCURRENT;
		else if (shareMode == SIByL::RHI::BufferShareMode::EXCLUSIVE) return VK_SHARING_MODE_EXCLUSIVE;
		else return VK_SHARING_MODE_MAX_ENUM;
	}
	
	inline auto getVkMemoryProperty(MemoryPropertiesFlags memoryProperties) noexcept -> VkMemoryPropertyFlags {
		uint32_t flags{};
		if ((uint32_t)memoryProperties & (uint32_t)MemoryProperty::DEVICE_LOCAL_BIT)
			flags |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
		if ((uint32_t)memoryProperties & (uint32_t)MemoryProperty::HOST_VISIBLE_BIT)
			flags |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
		if ((uint32_t)memoryProperties & (uint32_t)MemoryProperty::HOST_COHERENT_BIT)
			flags |= VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
		if ((uint32_t)memoryProperties & (uint32_t)MemoryProperty::HOST_CACHED_BIT)
			flags |= VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
		if ((uint32_t)memoryProperties & (uint32_t)MemoryProperty::LAZILY_ALLOCATED_BIT)
			flags |= VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT;
		if ((uint32_t)memoryProperties & (uint32_t)MemoryProperty::PROTECTED_BIT)
			flags |= VK_MEMORY_PROPERTY_PROTECTED_BIT;
		return (VkMemoryPropertyFlags)flags;
	}
	
	Buffer_VK::~Buffer_VK() {
		destroy();
	}

	Buffer_VK::Buffer_VK(Buffer_VK&& x)
		: buffer(x.buffer), bufferMemory(x.bufferMemory), _size(x._size)
	{
		x.buffer = nullptr;
		x.bufferMemory = nullptr;
	}
	
	auto Buffer_VK::operator=(Buffer_VK&& x) -> Buffer_VK& {
		buffer = x.buffer;
		bufferMemory = x.bufferMemory;
		_size = x._size;
		x.buffer = nullptr;
		x.bufferMemory = nullptr;
		return *this;
	}

	auto Buffer_VK::init(Device_VK* device, size_t size, BufferDescriptor const& desc) noexcept -> void {
		this->_size = size;
		this->device = device;
		//createBuffer(
		//	_size,
		//	getVkBufferUsage(desc.usage),
		//	getVkBufferShareMode(desc.shareMode),
		//	getVkMemoryProperty(desc.memoryProperties), 
		//	buffer,
		//	bufferMemory,
		//	device
		//);
	}
	
	inline auto mapMemory(Device_VK* device, Buffer_VK* buffer, size_t offset, size_t size, void*& mappedData) noexcept-> bool {
		VkResult result = vkMapMemory(device->getVkDevice(), buffer->getVkDeviceMemory(), offset, size, 0, &mappedData);
		if (result) buffer->setBufferMapState(BufferMapState::MAPPED);
		return result == VkResult::VK_SUCCESS ? true : false;
	}

	auto Buffer_VK::mapAsync(MapModeFlags mode, size_t offset, size_t size) noexcept -> std::future<bool> {
		mapState = BufferMapState::PENDING;
		return std::async(mapMemory, device, this, offset, size, std::ref(mappedData));
	}

	auto Buffer_VK::getMappedRange(size_t offset, size_t size) noexcept -> ArrayBuffer {
		return (void*)&(((char*)mappedData)[offset]);
	}

	auto Buffer_VK::unmap() noexcept -> void {
		vkUnmapMemory(device->getVkDevice(), bufferMemory);
		mappedData = nullptr;
		BufferMapState mapState = BufferMapState::UNMAPPED;
	}

	auto Buffer_VK::destroy() const noexcept -> void {
		if (buffer)		  vkDestroyBuffer(device->getVkDevice(), buffer, nullptr);
		if (bufferMemory) vkFreeMemory(device->getVkDevice(), bufferMemory, nullptr);
	}

	inline auto findMemoryType(Device_VK* device, uint32_t typeFilter, VkMemoryPropertyFlags properties) noexcept -> uint32_t {
		VkPhysicalDeviceMemoryProperties memProperties;
		vkGetPhysicalDeviceMemoryProperties(device->getAdapterVk()->getVkPhysicalDevice(), &memProperties);
		for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
			if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
				return i;
		Core::LogManager::Log("VULKAN :: failed to find suitable memory type!");
		return 0;
	}

	inline auto getVkMemoryPropertyFlags(MemoryPropertiesFlags memoryProperties) noexcept -> VkMemoryPropertyFlags {
		VkMemoryPropertyFlags flags = 0;
		if (memoryProperties & (uint32_t)MemoryProperty::DEVICE_LOCAL_BIT)		flags |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
		if (memoryProperties & (uint32_t)MemoryProperty::HOST_VISIBLE_BIT)		flags |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
		if (memoryProperties & (uint32_t)MemoryProperty::HOST_COHERENT_BIT)		flags |= VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
		if (memoryProperties & (uint32_t)MemoryProperty::HOST_CACHED_BIT)		flags |= VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
		if (memoryProperties & (uint32_t)MemoryProperty::LAZILY_ALLOCATED_BIT)	flags |= VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT;
		if (memoryProperties & (uint32_t)MemoryProperty::PROTECTED_BIT)			flags |= VK_MEMORY_PROPERTY_PROTECTED_BIT;
		if (memoryProperties == (uint32_t)MemoryProperty::FLAG_BITS_MAX_ENUM)	flags |= VK_MEMORY_PROPERTY_FLAG_BITS_MAX_ENUM;
		return flags;
	}
	
	auto Device_VK::createBuffer(BufferDescriptor const& desc) noexcept -> std::unique_ptr<Buffer> {
		VkBufferCreateInfo bufferInfo{};
		bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bufferInfo.size = desc.size;
		bufferInfo.usage = getVkBufferUsageFlags(desc.usage);
		bufferInfo.sharingMode = desc.shareMode == BufferShareMode::EXCLUSIVE ? 
			VK_SHARING_MODE_EXCLUSIVE : VK_SHARING_MODE_CONCURRENT;
		std::unique_ptr<Buffer_VK> buffer = std::make_unique<Buffer_VK>(this);
		buffer->init(this, desc.size, desc);
		if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer->getVkBuffer()) != VK_SUCCESS) {
			Core::LogManager::Log("VULKAN :: failed to create vertex buffer!");
		}
		// assign memory to buffer
		VkMemoryRequirements memRequirements;
		vkGetBufferMemoryRequirements(device, buffer->getVkBuffer(), &memRequirements);
		VkMemoryAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memRequirements.size;
		allocInfo.memoryTypeIndex = findMemoryType(this, memRequirements.memoryTypeBits, getVkMemoryPropertyFlags(desc.memoryProperties));
		if (vkAllocateMemory(device, &allocInfo, nullptr, &buffer->getVkDeviceMemory()) != VK_SUCCESS) {
			Core::LogManager::Log("VULKAN :: failed to allocate vertex buffer memory!");
		}
		vkBindBufferMemory(device, buffer->getVkBuffer(), buffer->getVkDeviceMemory(), 0);
		return buffer;
	}

#pragma endregion

	// Buffers Interface
	// ===========================================================================
	// Textures/TextureViews Interface

	export struct Texture_VK :public Texture {
		// Texture Behaviors
		// ---------------------------
		/** constructor */
		Texture_VK(Device_VK* device, TextureDescriptor const& desc);
		Texture_VK(Device_VK* device, VkImage image, TextureDescriptor const& desc);
		/** virtual descructor */
		virtual ~Texture_VK();
		/** create texture view of this texture */
		virtual auto createView(TextureViewDescriptor const& desc) noexcept -> std::unique_ptr<TextureView> override;
		/** destroy this texture */
		virtual auto destroy() noexcept -> void override;
		// Readonly Attributes
		// ---------------------------
		/** readonly width of the texture */
		virtual auto width() const noexcept -> uint32_t override;
		/** readonly height of the texture */
		virtual auto height() const noexcept -> uint32_t override;
		/** readonly depth or arrayLayers of the texture */
		virtual auto depthOrArrayLayers() const noexcept -> uint32_t override;
		/** readonly mip level count of the texture */
		virtual auto mipLevelCount() const noexcept -> uint32_t override;
		/** readonly sample count of the texture */
		virtual auto sampleCount() const noexcept -> uint32_t override;
		/** the dimension of the set of texel for each of this GPUTexture's subresources. */
		virtual auto dimension() const noexcept -> TextureDimension override;
		/** readonly format of the texture */
		virtual auto format() const noexcept -> TextureFormat override;
	public:
		/** get the VkImage */
		auto getVkImage() noexcept -> VkImage& { return image; }
	private:
		/** vulkan image */
		VkImage image;
		/** vulkan image device memory */
		VkDeviceMemory deviceMemory = nullptr;
		/** Texture Descriptor */
		TextureDescriptor descriptor;
		/** the device this texture is created on */
		Device_VK* device = nullptr;
	};

	export struct TextureView_VK :public TextureView {
		/** create textureviw */
		TextureView_VK(Device_VK* device, Texture_VK* texture, TextureViewDescriptor const& descriptor);
		/* copy functions */
		TextureView_VK(TextureView_VK const& view) = delete;
		TextureView_VK(TextureView_VK&& view);
		auto operator=(TextureView_VK const& view) -> TextureView_VK& = delete;
		auto operator=(TextureView_VK&& view) -> TextureView_VK&;
		/** virtual destructor */
		virtual ~TextureView_VK();
		/** get binded texture */
		virtual auto getTexture() noexcept -> Texture* { return texture; }
		/** Vulkan texture view */
		VkImageView imageView;
		/** Texture view descriptor */
		TextureViewDescriptor descriptor;
		/** The texture this view is pointing to */
		Texture_VK* texture = nullptr;
		/** The device that the pointed texture is created on */
		Device_VK* device = nullptr;
	};

#pragma region VK_TEXTURE_IMPL

	Texture_VK::Texture_VK(Device_VK* device, TextureDescriptor const& desc)
		: device(device), descriptor{desc}
	{}

	Texture_VK::Texture_VK(Device_VK* device, VkImage image, TextureDescriptor const& desc)
		: device(device), image(image), descriptor{ desc }
	{}

	Texture_VK::~Texture_VK() {
		destroy();
	}

	auto Texture_VK::createView(TextureViewDescriptor const& desc) noexcept -> std::unique_ptr<TextureView> {
		return std::make_unique<TextureView_VK>(device, this, desc);
	}

	auto Texture_VK::destroy() noexcept -> void {
		if (image && deviceMemory) vkDestroyImage(device->getVkDevice(), image, nullptr);
		if (deviceMemory) vkFreeMemory(device->getVkDevice(), deviceMemory, nullptr);
	}

	auto Texture_VK::width() const noexcept -> uint32_t {
		return descriptor.size.width;
	}

	auto Texture_VK::height() const noexcept -> uint32_t {
		return descriptor.size.height;
	}

	auto Texture_VK::depthOrArrayLayers() const noexcept -> uint32_t {
		return descriptor.size.depthOrArrayLayers;
	}

	auto Texture_VK::mipLevelCount() const noexcept -> uint32_t {
		return descriptor.mipLevelCount;
	}

	auto Texture_VK::sampleCount() const noexcept -> uint32_t {
		return descriptor.sampleCount;
	}

	auto Texture_VK::dimension() const noexcept -> TextureDimension {
		return descriptor.dimension;
	}

	auto Texture_VK::format() const noexcept -> TextureFormat {
		return descriptor.format;
	}

#pragma endregion

#pragma region VK_TEXTUREVIEW_IMPL

	inline auto getVkFormat(TextureFormat format) noexcept -> VkFormat {
		switch (format)
		{
		case SIByL::RHI::TextureFormat::DEPTH32STENCIL8:	return VK_FORMAT_D32_SFLOAT_S8_UINT; break;
		case SIByL::RHI::TextureFormat::DEPTH32_FLOAT:		return VK_FORMAT_D32_SFLOAT; break;
		case SIByL::RHI::TextureFormat::DEPTH24STENCIL8:	return VK_FORMAT_D24_UNORM_S8_UINT; break;
		case SIByL::RHI::TextureFormat::DEPTH24:			return VK_FORMAT_X8_D24_UNORM_PACK32; break;
		case SIByL::RHI::TextureFormat::DEPTH16_UNORM:		return VK_FORMAT_D16_UNORM; break;
		case SIByL::RHI::TextureFormat::STENCIL8:			return VK_FORMAT_S8_UINT; break;
		case SIByL::RHI::TextureFormat::RGBA32_FLOAT:		return VK_FORMAT_R32G32B32A32_SFLOAT; break;
		case SIByL::RHI::TextureFormat::RGBA32_SINT:		return VK_FORMAT_R32G32B32A32_SINT; break;
		case SIByL::RHI::TextureFormat::RGBA32_UINT:		return VK_FORMAT_R32G32B32A32_UINT; break;
		case SIByL::RHI::TextureFormat::RGBA16_FLOAT:		return VK_FORMAT_R16G16B16A16_SFLOAT; break;
		case SIByL::RHI::TextureFormat::RGBA16_SINT:		return VK_FORMAT_R16G16B16A16_SINT; break;
		case SIByL::RHI::TextureFormat::RGBA16_UINT:		return VK_FORMAT_R16G16B16A16_UINT; break;
		case SIByL::RHI::TextureFormat::RG32_FLOAT:			return VK_FORMAT_R32G32_SFLOAT; break;
		case SIByL::RHI::TextureFormat::RG32_SINT:			return VK_FORMAT_R32G32_SINT; break;
		case SIByL::RHI::TextureFormat::RG32_UINT:			return VK_FORMAT_R32G32_UINT; break;
		case SIByL::RHI::TextureFormat::RG11B10_UFLOAT:		return VK_FORMAT_B10G11R11_UFLOAT_PACK32;  break;
		case SIByL::RHI::TextureFormat::RGB10A2_UNORM:		return VK_FORMAT_A2R10G10B10_UNORM_PACK32; break;
		case SIByL::RHI::TextureFormat::RGB9E5_UFLOAT:		return VK_FORMAT_E5B9G9R9_UFLOAT_PACK32; break;
		case SIByL::RHI::TextureFormat::BGRA8_UNORM_SRGB:	return VK_FORMAT_B8G8R8A8_SRGB; break;
		case SIByL::RHI::TextureFormat::BGRA8_UNORM:		return VK_FORMAT_B8G8R8A8_UNORM; break;
		case SIByL::RHI::TextureFormat::RGBA8_SINT:			return VK_FORMAT_B8G8R8A8_SINT; break;
		case SIByL::RHI::TextureFormat::RGBA8_UINT:			return VK_FORMAT_B8G8R8A8_UINT; break;
		case SIByL::RHI::TextureFormat::RGBA8_SNORM:		return VK_FORMAT_R8G8B8A8_SNORM; break;
		case SIByL::RHI::TextureFormat::RGBA8_UNORM_SRGB:	return VK_FORMAT_R8G8B8A8_SRGB; break;
		case SIByL::RHI::TextureFormat::RGBA8_UNORM:		return VK_FORMAT_R8G8B8A8_UNORM; break;
		case SIByL::RHI::TextureFormat::RG16_FLOAT:			return VK_FORMAT_R16G16_SFLOAT; break;
		case SIByL::RHI::TextureFormat::RG16_SINT:			return VK_FORMAT_R16G16_SINT; break;
		case SIByL::RHI::TextureFormat::RG16_UINT:			return VK_FORMAT_R16G16_UINT; break;
		case SIByL::RHI::TextureFormat::R32_FLOAT:			return VK_FORMAT_R32_SFLOAT; break;
		case SIByL::RHI::TextureFormat::R32_SINT:			return VK_FORMAT_R32_SINT; break;
		case SIByL::RHI::TextureFormat::R32_UINT:			return VK_FORMAT_R32_UINT; break;
		case SIByL::RHI::TextureFormat::RG8_SINT:			return VK_FORMAT_R8G8_SINT; break;
		case SIByL::RHI::TextureFormat::RG8_UINT:			return VK_FORMAT_R8G8_UINT; break;
		case SIByL::RHI::TextureFormat::RG8_SNORM:			return VK_FORMAT_R8G8_SNORM; break;
		case SIByL::RHI::TextureFormat::RG8_UNORM:			return VK_FORMAT_R8G8_UNORM; break;
		case SIByL::RHI::TextureFormat::R16_FLOAT:			return VK_FORMAT_R16_SFLOAT; break;
		case SIByL::RHI::TextureFormat::R16_SINT:			return VK_FORMAT_R16_SINT; break;
		case SIByL::RHI::TextureFormat::R16_UINT: 			return VK_FORMAT_R16_UINT; break;
		case SIByL::RHI::TextureFormat::R8_SINT:			return VK_FORMAT_R8_SINT; break;
		case SIByL::RHI::TextureFormat::R8_UINT:			return VK_FORMAT_R8_UINT; break;
		case SIByL::RHI::TextureFormat::R8_SNORM:			return VK_FORMAT_R8_SNORM; break;
		case SIByL::RHI::TextureFormat::R8_UNORM:			return VK_FORMAT_R8_UNORM; break;
		default: return VK_FORMAT_UNDEFINED; break;
		}
	}

	inline auto getTextureFormat(VkFormat format) noexcept -> TextureFormat {
		switch (format)
		{
		case VK_FORMAT_D32_SFLOAT_S8_UINT:		 return SIByL::RHI::TextureFormat::DEPTH32STENCIL8;
		case VK_FORMAT_D32_SFLOAT:				 return SIByL::RHI::TextureFormat::DEPTH32_FLOAT;
		case VK_FORMAT_D24_UNORM_S8_UINT:		 return SIByL::RHI::TextureFormat::DEPTH24STENCIL8;
		case VK_FORMAT_X8_D24_UNORM_PACK32:		 return SIByL::RHI::TextureFormat::DEPTH24;
		case VK_FORMAT_D16_UNORM:				 return SIByL::RHI::TextureFormat::DEPTH16_UNORM;
		case VK_FORMAT_S8_UINT:					 return SIByL::RHI::TextureFormat::STENCIL8;
		case VK_FORMAT_R32G32B32A32_SFLOAT:		 return SIByL::RHI::TextureFormat::RGBA32_FLOAT;
		case VK_FORMAT_R32G32B32A32_SINT:		 return SIByL::RHI::TextureFormat::RGBA32_SINT;
		case VK_FORMAT_R32G32B32A32_UINT:		 return SIByL::RHI::TextureFormat::RGBA32_UINT;
		case VK_FORMAT_R16G16B16A16_SFLOAT:		 return SIByL::RHI::TextureFormat::RGBA16_FLOAT;
		case VK_FORMAT_R16G16B16A16_SINT:		 return SIByL::RHI::TextureFormat::RGBA16_SINT;
		case VK_FORMAT_R16G16B16A16_UINT:		 return SIByL::RHI::TextureFormat::RGBA16_UINT;
		case VK_FORMAT_R32G32_SFLOAT:			 return SIByL::RHI::TextureFormat::RG32_FLOAT;
		case VK_FORMAT_R32G32_SINT:				 return SIByL::RHI::TextureFormat::RG32_SINT;
		case VK_FORMAT_R32G32_UINT:				 return SIByL::RHI::TextureFormat::RG32_UINT;
		case VK_FORMAT_B10G11R11_UFLOAT_PACK32:	 return SIByL::RHI::TextureFormat::RG11B10_UFLOAT;
		case VK_FORMAT_A2R10G10B10_UNORM_PACK32: return SIByL::RHI::TextureFormat::RGB10A2_UNORM;
		case VK_FORMAT_E5B9G9R9_UFLOAT_PACK32:	 return SIByL::RHI::TextureFormat::RGB9E5_UFLOAT;
		case VK_FORMAT_B8G8R8A8_SRGB:			 return SIByL::RHI::TextureFormat::BGRA8_UNORM_SRGB;
		case VK_FORMAT_B8G8R8A8_UNORM:			 return SIByL::RHI::TextureFormat::BGRA8_UNORM;
		case VK_FORMAT_B8G8R8A8_SINT:			 return SIByL::RHI::TextureFormat::RGBA8_SINT;
		case VK_FORMAT_B8G8R8A8_UINT:			 return SIByL::RHI::TextureFormat::RGBA8_UINT;
		case VK_FORMAT_R8G8B8A8_SNORM:			 return SIByL::RHI::TextureFormat::RGBA8_SNORM;
		case VK_FORMAT_R8G8B8A8_SRGB:			 return SIByL::RHI::TextureFormat::RGBA8_UNORM_SRGB;
		case VK_FORMAT_R8G8B8A8_UNORM:			 return SIByL::RHI::TextureFormat::RGBA8_UNORM;
		case VK_FORMAT_R16G16_SFLOAT:			 return SIByL::RHI::TextureFormat::RG16_FLOAT;
		case VK_FORMAT_R16G16_SINT:				 return SIByL::RHI::TextureFormat::RG16_SINT;
		case VK_FORMAT_R16G16_UINT:				 return SIByL::RHI::TextureFormat::RG16_UINT;
		case VK_FORMAT_R32_SFLOAT:				 return SIByL::RHI::TextureFormat::R32_FLOAT;
		case VK_FORMAT_R32_SINT:				 return SIByL::RHI::TextureFormat::R32_SINT;
		case VK_FORMAT_R32_UINT:				 return SIByL::RHI::TextureFormat::R32_UINT;
		case VK_FORMAT_R8G8_SINT:				 return SIByL::RHI::TextureFormat::RG8_SINT;
		case VK_FORMAT_R8G8_UINT:				 return SIByL::RHI::TextureFormat::RG8_UINT;
		case VK_FORMAT_R8G8_SNORM:				 return SIByL::RHI::TextureFormat::RG8_SNORM;
		case VK_FORMAT_R8G8_UNORM:				 return SIByL::RHI::TextureFormat::RG8_UNORM;
		case VK_FORMAT_R16_SFLOAT:				 return SIByL::RHI::TextureFormat::R16_FLOAT;
		case VK_FORMAT_R16_SINT:				 return SIByL::RHI::TextureFormat::R16_SINT;
		case VK_FORMAT_R16_UINT:				 return SIByL::RHI::TextureFormat::R16_UINT;
		case VK_FORMAT_R8_SINT:					 return SIByL::RHI::TextureFormat::R8_SINT;
		case VK_FORMAT_R8_UINT:					 return SIByL::RHI::TextureFormat::R8_UINT;
		case VK_FORMAT_R8_SNORM:				 return SIByL::RHI::TextureFormat::R8_SNORM;
		case VK_FORMAT_R8_UNORM:				 return SIByL::RHI::TextureFormat::R8_UNORM;
		default: return SIByL::RHI::TextureFormat(0); break;
		}
	}

	inline auto getVkImageViewType(TextureViewDimension const& dim) noexcept -> VkImageViewType {
		switch (dim)
		{
		case TextureViewDimension::TEX1D:		return VkImageViewType::VK_IMAGE_VIEW_TYPE_1D; break;
		case TextureViewDimension::TEX2D:		return VkImageViewType::VK_IMAGE_VIEW_TYPE_2D; break;
		case TextureViewDimension::TEX2D_ARRAY:	return VkImageViewType::VK_IMAGE_VIEW_TYPE_2D_ARRAY; break;
		case TextureViewDimension::CUBE:		return VkImageViewType::VK_IMAGE_VIEW_TYPE_CUBE; break;
		case TextureViewDimension::CUBE_ARRAY:	return VkImageViewType::VK_IMAGE_VIEW_TYPE_CUBE_ARRAY; break;
		case TextureViewDimension::TEX3D:		return VkImageViewType::VK_IMAGE_VIEW_TYPE_3D; break;
		default:
			break;
		}
	}

	inline auto getVkImageAspectFlags(TextureAspectFlags aspect) noexcept -> VkImageAspectFlags {
		VkImageAspectFlags ret = 0;
		if (aspect & (uint32_t)TextureAspect::COLOR_BIT) ret |= VK_IMAGE_ASPECT_COLOR_BIT;
		if (aspect & (uint32_t)TextureAspect::DEPTH_BIT) ret |= VK_IMAGE_ASPECT_DEPTH_BIT;
		if (aspect & (uint32_t)TextureAspect::STENCIL_BIT) ret |= VK_IMAGE_ASPECT_STENCIL_BIT;
		return ret;
	}

	TextureView_VK::TextureView_VK(Device_VK* device, Texture_VK* texture, TextureViewDescriptor const& descriptor)
		: device(device), texture(texture), descriptor(descriptor)
	{
		VkImageViewCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		createInfo.image = texture->getVkImage();
		createInfo.viewType = getVkImageViewType(descriptor.dimension);
		createInfo.format = getVkFormat(descriptor.format);
		createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
		createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
		createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
		createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
		createInfo.subresourceRange.aspectMask = getVkImageAspectFlags(descriptor.aspect);
		createInfo.subresourceRange.baseMipLevel = descriptor.baseMipLevel;
		createInfo.subresourceRange.levelCount = descriptor.mipLevelCount;
		createInfo.subresourceRange.baseArrayLayer = descriptor.baseArrayLayer;
		createInfo.subresourceRange.layerCount = descriptor.arrayLayerCount;
		if (vkCreateImageView(device->getVkDevice(), &createInfo, nullptr, &imageView) != VK_SUCCESS) {
			Core::LogManager::Error("VULKAN :: failed to create image views!");
		}
	}

	TextureView_VK::TextureView_VK(TextureView_VK&& view)
		: imageView(view.imageView), descriptor(view.descriptor)
		, texture(view.texture), device(view.device)
	{
		view.imageView = nullptr;
	}

	auto TextureView_VK::operator=(TextureView_VK&& view) ->TextureView_VK& {
		imageView = view.imageView;
		descriptor = view.descriptor;
		texture = view.texture;
		device = view.device;
		view.imageView = nullptr; return *this;
	}

	TextureView_VK::~TextureView_VK() {
		if (imageView) vkDestroyImageView(device->getVkDevice(), imageView, nullptr);
	}

#pragma endregion

	export struct ExternalTexture_VK :public ExternalTexture {
		/** virtual destructor */
		virtual ~ExternalTexture_VK();
		/** indicates whether the texture has expired or not */
		virtual auto expired() const noexcept -> bool override;
	private:
		/** vulkan image */
		VkImage image;
		/** vulkan image device memory */
		VkDeviceMemory deviceMemory;
		/** Texture Descriptor */
		TextureDescriptor descriptor;
		/** the device this texture is created on */
		Device_VK* device = nullptr;
	};

	// Textures/TextureViews Interface
	// ===========================================================================
	// Samplers Interface

	export struct Sampler_VK :public Sampler {
		/** vulkan Texture Sampler */
		VkSampler textureSampler;
		/** the device this sampler is created on */
		Device_VK* device = nullptr;
	};

	// Samplers Interface
	// ===========================================================================
	// SwapChain Interface

	struct SwapChain_VK :public SwapChain {
		/** virtual destructor */
		virtual ~SwapChain_VK();
		/** intialize the swapchin */
		auto init(Device_VK* device, SwapChainDescriptor const& desc) noexcept -> void;
		/** get texture view */
		virtual auto getTextureView(int i) noexcept -> TextureView* override { return &textureViews[i]; }
		/** invalid swapchain */
		virtual auto recreate() noexcept -> void override;
		/** vulkan SwapChain */
		VkSwapchainKHR swapChain;
		/** vulkan SwapChain Extent */
		VkExtent2D swapChainExtend;
		/** vulkan SwapChain format */
		VkFormat swapChainImageFormat;
		/** vulkan SwapChain fetched images */
		std::vector<Texture_VK> swapChainTextures;
		/** vulkan SwapChain fetched images views */
		std::vector<TextureView_VK> textureViews;
		/** the device this sampler is created on */
		Device_VK* device = nullptr;
	};

#pragma region VK_SWAPCHAIN_IMPL

	inline auto chooseSwapSurfaceFormat(std::vector<VkSurfaceFormatKHR> const& availableFormats) noexcept -> VkSurfaceFormatKHR {
		for (auto const& availableFormat : availableFormats) {
			if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && 
				availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
				return availableFormat;
			}
		}
		return availableFormats[0];
	}
	
	inline auto chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) noexcept -> VkPresentModeKHR {
		for (const auto& availablePresentMode : availablePresentModes) {
			if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
				return availablePresentMode;
			}
		}
		return VK_PRESENT_MODE_FIFO_KHR;
	}

	inline auto chooseSwapExtent(
		VkSurfaceCapabilitiesKHR const& capabilities, 
		Platform::Window* bindedWindow) noexcept -> VkExtent2D 
	{
		if (capabilities.currentExtent.width != Math::uint32_max)
			return capabilities.currentExtent;
		else {
			int width, height;
			bindedWindow->getFramebufferSize(&width, &height);
			VkExtent2D actualExtent = {
				static_cast<uint32_t>(width),
				static_cast<uint32_t>(height)
			};
			actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
			actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
			return actualExtent;
		}
	}

	void createSwapChain(Device_VK* device, SwapChain_VK* swapchain) {
		Adapter_VK* adapater = device->getAdapterVk();
		SwapChainSupportDetails swapChainSupport = querySwapChainSupport(adapater->getContext(), adapater->getVkPhysicalDevice());
		VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
		VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
		VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities, adapater->getContext()->getBindedWindow());
		swapchain->swapChainExtend = extent;
		swapchain->swapChainImageFormat = surfaceFormat.format;
		uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
		if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
			imageCount = swapChainSupport.capabilities.maxImageCount;
		}
		VkSwapchainCreateInfoKHR createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
		createInfo.surface = adapater->getContext()->getVkSurfaceKHR();
		createInfo.minImageCount = imageCount;
		createInfo.imageFormat = surfaceFormat.format;
		createInfo.imageColorSpace = surfaceFormat.colorSpace;
		createInfo.imageExtent = extent;
		createInfo.imageArrayLayers = 1;
		createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
		QueueFamilyIndices_VK const& indices = adapater->getQueueFamilyIndices();
		uint32_t queueFamilyIndices[] = { indices.graphicsFamily.value(), indices.presentFamily.value() };
		if (indices.graphicsFamily != indices.presentFamily) {
			createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
			createInfo.queueFamilyIndexCount = 2;
			createInfo.pQueueFamilyIndices = queueFamilyIndices;
		}
		else {
			createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
			createInfo.queueFamilyIndexCount = 0; // Optional
			createInfo.pQueueFamilyIndices = nullptr; // Optional
		}
		createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
		createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
		createInfo.presentMode = presentMode;
		createInfo.clipped = VK_TRUE;
		createInfo.oldSwapchain = VK_NULL_HANDLE;
		if (vkCreateSwapchainKHR(device->getVkDevice(), &createInfo, nullptr, &swapchain->swapChain) != VK_SUCCESS) {
			Core::LogManager::Error("VULKAN :: failed to create swap chain!");
		}
	}
	
	SwapChain_VK::~SwapChain_VK() {
		if (swapChain)
			vkDestroySwapchainKHR(device->getVkDevice(), swapChain, nullptr);
	}

	auto SwapChain_VK::init(Device_VK* device, SwapChainDescriptor const& desc) noexcept -> void {
		this->device = device;
		recreate();
	}
	
	auto SwapChain_VK::recreate() noexcept -> void {
		device->waitIdle();
		// clean up swap chain
		swapChainTextures.clear();
		textureViews.clear();
		if(swapChain) vkDestroySwapchainKHR(device->getVkDevice(), swapChain, nullptr);
		// recreate swap chain
		createSwapChain(device, this);
		// retrieving the swap chian image
		uint32_t imageCount = 0;
		vkGetSwapchainImagesKHR(device->getVkDevice(), swapChain, &imageCount, nullptr);
		std::vector<VkImage> swapChainImages;
		swapChainImages.resize(imageCount);
		vkGetSwapchainImagesKHR(device->getVkDevice(), swapChain, &imageCount, swapChainImages.data());
		// create Image views for images
		TextureDescriptor textureDesc;
		textureDesc.dimension = TextureDimension::TEX2D;
		textureDesc.format = getTextureFormat(swapChainImageFormat);
		textureDesc.size = { swapChainExtend.width, swapChainExtend.height };
		textureDesc.usage = 0;
		TextureViewDescriptor viewDesc;
		viewDesc.format = getTextureFormat(swapChainImageFormat);
		viewDesc.aspect = (uint32_t)TextureAspect::COLOR_BIT;
		for (size_t i = 0; i < swapChainImages.size(); i++)
			swapChainTextures.push_back(Texture_VK{ device, swapChainImages[i], textureDesc });
		for (size_t i = 0; i < swapChainImages.size(); i++)
			textureViews.push_back(TextureView_VK{ device, &swapChainTextures[i], viewDesc });
	}

	auto Device_VK::createSwapChain(SwapChainDescriptor const& desc) noexcept -> std::unique_ptr<SwapChain> {
		std::unique_ptr<SwapChain_VK> swapChain = std::make_unique<SwapChain_VK>();
		swapChain->init(this, desc);
		return std::move(swapChain);
	}

#pragma endregion

	// SwapChain Interface
	// ===========================================================================
	// Resource Binding Interface

	export struct BindGroupLayout_VK :public BindGroupLayout {
		/** contructor */
		BindGroupLayout_VK(Device_VK* device, BindGroupLayoutDescriptor const& desc);
		/** destructor */
		~BindGroupLayout_VK();
		/** vulkan Descriptor Set Layout */
		VkDescriptorSetLayout layout;
		/** Bind Group Layout Descriptor */
		BindGroupLayoutDescriptor descriptor;
		/** the device this bind group layout is created on */
		Device_VK* device = nullptr;
	};

#pragma region VK_BINDGROUPLAYOUT_IMPL

	inline auto getVkDecriptorType(BindGroupLayoutEntry const& entry) -> VkDescriptorType {
		if (entry.buffer.has_value()) {
			switch (entry.buffer.value().type)
			{
			case BufferBindingType::UNIFORM: return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
			case BufferBindingType::STORAGE: return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			case BufferBindingType::READ_ONLY_STORAGE: return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;	
			default: return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			}
		}
		if (entry.sampler.has_value()) return VK_DESCRIPTOR_TYPE_SAMPLER;
		if (entry.texture.has_value()) return VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
		if (entry.storageTexture.has_value()) return VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		if (entry.externalTexture.has_value()) return VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
	}

	inline auto getVkShaderStageFlags(ShaderStagesFlags flags) noexcept ->  VkShaderStageFlags {
		VkShaderStageFlags ret = 0;
		if (flags & (uint32_t)ShaderStages::VERTEX) ret |= VkShaderStageFlagBits::VK_SHADER_STAGE_VERTEX_BIT;
		if (flags & (uint32_t)ShaderStages::FRAGMENT) ret |= VkShaderStageFlagBits::VK_SHADER_STAGE_FRAGMENT_BIT;
		if (flags & (uint32_t)ShaderStages::COMPUTE) ret |= VkShaderStageFlagBits::VK_SHADER_STAGE_COMPUTE_BIT;
		if (flags & (uint32_t)ShaderStages::RAYGEN) ret |= VkShaderStageFlagBits::VK_SHADER_STAGE_RAYGEN_BIT_NV;
		if (flags & (uint32_t)ShaderStages::MISS) ret |= VkShaderStageFlagBits::VK_SHADER_STAGE_MISS_BIT_NV;
		if (flags & (uint32_t)ShaderStages::CLOSEST_HIT) ret |= VkShaderStageFlagBits::VK_SHADER_STAGE_CLOSEST_HIT_BIT_NV;
		if (flags & (uint32_t)ShaderStages::INTERSECTION) ret |= VkShaderStageFlagBits::VK_SHADER_STAGE_INTERSECTION_BIT_NV;
		if (flags & (uint32_t)ShaderStages::ANY_HIT) ret |= VkShaderStageFlagBits::VK_SHADER_STAGE_ANY_HIT_BIT_NV;
		return ret;
	}

	BindGroupLayout_VK::BindGroupLayout_VK(Device_VK* device, BindGroupLayoutDescriptor const& desc)
		:device(device), descriptor(desc) 
	{
		std::vector<VkDescriptorSetLayoutBinding> bindings(desc.entries.size());
		for (int i = 0; i < desc.entries.size(); i++) {
			bindings[i].binding = desc.entries[i].binding;
			bindings[i].descriptorType = getVkDecriptorType(desc.entries[i]);
			bindings[i].descriptorCount = 1;
			bindings[i].stageFlags = getVkShaderStageFlags(desc.entries[i].visibility);
			bindings[i].pImmutableSamplers = nullptr;
		}
		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = bindings.size();
		layoutInfo.pBindings = bindings.data();
		if (vkCreateDescriptorSetLayout(device->getVkDevice(), &layoutInfo, nullptr, &layout) != VK_SUCCESS) {
			Core::LogManager::Error("VULKAN :: failed to create descriptor set layout!");
		}
	}

	BindGroupLayout_VK::~BindGroupLayout_VK() {
		if (layout) vkDestroyDescriptorSetLayout(device->getVkDevice(), layout, nullptr);
	}

	auto Device_VK::createBindGroupLayout(BindGroupLayoutDescriptor const& desc) noexcept -> std::unique_ptr<BindGroupLayout> {
		return std::make_unique<BindGroupLayout_VK>(this, desc);
	}

#pragma endregion

	export struct BindGroupPool_VK {
		/** initialzier */
		BindGroupPool_VK(Device_VK* device);
		/** destructor */
		~BindGroupPool_VK();
		/** vulkan Bind Group Pool */
		VkDescriptorPool  descriptorPool;
		/** the device this bind group pool is created on */
		Device_VK* device = nullptr;
	};

#pragma region VK_BINDGROUPPOOL_IMPL

	BindGroupPool_VK::BindGroupPool_VK(Device_VK* device)
		: device(device)
	{
		VkDescriptorPoolSize poolSize{};
		poolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		poolSize.descriptorCount = static_cast<uint32_t>(999);
		VkDescriptorPoolCreateInfo poolInfo{};
		poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		poolInfo.poolSizeCount = 1;
		poolInfo.pPoolSizes = &poolSize;
		poolInfo.maxSets = static_cast<uint32_t>(999);
		if (vkCreateDescriptorPool(device->getVkDevice(), &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
			Core::LogManager::Error("VULKAN :: failed to create descriptor pool!");
		}
	}

	BindGroupPool_VK::~BindGroupPool_VK() {
		if(descriptorPool) vkDestroyDescriptorPool(device->getVkDevice(), descriptorPool, nullptr);
	}

	auto Device_VK::createBindGroupPool() noexcept -> void {
		bindGroupPool = std::make_unique<BindGroupPool_VK>(this);
	}

#pragma endregion

	export struct BindGroup_VK :public BindGroup {
		/** initialzie */
		BindGroup_VK(Device_VK* device, BindGroupDescriptor const& desc);
		/** vulkan Descriptor Set */
		VkDescriptorSet set;
		/** the bind group set this bind group is created on */
		BindGroupPool_VK* descriptorPool;
		/** the device this bind group is created on */
		Device_VK* device = nullptr;
	};

#pragma region VK_BINDGROUP_IMPL

	BindGroup_VK::BindGroup_VK(Device_VK* device, BindGroupDescriptor const& desc) {
		VkDescriptorSetAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocInfo.descriptorPool = device->getBindGroupPool()->descriptorPool;
		allocInfo.descriptorSetCount = 1;
		allocInfo.pSetLayouts = &static_cast<BindGroupLayout_VK*>(desc.layout)->layout;
		if (vkAllocateDescriptorSets(device->getVkDevice(), &allocInfo, &set) != VK_SUCCESS) {
			Core::LogManager::Error("VULKAN :: failed to allocate descriptor sets!");
		}
		// configure the descriptors
		for (auto& entry : desc.entries) {
			if (entry.resource.bufferBinding.has_value()) {
				VkDescriptorBufferInfo bufferInfo{};
				bufferInfo.buffer = static_cast<Buffer_VK*>(entry.resource.bufferBinding.value().buffer)->getVkBuffer();
				bufferInfo.offset = entry.resource.bufferBinding.value().offset;
				bufferInfo.range = entry.resource.bufferBinding.value().size;
				VkWriteDescriptorSet descriptorWrite{};
				descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
				descriptorWrite.dstSet = set;
				descriptorWrite.dstBinding = entry.binding;
				descriptorWrite.dstArrayElement = 0;
				descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
				descriptorWrite.descriptorCount = 1;
				descriptorWrite.pBufferInfo = &bufferInfo;
				descriptorWrite.pImageInfo = nullptr;
				descriptorWrite.pTexelBufferView = nullptr;
				vkUpdateDescriptorSets(device->getVkDevice(), 1, &descriptorWrite, 0, nullptr);
			}
		}
	}

	auto Device_VK::createBindGroup(BindGroupDescriptor const& desc) noexcept -> std::unique_ptr<BindGroup> {
		return std::make_unique<BindGroup_VK>(this, desc);
	}

#pragma endregion


	export struct PipelineLayout_VK :public PipelineLayout {
		/** intializer */
		PipelineLayout_VK(Device_VK* device, PipelineLayoutDescriptor const& desc);
		/** virtual destructor */
		virtual ~PipelineLayout_VK();
		/** copy functions */
		PipelineLayout_VK(PipelineLayout_VK const& layout) = delete;
		PipelineLayout_VK(PipelineLayout_VK&& layout);
		auto operator=(PipelineLayout_VK const& layout) -> PipelineLayout_VK & = delete;
		auto operator=(PipelineLayout_VK&& layout) -> PipelineLayout_VK&;
		/** vulkan pipeline layout */
		VkPipelineLayout pipelineLayout;
		/** the push constans on pipeline layouts */
		std::vector<VkPushConstantRange> pushConstants;
		/** the device this pipeline layout is created on */
		Device_VK* device = nullptr;
	};

#pragma region VK_PIPELINELAYOUT_IMPL

	PipelineLayout_VK::PipelineLayout_VK(Device_VK* device, PipelineLayoutDescriptor const& desc)
		: device(device) 
	{
		std::vector<VkDescriptorSetLayout> descriptorSets;
		for (auto& bindgroupLayout : desc.bindGroupLayouts) {
			descriptorSets.push_back(static_cast<BindGroupLayout_VK*>(bindgroupLayout)->layout);
		}
		VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
		pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutInfo.setLayoutCount = descriptorSets.size();
		pipelineLayoutInfo.pSetLayouts = descriptorSets.data();
		pipelineLayoutInfo.pushConstantRangeCount = 0; // Optional
		pipelineLayoutInfo.pPushConstantRanges = nullptr; // Optional
		if (vkCreatePipelineLayout(device->getVkDevice(), &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
			Core::LogManager::Error("failed to create pipeline layout!");
		}
	}

	PipelineLayout_VK::~PipelineLayout_VK() {
		if (pipelineLayout) vkDestroyPipelineLayout(device->getVkDevice(), pipelineLayout, nullptr);
	}

	PipelineLayout_VK::PipelineLayout_VK(PipelineLayout_VK&& layout)
		: pipelineLayout(layout.pipelineLayout)
		, pushConstants(layout.pushConstants)
		, device(layout.device) {
		layout.pipelineLayout = nullptr;
	}

	auto PipelineLayout_VK::operator=(PipelineLayout_VK&& layout) -> PipelineLayout_VK& {
		pipelineLayout = layout.pipelineLayout;
		pushConstants = layout.pushConstants;
		device = layout.device;
		layout.pipelineLayout = nullptr;
		return *this;
	}

	auto Device_VK::createPipelineLayout(PipelineLayoutDescriptor const& desc) noexcept -> std::unique_ptr<PipelineLayout> {
		return std::make_unique<PipelineLayout_VK>(this, desc);
	}

#pragma endregion


	// Resource Binding Interface
	// ===========================================================================
	// Shader Modules Interface

	export struct ShaderModule_VK :public ShaderModule {
		/** initalize shader module */
		ShaderModule_VK(Device_VK* device, ShaderModuleDescriptor const& desc);
		/** virtual descructor */
		~ShaderModule_VK();
		/** copy functions */
		ShaderModule_VK(ShaderModule_VK const& shader) = delete;
		ShaderModule_VK(ShaderModule_VK&& shader);
		auto operator=(ShaderModule_VK const& shader) -> ShaderModule_VK & = delete;
		auto operator=(ShaderModule_VK&& shader) -> ShaderModule_VK&;
		/** the shader stages included in this module */
		ShaderStagesFlags stages;
		/** vulkan shader module */
		VkShaderModule shaderModule = {};
		/** vulkan shader stage create info */
		VkPipelineShaderStageCreateInfo shaderStageInfo{};
		/** the device this shader module is created on */
		Device_VK* device = nullptr;
	};

#pragma region VK_SHADERMODULE_IMPL

	inline auto getVkShaderStageFlagBits(ShaderStages flag) noexcept -> VkShaderStageFlagBits {
		switch (flag)
		{
		case SIByL::RHI::ShaderStages::COMPUTE:		return VkShaderStageFlagBits::VK_SHADER_STAGE_COMPUTE_BIT; break;
		case SIByL::RHI::ShaderStages::FRAGMENT:	return VkShaderStageFlagBits::VK_SHADER_STAGE_FRAGMENT_BIT; break;
		case SIByL::RHI::ShaderStages::VERTEX:		return VkShaderStageFlagBits::VK_SHADER_STAGE_VERTEX_BIT; break;
		default: return VkShaderStageFlagBits::VK_SHADER_STAGE_ALL; break;
			break;
		}
	}

	ShaderModule_VK::ShaderModule_VK(Device_VK* device, ShaderModuleDescriptor const& desc)
		: device(device), stages((uint32_t)desc.stage) {
		VkShaderModuleCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		createInfo.codeSize = desc.code->size;
		createInfo.pCode = reinterpret_cast<const uint32_t*>(desc.code->data);
		if (vkCreateShaderModule(device->getVkDevice(), &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
			Core::LogManager::Error("VULKAN :: failed to create shader module!");
		}
		// create info
		shaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		shaderStageInfo.stage = getVkShaderStageFlagBits(desc.stage);
		shaderStageInfo.module = shaderModule;
		shaderStageInfo.pName = "main";
	}

	ShaderModule_VK::ShaderModule_VK(ShaderModule_VK&& shader)
		: device(shader.device), stages(shader.stages), shaderModule(shader.shaderModule)
		, shaderStageInfo(shader.shaderStageInfo) {
		shader.shaderModule = nullptr;
	}

	auto ShaderModule_VK::operator=(ShaderModule_VK&& shader) -> ShaderModule_VK& {
		device = shader.device;
		stages = shader.stages;
		shaderModule = shader.shaderModule;
		shaderStageInfo = shader.shaderStageInfo;
		shader.shaderModule = nullptr;
		return *this;
	}

	ShaderModule_VK::~ShaderModule_VK() {
		if (shaderModule) vkDestroyShaderModule(device->getVkDevice(), shaderModule, nullptr);
	}

	auto Device_VK::createShaderModule(ShaderModuleDescriptor const& desc) noexcept -> std::unique_ptr<ShaderModule> {
		std::unique_ptr<ShaderModule_VK> shadermodule = std::make_unique<ShaderModule_VK>(this, desc);
		return shadermodule;
	}

#pragma endregion

	// Shader Modules Interface
	// ===========================================================================
	// Pipelines Interface

	export struct ComputePipeline_VK :public ComputePipeline {
		/** vulkan compute pipeline */
		VkPipeline pipeline;
		/** the device this compute pipeline is created on */
		Device_VK* device = nullptr;
	};

	export struct RenderPass_VK {
		/** render pass initialize */
		RenderPass_VK(Device_VK* device, RenderPassDescriptor const& desc);
		/** virtual destructor */
		virtual ~RenderPass_VK();
		/** copy functions */
		RenderPass_VK(RenderPass_VK const& pass) = delete;
		RenderPass_VK(RenderPass_VK&& pass);
		auto operator=(RenderPass_VK const& pass) -> RenderPass_VK & = delete;
		auto operator=(RenderPass_VK&& pass) -> RenderPass_VK&;
		/** vulkan render pass */
		VkRenderPass renderPass;
		/** vulkan render pass clear value */
		std::vector<VkClearValue> clearValues;
		/** the device this render pass is created on */
		Device_VK* device = nullptr;
	};

#pragma region VK_RENDERPASS_IMPL

	inline auto getVkAttachmentLoadOp(LoadOp op) noexcept -> VkAttachmentLoadOp {
		switch (op)
		{
		case SIByL::RHI::LoadOp::DONT_CARE: return VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		case SIByL::RHI::LoadOp::CLEAR: return VK_ATTACHMENT_LOAD_OP_CLEAR;
		case SIByL::RHI::LoadOp::LOAD: return VK_ATTACHMENT_LOAD_OP_LOAD;
		default: return VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		}
	}

	inline auto getVkAttachmentStoreOp(StoreOp op) noexcept -> VkAttachmentStoreOp {
		switch (op)
		{
		case SIByL::RHI::StoreOp::DISCARD: return VK_ATTACHMENT_STORE_OP_DONT_CARE;
		case SIByL::RHI::StoreOp::STORE: return VK_ATTACHMENT_STORE_OP_STORE;
		default: return VK_ATTACHMENT_STORE_OP_DONT_CARE;
		}
	}

	RenderPass_VK::RenderPass_VK(Device_VK* device, RenderPassDescriptor const& desc)
		: device(device) 
	{
		std::vector<VkAttachmentDescription> attachments;
		for (auto const& colorAttach : desc.colorAttachments) {
			VkAttachmentDescription colorAttachment{};
			colorAttachment.format = getVkFormat(static_cast<TextureView_VK*>(colorAttach.view)->descriptor.format);
			colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
			colorAttachment.loadOp = getVkAttachmentLoadOp(colorAttach.loadOp);
			colorAttachment.storeOp = getVkAttachmentStoreOp(colorAttach.storeOp);
			colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
			colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
			colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
			attachments.emplace_back(colorAttachment);
		}
		VkAttachmentReference colorAttachmentRef{};
		colorAttachmentRef.attachment = 0;
		colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		VkSubpassDescription subpass{};
		subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpass.colorAttachmentCount = 1;
		subpass.pColorAttachments = &colorAttachmentRef;
		VkRenderPassCreateInfo renderPassInfo{};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		renderPassInfo.attachmentCount = 1;
		renderPassInfo.pAttachments = attachments.data();
		renderPassInfo.subpassCount = 1;
		renderPassInfo.pSubpasses = &subpass;
		if (vkCreateRenderPass(device->getVkDevice(), &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
			Core::LogManager::Error("VULKAN :: failed to create render pass!");
		}
	}

	RenderPass_VK::~RenderPass_VK() {
		if (renderPass) vkDestroyRenderPass(device->getVkDevice(), renderPass, nullptr);
	}

	RenderPass_VK::RenderPass_VK(RenderPass_VK&& pass)
		: device(pass.device), renderPass(pass.renderPass), clearValues(pass.clearValues) {
		pass.renderPass = nullptr;
	}

	auto RenderPass_VK::operator=(RenderPass_VK&& pass) -> RenderPass_VK& {
		device = pass.device;
		renderPass = pass.renderPass;
		clearValues = pass.clearValues;
		pass.renderPass = nullptr;
		return *this;
	}

#pragma endregion


	export struct RenderPipeline_VK :public RenderPipeline {
		/** constructor */
		RenderPipeline_VK(Device_VK* device, RenderPipelineDescriptor const& desc);
		/** virtual destructor */
		virtual ~RenderPipeline_VK();
		/** copy functions */
		RenderPipeline_VK(RenderPipeline_VK const& pipeline) = delete;
		RenderPipeline_VK(RenderPipeline_VK&& pipeline);
		auto operator=(RenderPipeline_VK const& pipeline) -> RenderPipeline_VK & = delete;
		auto operator=(RenderPipeline_VK&& pipeline) -> RenderPipeline_VK&;
		/** vulkan render pipeline */
		VkPipeline pipeline = {};
		/** vulkan render pipeline fixed function settings */
		struct RenderPipelineFixedFunctionSettings {
			// shader stages
			std::vector<VkPipelineShaderStageCreateInfo> shaderStages = {};
			// dynamic state
			VkPipelineDynamicStateCreateInfo dynamicState = {};
			std::vector<VkDynamicState>		 dynamicStates = {};
			// vertex layout
			VkPipelineVertexInputStateCreateInfo		   vertexInputState = {};
			std::vector<VkVertexInputBindingDescription>   vertexBindingDescriptor = {};
			std::vector<VkVertexInputAttributeDescription> vertexAttributeDescriptions = {};
			// input assembly
			VkPipelineInputAssemblyStateCreateInfo	assemblyState = {};
			// viewport settings
			VkViewport viewport = {}; VkRect2D scissor = {};
			VkPipelineViewportStateCreateInfo 		viewportState = {};
			// multisample
			VkPipelineMultisampleStateCreateInfo	multisampleState = {};
			VkPipelineRasterizationStateCreateInfo	rasterizationState = {};
			VkPipelineDepthStencilStateCreateInfo	depthStencilState = {};
			std::vector<VkPipelineColorBlendAttachmentState> colorBlendAttachmentStates = {};
			VkPipelineColorBlendStateCreateInfo		colorBlendState = {};
			PipelineLayout*							pipelineLayout = {};
		} fixedFunctionSetttings;
		/** the reusable create information of the pipeline */
		VkGraphicsPipelineCreateInfo pipelineInfo{};
		/** combine the pipelien with a render pass and then re-valid it */
		auto combineRenderPass(RenderPass_VK* renderpass) noexcept -> void;
		/** the device this render pipeline is created on */
		Device_VK* device = nullptr;
	};

#pragma region VK_RENDERPIPELINE_IMPL

	inline auto getVkPrimitiveTopology(PrimitiveTopology topology) noexcept -> VkPrimitiveTopology {
		switch (topology)
		{
		case SIByL::RHI::PrimitiveTopology::TRIANGLE_STRIP: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP; break;
		case SIByL::RHI::PrimitiveTopology::TRIANGLE_LIST:	return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST; break;
		case SIByL::RHI::PrimitiveTopology::LINE_STRIP:		return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP; break;
		case SIByL::RHI::PrimitiveTopology::LINE_LIST:		return VK_PRIMITIVE_TOPOLOGY_LINE_LIST; break;
		case SIByL::RHI::PrimitiveTopology::POINT_LIST:		return VK_PRIMITIVE_TOPOLOGY_POINT_LIST; break;
		default: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP; break;
		}
	}

	inline auto getVkPipelineInputAssemblyStateCreateInfo(PrimitiveTopology topology) noexcept -> VkPipelineInputAssemblyStateCreateInfo {
		VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
		inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		inputAssembly.topology = getVkPrimitiveTopology(topology);
		inputAssembly.primitiveRestartEnable = VK_FALSE;
		return inputAssembly;
	}

	inline auto getVkCullModeFlagBits(CullMode cullmode) noexcept -> VkCullModeFlagBits {
		switch (cullmode)
		{
		case SIByL::RHI::CullMode::BACK: return VkCullModeFlagBits::VK_CULL_MODE_BACK_BIT;
		case SIByL::RHI::CullMode::FRONT: return VkCullModeFlagBits::VK_CULL_MODE_FRONT_BIT;
		case SIByL::RHI::CullMode::NONE: return VkCullModeFlagBits::VK_CULL_MODE_NONE;
		case SIByL::RHI::CullMode::BOTH: return VkCullModeFlagBits::VK_CULL_MODE_FRONT_AND_BACK;
		default: return VkCullModeFlagBits::VK_CULL_MODE_NONE;
		}
	}

	inline auto getVkFrontFace(FrontFace ff) noexcept -> VkFrontFace {
		switch (ff)
		{
		case SIByL::RHI::FrontFace::CW: return VkFrontFace::VK_FRONT_FACE_CLOCKWISE;
		case SIByL::RHI::FrontFace::CCW: return VkFrontFace::VK_FRONT_FACE_COUNTER_CLOCKWISE;
		default: return VkFrontFace::VK_FRONT_FACE_CLOCKWISE;
		}
	}

	inline auto getVkPipelineRasterizationStateCreateInfo(DepthStencilState const& dsstate, FragmentState const& fstate,
		PrimitiveState const& pstate) noexcept -> VkPipelineRasterizationStateCreateInfo {
		VkPipelineRasterizationStateCreateInfo rasterizer{};
		rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterizer.depthClampEnable = VK_FALSE;
		rasterizer.rasterizerDiscardEnable = VK_FALSE;
		rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
		rasterizer.lineWidth = 1.0f;
		rasterizer.cullMode = getVkCullModeFlagBits(pstate.cullMode);
		rasterizer.frontFace = getVkFrontFace(pstate.frontFace);
		rasterizer.depthBiasEnable = (dsstate.depthBias == 0) ? VK_FALSE : VK_TRUE;
		rasterizer.depthBiasConstantFactor = dsstate.depthBias;
		rasterizer.depthBiasClamp = dsstate.depthBiasClamp;
		rasterizer.depthBiasSlopeFactor = dsstate.depthBiasSlopeScale;
		return rasterizer;
	}

	inline auto getVkPipelineViewportStateCreateInfo(VkViewport& viewport, VkRect2D& scissor) noexcept -> VkPipelineViewportStateCreateInfo {
		viewport.x = 0.0f;
		viewport.y = 0.0f;
		viewport.width = (float)720;
		viewport.height = (float)480;
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;
		scissor.offset = { 0, 0 };
		scissor.extent = { 720,480 };
		VkPipelineViewportStateCreateInfo viewportState{};
		viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportState.viewportCount = 1;
		viewportState.pViewports = &viewport;
		viewportState.scissorCount = 1;
		viewportState.pScissors = &scissor;
		return viewportState;
	}

	inline auto getVkPipelineMultisampleStateCreateInfo(MultisampleState const& state) noexcept -> VkPipelineMultisampleStateCreateInfo {
		VkPipelineMultisampleStateCreateInfo multisampling = {};
		multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multisampling.sampleShadingEnable = VK_FALSE;
		multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
		multisampling.minSampleShading = 1.0f; // Optional
		multisampling.pSampleMask = nullptr; // Optional
		multisampling.alphaToCoverageEnable = state.alphaToCoverageEnabled ? VK_TRUE : VK_FALSE;
		multisampling.alphaToOneEnable = VK_FALSE; // Optional
		return multisampling;
	}

	inline auto getVkCompareOp(CompareFunction compare) noexcept -> VkCompareOp {
		switch (compare)
		{
		case SIByL::RHI::CompareFunction::ALWAYS:			return VK_COMPARE_OP_ALWAYS; break;
		case SIByL::RHI::CompareFunction::GREATER_EQUAL:	return VK_COMPARE_OP_GREATER_OR_EQUAL; break;
		case SIByL::RHI::CompareFunction::NOT_EQUAL:		return VK_COMPARE_OP_NOT_EQUAL; break;
		case SIByL::RHI::CompareFunction::GREATER:			return VK_COMPARE_OP_GREATER; break;
		case SIByL::RHI::CompareFunction::LESS_EQUAL:		return VK_COMPARE_OP_LESS_OR_EQUAL; break;
		case SIByL::RHI::CompareFunction::EQUAL:			return VK_COMPARE_OP_EQUAL;  break;
		case SIByL::RHI::CompareFunction::LESS:				return VK_COMPARE_OP_LESS; break;
		case SIByL::RHI::CompareFunction::NEVER:			return VK_COMPARE_OP_NEVER; break;
		default: return VK_COMPARE_OP_ALWAYS; break;
		}
	}

	inline auto getVkPipelineDepthStencilStateCreateInfo(DepthStencilState const& state) noexcept -> VkPipelineDepthStencilStateCreateInfo {
		VkPipelineDepthStencilStateCreateInfo depthStencil = {};
		depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthStencil.depthTestEnable = state.depthCompare != CompareFunction::ALWAYS ? VK_TRUE : VK_FALSE;
		depthStencil.depthWriteEnable = state.depthWriteEnabled ? VK_TRUE : VK_FALSE;
		depthStencil.depthCompareOp = getVkCompareOp(state.depthCompare);
		depthStencil.depthBoundsTestEnable = VK_FALSE;
		depthStencil.minDepthBounds = 0.0f;
		depthStencil.maxDepthBounds = 1.0f;
		depthStencil.stencilTestEnable = VK_FALSE;
		depthStencil.front = {};
		depthStencil.back = {};
		return depthStencil;
	}

	inline auto getVkBlendFactor(BlendFactor factor) noexcept -> VkBlendFactor {
		switch (factor)
		{
		case SIByL::RHI::BlendFactor::ONE_MINUS_CONSTANT: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR; break;
		case SIByL::RHI::BlendFactor::CONSTANT: return VK_BLEND_FACTOR_CONSTANT_COLOR; break;
		case SIByL::RHI::BlendFactor::SRC_ALPHA_SATURATED: return VK_BLEND_FACTOR_SRC_ALPHA_SATURATE; break;
		case SIByL::RHI::BlendFactor::ONE_MINUS_DST_ALPHA: return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA; break;
		case SIByL::RHI::BlendFactor::DST_ALPHA: return VK_BLEND_FACTOR_DST_ALPHA; break;
		case SIByL::RHI::BlendFactor::ONE_MINUS_DST: return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR; break;
		case SIByL::RHI::BlendFactor::DST: return VK_BLEND_FACTOR_DST_COLOR; break;
		case SIByL::RHI::BlendFactor::ONE_MINUS_SRC_ALPHA: return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA; break;
		case SIByL::RHI::BlendFactor::SRC_ALPHA: return VK_BLEND_FACTOR_SRC_ALPHA; break;
		case SIByL::RHI::BlendFactor::ONE_MINUS_SRC: return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR; break;
		case SIByL::RHI::BlendFactor::SRC: return VK_BLEND_FACTOR_SRC_COLOR; break;
		case SIByL::RHI::BlendFactor::ONE: return VK_BLEND_FACTOR_ONE; break;
		case SIByL::RHI::BlendFactor::ZERO: return VK_BLEND_FACTOR_ZERO; break;
		default: return VK_BLEND_FACTOR_MAX_ENUM; break;
		}
	}

	inline auto getVkBlendOp(BlendOperation const& op) noexcept -> VkBlendOp {
		switch (op)
		{
		case BlendOperation::ADD: return VkBlendOp::VK_BLEND_OP_ADD;
		case BlendOperation::SUBTRACT: return VkBlendOp::VK_BLEND_OP_SUBTRACT;
		case BlendOperation::REVERSE_SUBTRACT: return VkBlendOp::VK_BLEND_OP_REVERSE_SUBTRACT;
		case BlendOperation::MIN: return VkBlendOp::VK_BLEND_OP_MIN;
		case BlendOperation::MAX: return VkBlendOp::VK_BLEND_OP_MAX;
		default: return VkBlendOp::VK_BLEND_OP_MAX_ENUM;
		}
	}

	inline auto getVkPipelineColorBlendAttachmentState(FragmentState const& state) noexcept -> std::vector<VkPipelineColorBlendAttachmentState> {
		std::vector<VkPipelineColorBlendAttachmentState> attachmentStates;
		for (ColorTargetState const& attchment : state.targets) {
			VkPipelineColorBlendAttachmentState colorBlendAttachment{};
			colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
			colorBlendAttachment.blendEnable = attchment.blend.blendEnable() ? VK_TRUE : VK_FALSE;
			colorBlendAttachment.srcColorBlendFactor = getVkBlendFactor(attchment.blend.color.srcFactor);
			colorBlendAttachment.dstColorBlendFactor = getVkBlendFactor(attchment.blend.color.dstFactor);
			colorBlendAttachment.colorBlendOp = getVkBlendOp(attchment.blend.color.operation);
			colorBlendAttachment.srcAlphaBlendFactor = getVkBlendFactor(attchment.blend.alpha.srcFactor);
			colorBlendAttachment.dstAlphaBlendFactor = getVkBlendFactor(attchment.blend.alpha.dstFactor);
			colorBlendAttachment.alphaBlendOp = getVkBlendOp(attchment.blend.color.operation);
			attachmentStates.emplace_back(colorBlendAttachment);
		}
		return attachmentStates;
	}

	inline auto getVkPipelineColorBlendStateCreateInfo(
		std::vector<VkPipelineColorBlendAttachmentState> & colorBlendAttachments
	) noexcept -> VkPipelineColorBlendStateCreateInfo {
		VkPipelineColorBlendStateCreateInfo colorBlending{};
		colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		colorBlending.logicOpEnable = VK_FALSE;
		colorBlending.logicOp = VK_LOGIC_OP_COPY;
		colorBlending.attachmentCount = colorBlendAttachments.size();
		colorBlending.pAttachments = colorBlendAttachments.data();
		colorBlending.blendConstants[0] = 0.0f;
		colorBlending.blendConstants[1] = 0.0f;
		colorBlending.blendConstants[2] = 0.0f;
		colorBlending.blendConstants[3] = 0.0f;
		return colorBlending;
	}

	auto getVkVertexInputBindingDescription(VertexState const& state) noexcept -> std::vector<VkVertexInputBindingDescription> {
		std::vector<VkVertexInputBindingDescription> descriptions;
		for (auto& buffer : state.buffers) {
			VkVertexInputBindingDescription bindingDescription{};
			bindingDescription.binding = 0;
			bindingDescription.stride = buffer.arrayStride;
			bindingDescription.inputRate = buffer.stepMode == VertexStepMode::VERTEX ?
				VK_VERTEX_INPUT_RATE_VERTEX : VK_VERTEX_INPUT_RATE_INSTANCE;
			descriptions.push_back(bindingDescription);
		}
		return descriptions;
	}

	inline auto getVkFormat(VertexFormat format) noexcept -> VkFormat {
		switch (format)
		{
		case SIByL::RHI::VertexFormat::SINT32X4:	return VK_FORMAT_R32G32B32A32_SINT;
		case SIByL::RHI::VertexFormat::SINT32X3:	return VK_FORMAT_R32G32B32_SINT;
		case SIByL::RHI::VertexFormat::SINT32X2:	return VK_FORMAT_R32G32_SINT;
		case SIByL::RHI::VertexFormat::SINT32:		return VK_FORMAT_R32_SINT;
		case SIByL::RHI::VertexFormat::UINT32X4:	return VK_FORMAT_R32G32B32A32_UINT;
		case SIByL::RHI::VertexFormat::UINT32X3:	return VK_FORMAT_R32G32B32_UINT;
		case SIByL::RHI::VertexFormat::UINT32X2:	return VK_FORMAT_R32G32_UINT;
		case SIByL::RHI::VertexFormat::UINT32:		return VK_FORMAT_R32_UINT;
		case SIByL::RHI::VertexFormat::FLOAT32X4:	return VK_FORMAT_R32G32B32A32_SFLOAT;
		case SIByL::RHI::VertexFormat::FLOAT32X3:	return VK_FORMAT_R32G32B32_SFLOAT;
		case SIByL::RHI::VertexFormat::FLOAT32X2:	return VK_FORMAT_R32G32_SFLOAT;
		case SIByL::RHI::VertexFormat::FLOAT32:		return VK_FORMAT_R32_SFLOAT;
		case SIByL::RHI::VertexFormat::FLOAT16X4:	return VK_FORMAT_R16G16B16A16_SFLOAT;
		case SIByL::RHI::VertexFormat::FLOAT16X2:	return VK_FORMAT_R16G16_SFLOAT;
		case SIByL::RHI::VertexFormat::SNORM16X4:	return VK_FORMAT_R16G16B16A16_SNORM;
		case SIByL::RHI::VertexFormat::SNORM16X2:	return VK_FORMAT_R16G16_SNORM;
		case SIByL::RHI::VertexFormat::UNORM16X4:	return VK_FORMAT_R16G16B16A16_UNORM;
		case SIByL::RHI::VertexFormat::UNORM16X2:	return VK_FORMAT_R16G16_UNORM;
		case SIByL::RHI::VertexFormat::SINT16X4:	return VK_FORMAT_R16G16B16A16_SINT;
		case SIByL::RHI::VertexFormat::SINT16X2:	return VK_FORMAT_R16G16_SINT;
		case SIByL::RHI::VertexFormat::UINT16X4:	return VK_FORMAT_R16G16B16A16_UINT;
		case SIByL::RHI::VertexFormat::UINT16X2:	return VK_FORMAT_R16G16_UINT;
		case SIByL::RHI::VertexFormat::SNORM8X4:	return VK_FORMAT_R8G8B8A8_SNORM;
		case SIByL::RHI::VertexFormat::SNORM8X2:	return VK_FORMAT_R8G8_SNORM;
		case SIByL::RHI::VertexFormat::UNORM8X4:	return VK_FORMAT_R8G8B8A8_UNORM;
		case SIByL::RHI::VertexFormat::UNORM8X2:	return VK_FORMAT_R8G8_UNORM;
		case SIByL::RHI::VertexFormat::SINT8X4:		return VK_FORMAT_R8G8B8A8_SINT;
		case SIByL::RHI::VertexFormat::SINT8X2:		return VK_FORMAT_R8G8_SINT;
		case SIByL::RHI::VertexFormat::UINT8X4:		return VK_FORMAT_R8G8B8A8_UINT;
		case SIByL::RHI::VertexFormat::UINT8X2:		return VK_FORMAT_R8G8_UINT;
		default: return VK_FORMAT_MAX_ENUM;
		}
	}

	inline auto getAttributeDescriptions(VertexState const& state) noexcept -> std::vector<VkVertexInputAttributeDescription> {
		std::vector<VkVertexInputAttributeDescription> attributeDescriptions;
		for (int i = 0; i < state.buffers.size(); ++i) {
			auto& buffer = state.buffers[i];
			for (int j = 0; j < buffer.attributes.size(); ++j) {
				auto& attribute = buffer.attributes[j];
				VkVertexInputAttributeDescription description = {};
				description.binding = i;
				description.location = attribute.shaderLocation;
				description.format = getVkFormat(attribute.format);
				description.offset = attribute.offset;
				attributeDescriptions.push_back(description);
			}
		}
		return attributeDescriptions;
	}

	inline auto fillFixedFunctionSettingDynamicInfo(
		RenderPipeline_VK::RenderPipelineFixedFunctionSettings& settings) noexcept -> void {
		// fill in 2 structure in the settings:
		// 1. std::vector<VkDynamicState> dynamicStates
		settings.dynamicStates = {
			VK_DYNAMIC_STATE_VIEWPORT,
			VK_DYNAMIC_STATE_SCISSOR
		};
		// 2. VkPipelineDynamicStateCreateInfo dynamicState
		settings.dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
		settings.dynamicState.dynamicStateCount = static_cast<uint32_t>(settings.dynamicStates.size());
		settings.dynamicState.pDynamicStates = settings.dynamicStates.data();
	}

	inline auto fillFixedFunctionSettingVertexInfo(VertexState const& state,
		RenderPipeline_VK::RenderPipelineFixedFunctionSettings& settings) noexcept -> void {
		// fill in 3 structure in the settings:
		// 1. std::vector<VkVertexInputBindingDescription>   vertexBindingDescriptor
		settings.vertexBindingDescriptor = getVkVertexInputBindingDescription(state);
		// 2. std::vector<VkVertexInputAttributeDescription> vertexAttributeDescriptions{};
		settings.vertexAttributeDescriptions = getAttributeDescriptions(state);
		// 3. VkPipelineVertexInputStateCreateInfo		   vertexInputState = {};
		VkPipelineVertexInputStateCreateInfo& vertexInput = settings.vertexInputState;
		vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		vertexInput.vertexBindingDescriptionCount = settings.vertexBindingDescriptor.size();
		vertexInput.pVertexBindingDescriptions = settings.vertexBindingDescriptor.data();
		vertexInput.vertexAttributeDescriptionCount = static_cast<uint32_t>(settings.vertexAttributeDescriptions.size());
		vertexInput.pVertexAttributeDescriptions = settings.vertexAttributeDescriptions.data();
	}

	inline auto fillFixedFunctionSettingViewportInfo(
		RenderPipeline_VK::RenderPipelineFixedFunctionSettings& settings) noexcept -> void {
		// fill in 1 structure in the settings, whose viewport & scisor could be set later
		VkPipelineViewportStateCreateInfo& viewportState = settings.viewportState;
		viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportState.viewportCount = 1;
		viewportState.scissorCount = 1;
	}

	RenderPipeline_VK::RenderPipeline_VK(Device_VK* device, RenderPipelineDescriptor const& desc)
		:device(device) 
	{
		if (desc.vertex.module)		fixedFunctionSetttings.shaderStages.push_back(static_cast<ShaderModule_VK*>(desc.vertex.module)->shaderStageInfo);
		if (desc.fragment.module)	fixedFunctionSetttings.shaderStages.push_back(static_cast<ShaderModule_VK*>(desc.fragment.module)->shaderStageInfo);
		
		fillFixedFunctionSettingDynamicInfo(fixedFunctionSetttings);
		fillFixedFunctionSettingVertexInfo(desc.vertex, fixedFunctionSetttings);
		fixedFunctionSetttings.assemblyState = getVkPipelineInputAssemblyStateCreateInfo(desc.primitive.topology);
		fillFixedFunctionSettingViewportInfo(fixedFunctionSetttings);
		fixedFunctionSetttings.rasterizationState = getVkPipelineRasterizationStateCreateInfo(desc.depthStencil, desc.fragment, desc.primitive);

		fixedFunctionSetttings.multisampleState = getVkPipelineMultisampleStateCreateInfo(desc.multisample);
		fixedFunctionSetttings.depthStencilState = getVkPipelineDepthStencilStateCreateInfo(desc.depthStencil);
		fixedFunctionSetttings.colorBlendAttachmentStates = getVkPipelineColorBlendAttachmentState(desc.fragment);
		fixedFunctionSetttings.colorBlendState = getVkPipelineColorBlendStateCreateInfo(fixedFunctionSetttings.colorBlendAttachmentStates);
		fixedFunctionSetttings.pipelineLayout = desc.layout;

		pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipelineInfo.stageCount = fixedFunctionSetttings.shaderStages.size();
		pipelineInfo.pStages = fixedFunctionSetttings.shaderStages.data();
		pipelineInfo.pVertexInputState = &fixedFunctionSetttings.vertexInputState;
		pipelineInfo.pInputAssemblyState = &fixedFunctionSetttings.assemblyState;
		pipelineInfo.pViewportState = &fixedFunctionSetttings.viewportState;
		pipelineInfo.pRasterizationState = &fixedFunctionSetttings.rasterizationState;
		pipelineInfo.pMultisampleState = &fixedFunctionSetttings.multisampleState;
		pipelineInfo.pDepthStencilState = &fixedFunctionSetttings.depthStencilState;
		pipelineInfo.pColorBlendState = &fixedFunctionSetttings.colorBlendState;
		pipelineInfo.pDynamicState = &fixedFunctionSetttings.dynamicState;
		pipelineInfo.layout = static_cast<PipelineLayout_VK*>(fixedFunctionSetttings.pipelineLayout)->pipelineLayout;
	}

	auto RenderPipeline_VK::combineRenderPass(RenderPass_VK* renderpass) noexcept -> void {
		// destroy current pipeline
		if (pipeline) {
			vkDestroyPipeline(device->getVkDevice(), pipeline, nullptr);
			pipeline = {};
		}
		pipelineInfo.renderPass = renderpass->renderPass;
		pipelineInfo.subpass = 0;
		pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
		pipelineInfo.basePipelineIndex = -1;

		if (vkCreateGraphicsPipelines(device->getVkDevice(), VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline) != VK_SUCCESS) {
			Core::LogManager::Error("VULKAN :: failed to create graphics pipeline!");
		}
	}

	RenderPipeline_VK::~RenderPipeline_VK() {
		if(pipeline) vkDestroyPipeline(device->getVkDevice(), pipeline, nullptr);
	}

	RenderPipeline_VK::RenderPipeline_VK(RenderPipeline_VK&& pipeline)
		: device(pipeline.device), pipeline(pipeline.pipeline) {
		pipeline.pipeline = nullptr;
	}

	auto RenderPipeline_VK::operator=(RenderPipeline_VK&& pipeline) -> RenderPipeline_VK& {
		device = pipeline.device;
		this->pipeline = pipeline.pipeline;
		pipeline.pipeline = nullptr;
		return *this;
	}

	auto Device_VK::createRenderPipeline(RenderPipelineDescriptor const& desc) noexcept -> std::unique_ptr<RenderPipeline> {
		return std::make_unique<RenderPipeline_VK>(this, desc);
	}

#pragma endregion

	// Pipelines Interface
	// ===========================================================================
	// Command Buffers Interface

	export struct CommandPool_VK {
		/** initialize */
		CommandPool_VK(Device_VK* device);
		/** destructor */
		~CommandPool_VK();
		/** allocate command buffer */
		auto allocateCommandBuffer() noexcept -> std::unique_ptr<CommandBuffer_VK>;
		/** vulkan command pool */
		VkCommandPool commandPool;
		/** the device this command pool is created on */
		Device_VK* device = nullptr;
	};

#pragma region VK_COMMANDPOOL_IMPL

	CommandPool_VK::CommandPool_VK(Device_VK* device)
		: device(device) 
	{
		VkCommandPoolCreateInfo poolInfo{};
		poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		poolInfo.queueFamilyIndex = device->getAdapterVk()->getQueueFamilyIndices().graphicsFamily.value();
		if (vkCreateCommandPool(device->getVkDevice(), &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
			Core::LogManager::Error("VULKAN :: failed to create command pool!");
		}
	}

	CommandPool_VK::~CommandPool_VK() {
		if(commandPool) vkDestroyCommandPool(device->getVkDevice(), commandPool, nullptr);
	}

	auto Device_VK::allocateCommandBuffer() noexcept -> std::unique_ptr<CommandBuffer_VK> { 
		return graphicPool->allocateCommandBuffer(); 
	}

#pragma endregion

	export struct CommandBuffer_VK :public CommandBuffer {
		/** vulkan command buffer */
		VkCommandBuffer commandBuffer;
		/** destructor */
		virtual ~CommandBuffer_VK();
		/** command pool the buffer is on */
		CommandPool_VK* commandPool = nullptr;
		/** the device this command buffer is created on */
		Device_VK* device = nullptr;
	};

#pragma region VK_COMMANDBUFFER_IMPL

	CommandBuffer_VK::~CommandBuffer_VK() {
		vkFreeCommandBuffers(device->getVkDevice(), commandPool->commandPool, 1, &commandBuffer);
	}

#pragma endregion

	export struct Semaphore_VK :public Semaphore {
		/** initialize */
		Semaphore_VK() = default;
		Semaphore_VK(Device_VK* device);
		/** virtual destructor */
		virtual ~Semaphore_VK();
		/** vulkan semaphore */
		VkSemaphore semaphore;
		/** the device this semaphore is created on */
		Device_VK* device = nullptr;
	};

#pragma region VK_SEMAPHORE_IMPL

	Semaphore_VK::Semaphore_VK(Device_VK* device)
		:device(device)
	{
		VkSemaphoreCreateInfo semaphoreInfo{};
		semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
		if (vkCreateSemaphore(device->getVkDevice(), &semaphoreInfo, nullptr, &semaphore) != VK_SUCCESS)
			Core::LogManager::Error("VULKAN :: failed to create semaphores!");
	}

	Semaphore_VK::~Semaphore_VK() {
		if (semaphore) vkDestroySemaphore(device->getVkDevice(), semaphore, nullptr);
	}

	auto Queue_VK::presentSwapChain(
		SwapChain* swapchain,
		uint32_t imageIndex,
		Semaphore* semaphore) noexcept -> void {
		VkPresentInfoKHR presentInfo{};
		presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
		presentInfo.waitSemaphoreCount = 1;
		presentInfo.pWaitSemaphores = &static_cast<Semaphore_VK*>(semaphore)->semaphore;
		VkSwapchainKHR swapChains[] = { static_cast<SwapChain_VK*>(swapchain)->swapChain };
		presentInfo.swapchainCount = 1;
		presentInfo.pSwapchains = swapChains;
		presentInfo.pImageIndices = &imageIndex;
		presentInfo.pResults = nullptr;
		vkQueuePresentKHR(queue, &presentInfo);
	}

#pragma endregion

	export struct Fence_VK :public Fence {
		/** initialize */
		Fence_VK() = default;
		Fence_VK(Device_VK* device);
		/** virtual destructor */
		virtual ~Fence_VK();
		/* wait the fence */
		virtual auto wait() noexcept -> void override;
		/* reset the fence */
		virtual auto reset() noexcept -> void override;
		/** vulkan fence */
		VkFence fence;
		/** the device this fence is created on */
		Device_VK* device = nullptr;
	};

#pragma region VK_FENCE_IMPL

	Fence_VK::Fence_VK(Device_VK* device)
		: device(device) {
		VkFenceCreateInfo fenceInfo{};
		fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
		if (vkCreateFence(device->getVkDevice(), &fenceInfo, nullptr, &fence) != VK_SUCCESS) {
			Core::LogManager::Error("VULKAN :: failed to create fence");
		}
	}

	Fence_VK::~Fence_VK() {
		if (fence) vkDestroyFence(device->getVkDevice(), fence, nullptr);
	}
	
	auto Fence_VK::wait() noexcept -> void {
		vkWaitForFences(device->getVkDevice(), 1, &fence, VK_TRUE, UINT64_MAX);
	}
	
	auto Fence_VK::reset() noexcept -> void {
		vkResetFences(device->getVkDevice(), 1, &fence);
	}

#pragma endregion


#pragma region VK_COMMANDBUFFER_IMPL

	auto CommandPool_VK::allocateCommandBuffer() noexcept -> std::unique_ptr<CommandBuffer_VK> {
		std::unique_ptr<CommandBuffer_VK> command = std::make_unique<CommandBuffer_VK>();
		command->device = device;
		command->commandPool = this;
		VkCommandBufferAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.commandPool = commandPool;
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandBufferCount = 1;
		if (vkAllocateCommandBuffers(device->getVkDevice(), &allocInfo, &command->commandBuffer) != VK_SUCCESS) {
			Core::LogManager::Error("VULKAN :: failed to allocate command buffers!");
		}
		return command;
	}

	auto Queue_VK::submit(std::vector<CommandBuffer*> const& commandBuffers) noexcept -> void {
		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		std::vector<VkCommandBuffer> vkCommandBuffers;
		for (auto buffer : commandBuffers)
			vkCommandBuffers.push_back((static_cast<CommandBuffer_VK*>(buffer))->commandBuffer);
		submitInfo.commandBufferCount = vkCommandBuffers.size();
		submitInfo.pCommandBuffers = vkCommandBuffers.data();
		vkQueueSubmit(device->getVkGraphicsQueue().queue, 1, &submitInfo, VK_NULL_HANDLE);
	}
	
	auto Queue_VK::submit(std::vector<CommandBuffer*> const& commandBuffers,
		Semaphore* wait, Semaphore* signal, Fence* fence) noexcept -> void {
		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		VkSemaphore waitSemaphores[] = { static_cast<Semaphore_VK*>(wait)->semaphore };
		VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
		if (wait) {
			submitInfo.waitSemaphoreCount = 1;
			submitInfo.pWaitSemaphores = waitSemaphores;
			submitInfo.pWaitDstStageMask = waitStages;
		}
		std::vector<VkCommandBuffer> vkCommandBuffers;
		for (auto buffer : commandBuffers)
			vkCommandBuffers.push_back((static_cast<CommandBuffer_VK*>(buffer))->commandBuffer);
		submitInfo.commandBufferCount = vkCommandBuffers.size();
		submitInfo.pCommandBuffers = vkCommandBuffers.data();
		VkSemaphore signalSemaphores[] = { static_cast<Semaphore_VK*>(signal)->semaphore };
		if (signal) {
			submitInfo.signalSemaphoreCount = 1;
			submitInfo.pSignalSemaphores = signalSemaphores;
		}
		if (vkQueueSubmit(device->getVkGraphicsQueue().queue, 1, &submitInfo, static_cast<Fence_VK*>(fence)->fence) != VK_SUCCESS) {
			Core::LogManager::Error("VULKAN :: failed to submit draw command buffer!");
		}
	}

#pragma endregion

	// Command Buffers Interface
	// ===========================================================================
	// Command Encoding Interface

	export struct MultiFrameFlights_VK :public MultiFrameFlights {
		/** initialize */
		MultiFrameFlights_VK(Device_VK* device, int maxFlightNum = 2, SwapChain* swapchain = nullptr);
		/** virtual destructor */
		~MultiFrameFlights_VK() = default;
		/** start frame */
		virtual auto frameStart() noexcept -> void override;
		/** end frame */
		virtual auto frameEnd() noexcept -> void override;
		/** get current flight id */
		virtual auto getFlightIndex() noexcept -> uint32_t { return currentFrame; }
		/** get current swapchain id */
		virtual auto getSwapchainIndex() noexcept -> uint32_t { return imageIndex; }
		/** get current command buffer */
		virtual auto getCommandBuffer() noexcept -> CommandBuffer* override;
		/** get current Image Available Semaphore */
		virtual auto getImageAvailableSeamaphore() noexcept -> Semaphore* override {
			return &imageAvailableSemaphores[currentFrame];
		}
		/** get current Render Finished Semaphore */
		virtual auto getRenderFinishedSeamaphore() noexcept -> Semaphore* override {
			return &renderFinishedSemaphores[currentFrame];
		}
		/** get current fence */
		virtual auto getFence() noexcept -> Fence* override { return &inFlightFences[currentFrame]; }
		std::vector<std::unique_ptr<CommandBuffer_VK>> commandBuffers;
		std::vector<Semaphore_VK> imageAvailableSemaphores;
		std::vector<Semaphore_VK> renderFinishedSemaphores;
		std::vector<Fence_VK> inFlightFences;
		SwapChain_VK* swapChain = nullptr;
		uint32_t currentFrame = 0;
		int maxFlightNum = 0;
		Device_VK* device = nullptr;
		uint32_t imageIndex;
	};

#pragma region VK_MULTIFRAMEFLIGHTS_IMPL

	MultiFrameFlights_VK::MultiFrameFlights_VK(Device_VK* device, int maxFlightNum, SwapChain* swapchain)
		: device(device), maxFlightNum(maxFlightNum), swapChain(static_cast<SwapChain_VK*>(swapchain))
	{
		commandBuffers.resize(maxFlightNum);
		for (size_t i = 0; i < maxFlightNum; ++i) {
			commandBuffers[i] = device->allocateCommandBuffer();
		}
		// createSyncObjects 
		imageAvailableSemaphores.resize(maxFlightNum);
		renderFinishedSemaphores.resize(maxFlightNum);
		inFlightFences.resize(maxFlightNum);
		VkSemaphoreCreateInfo semaphoreInfo{};
		semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
		VkFenceCreateInfo fenceInfo{};
		fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
		for (size_t i = 0; i < maxFlightNum; ++i) {
			if (vkCreateSemaphore(device->getVkDevice(), &semaphoreInfo, nullptr, &imageAvailableSemaphores[i].semaphore) != VK_SUCCESS ||
				vkCreateSemaphore(device->getVkDevice(), &semaphoreInfo, nullptr, &renderFinishedSemaphores[i].semaphore) != VK_SUCCESS ||
				vkCreateFence(device->getVkDevice(), &fenceInfo, nullptr, &inFlightFences[i].fence) != VK_SUCCESS) {
				Core::LogManager::Error("VULKAN :: failed to create synchronization objects for a frame!");
			}
			else {
				imageAvailableSemaphores[i].device = device;
				renderFinishedSemaphores[i].device = device;
				inFlightFences[i].device = device;
			}
		}
	}

	auto MultiFrameFlights_VK::frameStart() noexcept -> void {
		vkWaitForFences(device->getVkDevice(), 1, &inFlightFences[currentFrame].fence, VK_TRUE, UINT64_MAX);
		vkResetFences(device->getVkDevice(), 1, &inFlightFences[currentFrame].fence);
		vkAcquireNextImageKHR(device->getVkDevice(), swapChain->swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame].semaphore, VK_NULL_HANDLE, &imageIndex);
		vkResetCommandBuffer(commandBuffers[currentFrame]->commandBuffer, 0);
	}

	auto MultiFrameFlights_VK::frameEnd() noexcept -> void {
		device->getPresentQueue()->presentSwapChain(swapChain, imageIndex, &renderFinishedSemaphores[currentFrame]);
		currentFrame = (currentFrame + 1) % maxFlightNum;
	}

	auto MultiFrameFlights_VK::getCommandBuffer() noexcept -> CommandBuffer* {
		return commandBuffers[currentFrame].get();
	}

	auto Device_VK::createMultiFrameFlights(MultiFrameFlightsDescriptor const& desc) noexcept
		-> std::unique_ptr<MultiFrameFlights> 
	{
		return std::make_unique<MultiFrameFlights_VK>(this, desc.maxFlightNum, desc.swapchain);
	}

#pragma endregion

	export struct CommandEncoder_VK :public CommandEncoder {
		/** virtual descructor */
		virtual ~CommandEncoder_VK();
		/** Begins encoding a render pass described by descriptor. */
		virtual auto beginRenderPass(RenderPassDescriptor const& desc) noexcept -> std::unique_ptr<RenderPassEncoder> override;
		/** Begins encoding a compute pass described by descriptor. */
		virtual auto beginComputePass(ComputePassDescriptor const& desc) noexcept -> std::unique_ptr<ComputePassEncoder> override;
		/**  Encode a command into the CommandEncoder that copies data from
		* a sub-region of a GPUBuffer to a sub-region of another Buffer. */
		virtual auto copyBufferToBuffer(
			Buffer* source,
			size_t	sourceOffset,
			Buffer* destination,
			size_t	destinationOffset,
			size_t	size) noexcept -> void override;
		/** Encode a command into the CommandEncoder that fills a sub-region of a Buffer with zeros. */
		virtual auto clearBuffer(Buffer* buffer, size_t	offset, size_t	size) noexcept -> void override;
		/** Encode a command into the CommandEncoder that copies data from a sub-region of a Buffer
		* to a sub-region of one or multiple continuous texture subresources. */
		virtual auto copyBufferToTexture(
			ImageCopyBuffer  const& source,
			ImageCopyTexture const& destination,
			Extend3D		 const& copySize) noexcept -> void override;
		/** Encode a command into the CommandEncoder that copies data from a sub-region of
		* one or multiple continuous texture subresourcesto a sub-region of a Buffer. */
		virtual auto copyTextureToBuffer(
			ImageCopyTexture const& source,
			ImageCopyBuffer  const& destination,
			Extend3D		 const& copySize) noexcept -> void override;
		/** Encode a command into the CommandEncoder that copies data from
		* a sub-region of one or multiple contiguous texture subresources to
		* another sub-region of one or multiple continuous texture subresources. */
		virtual auto copyTextureToTexture(
			ImageCopyTexture const& source,
			ImageCopyTexture const& destination,
			Extend3D		 const& copySize) noexcept -> void override;
		/** Writes a timestamp value into a querySet when all
		* previous commands have completed executing. */
		virtual auto writeTimestamp(
			QuerySet* querySet,
			uint32_t  queryIndex) noexcept -> void override;
		/** Resolves query results from a QuerySet out into a range of a Buffer. */
		virtual auto resolveQuerySet(
			QuerySet* querySet,
			uint32_t  firstQuery,
			uint32_t  queryCount,
			Buffer&   destination,
			uint64_t  destinationOffset) noexcept -> void override;
		/** Completes recording of the commands sequence and returns a corresponding GPUCommandBuffer. */
		virtual auto finish(std::optional<CommandBufferDescriptor> const& descriptor = {}) noexcept -> CommandBuffer* override;
		/** underlying command buffer */
		std::unique_ptr<CommandBuffer_VK> commandBufferOnce = nullptr;
		/** underlying command buffer */
		CommandBuffer_VK* commandBuffer = nullptr;
	};

#pragma region VK_COMMANDENCODER_IMPL

	CommandEncoder_VK::~CommandEncoder_VK() {}

	auto CommandEncoder_VK::beginComputePass(ComputePassDescriptor const& desc) noexcept -> std::unique_ptr<ComputePassEncoder> {
		return nullptr;
	}
	
	auto CommandEncoder_VK::copyBufferToBuffer(
		Buffer* source,
		size_t	sourceOffset,
		Buffer* destination,
		size_t	destinationOffset,
		size_t	size) noexcept -> void
	{
		VkBufferCopy copyRegion{};
		copyRegion.srcOffset = sourceOffset;
		copyRegion.dstOffset = destinationOffset;
		copyRegion.size = size;
		vkCmdCopyBuffer(commandBuffer->commandBuffer, 
			static_cast<Buffer_VK*>(source)->getVkBuffer(), 
			static_cast<Buffer_VK*>(destination)->getVkBuffer(), 1, &copyRegion);
	}

	auto CommandEncoder_VK::clearBuffer(Buffer* buffer, size_t	offset, size_t	size) noexcept -> void {

	}
	
	auto CommandEncoder_VK::copyBufferToTexture(
		ImageCopyBuffer  const& source,
		ImageCopyTexture const& destination,
		Extend3D		 const& copySize) noexcept -> void
	{

	}

	auto CommandEncoder_VK::copyTextureToBuffer(
		ImageCopyTexture const& source,
		ImageCopyBuffer  const& destination,
		Extend3D		 const& copySize) noexcept -> void
	{

	}
	
	auto CommandEncoder_VK::copyTextureToTexture(
		ImageCopyTexture const& source,
		ImageCopyTexture const& destination,
		Extend3D		 const& copySize) noexcept -> void
	{

	}
	
	auto CommandEncoder_VK::writeTimestamp(
		QuerySet* querySet,
		uint32_t queryIndex) noexcept -> void
	{

	}
	
	auto CommandEncoder_VK::resolveQuerySet(
		QuerySet* querySet,
		uint32_t firstQuery,
		uint32_t queryCount,
		Buffer& destination,
		uint64_t destinationOffset) noexcept -> void
	{

	}
	
	auto CommandEncoder_VK::finish(std::optional<CommandBufferDescriptor> const& descriptor) noexcept -> CommandBuffer* {
		if (vkEndCommandBuffer(commandBuffer->commandBuffer) != VK_SUCCESS) {
			Core::LogManager::Error("VULKAN :: failed to record command buffer!");
		}
		return commandBuffer;
	}

	auto Device_VK::createCommandEncoder(CommandEncoderDescriptor const& desc) noexcept
		-> std::unique_ptr<CommandEncoder> 
	{
		std::unique_ptr<CommandEncoder_VK> encoder = std::make_unique<CommandEncoder_VK>();
		if (desc.externalCommandBuffer) {
			encoder->commandBuffer = static_cast<CommandBuffer_VK*>(desc.externalCommandBuffer);
		}
		else {
			encoder->commandBufferOnce = graphicPool->allocateCommandBuffer();
			encoder->commandBuffer = encoder->commandBufferOnce.get();
		}
		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = 0;
		beginInfo.pInheritanceInfo = nullptr;
		if (vkBeginCommandBuffer(encoder->commandBuffer->commandBuffer, &beginInfo) != VK_SUCCESS) {
			Core::LogManager::Error("VULKAN :: failed to begin recording command buffer!");
		}
		return encoder;
	}

#pragma endregion

	// Command Encoding Interface
	// ===========================================================================
	// Programmable Passes Interface



	// Programmable Passes Interface
	// ===========================================================================
	// Debug Marks Interface



	// Debug Marks Interface
	// ===========================================================================
	// Compute Passes Interface



	// Compute Passes Interface
	// ===========================================================================
	// Render Passes Interface

	export struct FrameBuffer_VK {
		/** intializer */
		FrameBuffer_VK(Device_VK* device, RHI::RenderPassDescriptor const& desc, RenderPass_VK* renderpass);
		/** destructor */
		~FrameBuffer_VK();
		/** get width of the framebuffer */
		auto width() -> uint32_t { return _width; }
		/** get height of the framebuffer */
		auto height() -> uint32_t { return _height; }
		/** vulkan framebuffer */
		VkFramebuffer framebuffer = {};
		/** clear values */
		std::vector<VkClearValue> clearValues = {};
		/** vulkan device the framebuffer created on */
		Device_VK* device = nullptr;
		/** width / height */
		uint32_t _width = 0, _height = 0;
	};

#pragma region VK_FRAMEBUFFER_IMPL

	FrameBuffer_VK::FrameBuffer_VK(Device_VK* device, RHI::RenderPassDescriptor const& desc, RenderPass_VK* renderpass)
		: device(device)
	{
		std::vector<VkImageView> attachments;
		for (int i = 0; i < desc.colorAttachments.size(); ++i) {
			attachments.push_back(static_cast<TextureView_VK*>(desc.colorAttachments[i].view)->imageView);
			clearValues.push_back(VkClearValue{ VkClearColorValue{
				(float)desc.colorAttachments[i].clearValue.r,
				(float)desc.colorAttachments[i].clearValue.g,
				(float)desc.colorAttachments[i].clearValue.b,
				(float)desc.colorAttachments[i].clearValue.a,
				} });
		}
		VkFramebufferCreateInfo framebufferInfo{};
		framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		framebufferInfo.renderPass = renderpass->renderPass;
		framebufferInfo.attachmentCount = attachments.size();
		framebufferInfo.pAttachments = attachments.data();
		framebufferInfo.width = (desc.colorAttachments.size() > 0) ?
			desc.colorAttachments[0].view->getTexture()->width() : desc.depthStencilAttachment.view->getTexture()->width();
		framebufferInfo.height = (desc.colorAttachments.size() > 0) ?
			desc.colorAttachments[0].view->getTexture()->height() : desc.depthStencilAttachment.view->getTexture()->height();
		_width = framebufferInfo.width;
		_height = framebufferInfo.height;
		framebufferInfo.layers = 1;
		if (vkCreateFramebuffer(device->getVkDevice(), &framebufferInfo, nullptr, &framebuffer) != VK_SUCCESS) {
			Core::LogManager::Error("VULKAN :: failed to create framebuffer!");
		}
	}

	FrameBuffer_VK::~FrameBuffer_VK() {
		if (framebuffer) vkDestroyFramebuffer(device->getVkDevice(), framebuffer, nullptr);
	}

#pragma endregion


	export struct RenderPassEncoder_VK :public RenderPassEncoder {
		/** virtual descructor */
		virtual ~RenderPassEncoder_VK();
		/** Sets the current GPURenderPipeline. */
		virtual auto setPipeline(RenderPipeline* pipeline) noexcept -> void override;
		/** Sets the current index buffer. */
		virtual auto setIndexBuffer(Buffer* buffer, IndexFormat indexFormat,
			uint64_t offset = 0, uint64_t size = 0) noexcept -> void override;
		/** Sets the current vertex buffer for the given slot. */
		virtual auto setVertexBuffer(uint32_t slot, Buffer* buffer,
			uint64_t offset = 0, uint64_t size = 0) noexcept -> void override;
		/** Draws primitives. */
		virtual auto draw(uint32_t vertexCount, uint32_t instanceCount = 1,
			uint32_t firstVertex = 0, uint32_t firstInstance = 0) noexcept -> void override;
		/** Draws indexed primitives. */
		virtual auto drawIndexed(uint32_t indexCount, uint32_t instanceCount = 1,
			uint32_t firstIndex = 0,
			int32_t  baseVertex = 0,
			uint32_t firstInstance = 0) noexcept -> void override;
		/** Draws primitives using parameters read from a GPUBuffer. */
		virtual auto drawIndirect(Buffer* indirectBuffer, uint64_t indirectOffset) noexcept -> void override;
		/** Draws indexed primitives using parameters read from a GPUBuffer. */
		virtual auto drawIndexedIndirect(Buffer* indirectBuffer, uint64_t indirectOffset) noexcept -> void override;
		/** Sets the viewport used during the rasterization stage to linearly map
		* from normalized device coordinates to viewport coordinates. */
		virtual auto setViewport(
			float x, float y,
			float width, float height,
			float minDepth, float maxDepth) noexcept -> void override;
		/** Sets the scissor rectangle used during the rasterization stage.
		* After transformation into viewport coordinates any fragments
		* which fall outside the scissor rectangle will be discarded. */
		virtual auto setScissorRect(
			IntegerCoordinate x, IntegerCoordinate y,
			IntegerCoordinate width, IntegerCoordinate height) noexcept -> void override;
		/** Sets the constant blend color and alpha values used with
		* "constant" and "one-minus-constant" GPUBlendFactors. */
		virtual auto setBlendConstant(Color color) noexcept -> void override;
		/** Sets the [[stencil_reference]] value used during
		* stencil tests with the "replace" GPUStencilOperation. */
		virtual auto setStencilReference(StencilValue reference) noexcept -> void override;
		/** begin occlusion query */
		virtual auto beginOcclusionQuery(uint32_t queryIndex) noexcept -> void override;
		/** end occlusion query */
		virtual auto endOcclusionQuery() noexcept -> void override;
		/** Executes the commands previously recorded into the given GPURenderBundles as part of this render pass. */
		virtual auto executeBundles(std::vector<RenderBundle> const& bundles) noexcept -> void override;
		/** Completes recording of the render pass commands sequence. */
		virtual auto end() noexcept -> void override;
		/** Sets the current GPUBindGroup for the given index. */
		virtual auto setBindGroup(uint32_t index, BindGroup* bindgroup,
			std::vector<BufferDynamicOffset> const& dynamicOffsets = {}) noexcept -> void override;
		/** Sets the current GPUBindGroup for the given index. */
		virtual auto setBindGroup(uint32_t index, BindGroup* bindgroup,
			uint64_t dynamicOffsetDataStart, uint32_t dynamicOffsetDataLength) noexcept -> void override;
		/** render pass */
		std::unique_ptr<RenderPass_VK> renderPass = nullptr;
		/** frame buffer */
		std::unique_ptr<FrameBuffer_VK> frameBuffer = nullptr;
		/* current render pipeline */
		RenderPipeline_VK* renderPipeline = nullptr;
		/** command buffer binded */
		CommandBuffer_VK* commandBuffer = nullptr;
	};
	
	RenderPassEncoder_VK::~RenderPassEncoder_VK() {

	}

	auto RenderPassEncoder_VK::setPipeline(RenderPipeline* pipeline) noexcept -> void {
		RenderPipeline_VK* vkpipeline = static_cast<RenderPipeline_VK*>(pipeline);
		renderPipeline = vkpipeline;
		vkpipeline->combineRenderPass(renderPass.get());
		vkCmdBindPipeline(commandBuffer->commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vkpipeline->pipeline);
	}

	auto RenderPassEncoder_VK::setIndexBuffer(Buffer* buffer, IndexFormat indexFormat, uint64_t offset, uint64_t size) noexcept -> void {
		vkCmdBindIndexBuffer(commandBuffer->commandBuffer, static_cast<Buffer_VK*>(buffer)->getVkBuffer(),
			offset, indexFormat == IndexFormat::UINT16_t ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32);
	}

	auto RenderPassEncoder_VK::setVertexBuffer(uint32_t slot, Buffer* buffer, uint64_t offset, uint64_t size) noexcept -> void {
		VkBuffer vertexBuffers[] = { static_cast<Buffer_VK*>(buffer)->getVkBuffer() };
		VkDeviceSize offsets[] = { offset };
		vkCmdBindVertexBuffers(commandBuffer->commandBuffer, 0, 1, vertexBuffers, offsets);
	}
	
	auto RenderPassEncoder_VK::draw(uint32_t vertexCount, uint32_t instanceCount, uint32_t firstVertex, uint32_t firstInstance) noexcept -> void {
		vkCmdDraw(commandBuffer->commandBuffer, vertexCount, instanceCount, firstVertex, firstInstance);
	}

	auto RenderPassEncoder_VK::drawIndexed(uint32_t indexCount, uint32_t instanceCount, uint32_t firstIndex, int32_t baseVertex, uint32_t firstInstance) noexcept -> void {
		vkCmdDrawIndexed(commandBuffer->commandBuffer, indexCount, instanceCount, firstIndex, baseVertex, firstInstance);
	}
	
	auto RenderPassEncoder_VK::drawIndirect(Buffer* indirectBuffer, uint64_t indirectOffset) noexcept -> void {

	}
	
	auto RenderPassEncoder_VK::drawIndexedIndirect(Buffer* indirectBuffer, uint64_t indirectOffset) noexcept -> void {

	}
	
	auto RenderPassEncoder_VK::setViewport(
		float x, float y,
		float width, float height,
		float minDepth, float maxDepth) noexcept -> void 
	{
		VkViewport viewport = {};
		viewport.x = x;
		viewport.y = y;
		viewport.width = width;
		viewport.height = height;
		viewport.minDepth = minDepth;
		viewport.maxDepth = maxDepth;
		vkCmdSetViewport(commandBuffer->commandBuffer, 0, 1, &viewport);
	}

	auto RenderPassEncoder_VK::setScissorRect(
		IntegerCoordinate x, IntegerCoordinate y,
		IntegerCoordinate width, IntegerCoordinate height) noexcept -> void
	{
		VkRect2D scissor;
		scissor.offset.x = x;
		scissor.offset.y = y;
		scissor.extent.width = width;
		scissor.extent.height = height;
		vkCmdSetScissor(commandBuffer->commandBuffer, 0, 1, &scissor);
	}

	auto RenderPassEncoder_VK::setBlendConstant(Color color) noexcept -> void {

	}
	
	auto RenderPassEncoder_VK::setStencilReference(StencilValue reference) noexcept -> void {

	}
	
	auto RenderPassEncoder_VK::beginOcclusionQuery(uint32_t queryIndex) noexcept -> void {

	}

	auto RenderPassEncoder_VK::endOcclusionQuery() noexcept -> void {

	}

	auto RenderPassEncoder_VK::executeBundles(std::vector<RenderBundle> const& bundles) noexcept -> void {

	}
	
	auto RenderPassEncoder_VK::end() noexcept -> void {
		vkCmdEndRenderPass(commandBuffer->commandBuffer);
	}

	auto RenderPassEncoder_VK::setBindGroup(uint32_t index, BindGroup* bindgroup,
		std::vector<BufferDynamicOffset> const& dynamicOffsets) noexcept -> void {
		vkCmdBindDescriptorSets(commandBuffer->commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, 
			static_cast<PipelineLayout_VK*>(renderPipeline->fixedFunctionSetttings.pipelineLayout)->pipelineLayout, 
			index, 1, &static_cast<BindGroup_VK*>(bindgroup)->set, 0, nullptr);
	}

	auto RenderPassEncoder_VK::setBindGroup(uint32_t index, BindGroup* bindgroup,
		uint64_t dynamicOffsetDataStart, uint32_t dynamicOffsetDataLength) noexcept -> void {
		vkCmdBindDescriptorSets(commandBuffer->commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
			static_cast<PipelineLayout_VK*>(renderPipeline->fixedFunctionSetttings.pipelineLayout)->pipelineLayout,
			index, 1, &static_cast<BindGroup_VK*>(bindgroup)->set, 0, nullptr);
	}

	auto CommandEncoder_VK::beginRenderPass(RenderPassDescriptor const& desc) noexcept -> std::unique_ptr<RenderPassEncoder> {
		std::unique_ptr<RenderPassEncoder_VK> renderpassEncoder = std::make_unique<RenderPassEncoder_VK>();
		renderpassEncoder->renderPass = std::make_unique<RenderPass_VK>(commandBuffer->device, desc);
		renderpassEncoder->commandBuffer = commandBuffer;
		renderpassEncoder->frameBuffer = std::make_unique<FrameBuffer_VK>(commandBuffer->device, desc, renderpassEncoder->renderPass.get());
		// render pass
		VkRenderPassBeginInfo renderPassInfo{};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		renderPassInfo.renderPass = renderpassEncoder->renderPass->renderPass;
		renderPassInfo.framebuffer = renderpassEncoder->frameBuffer->framebuffer;
		renderPassInfo.renderArea.offset = { 0, 0 };
		renderPassInfo.renderArea.extent = VkExtent2D{ renderpassEncoder->frameBuffer->width(), renderpassEncoder->frameBuffer->height() };
		renderPassInfo.pClearValues = renderpassEncoder->frameBuffer->clearValues.data();
		renderPassInfo.clearValueCount = renderpassEncoder->frameBuffer->clearValues.size();
		vkCmdBeginRenderPass(commandBuffer->commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
		return renderpassEncoder;
	}

	// Render Passes Interface
	// ===========================================================================
	// Bundles Interface



	// Bundles Interface
	// ===========================================================================
	// Queue Interface

}