module;
#include <set>
#include <vector>
#include <format>
#include <vulkan/vulkan.h>
#include <glfw3.h>
#include <memory>
#include <optional>
#include <algorithm>
export module RHI:VK;
import :Interface;
import Core.Log;
import Platform.Window;

namespace SIByL::RHI
{
	// **************************
	// Initialization			|
	struct Context_VK;
	struct Adapter_VK;
	struct Device_VK;
	struct VkQueueFamilyIndices;
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
	// Resource Binding		    |
	struct BindGroupLayout_VK;	
	struct BindGroup_VK;		
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
		/** initialize the context */
		virtual auto init(Platform::Window* window = nullptr, ContextExtensionsFlags ext = 0) noexcept -> bool override;
		/** Request an adapter */
		virtual auto requestAdapter(RequestAdapterOptions const& options) noexcept -> std::unique_ptr<Adapter> override;
		/** Get the binded window */
		virtual auto getBindedWindow() const noexcept -> Platform::Window* override;
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

	export struct VkQueueFamilyIndices {
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
		/** get VkPhysicalDevice */
		auto getVkPhysicalDevice() noexcept -> VkPhysicalDevice& { return physicalDevice; }
		/** get TimestampPeriod */
		auto getTimestampPeriod() const noexcept -> float { return timestampPeriod; }
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
		/** VkQueueFamilyIndices */
		VkQueueFamilyIndices queueFamilyIndices;
	};

	////////////////////////////////////
	//
	// Device
	//

	export struct Queue_VK :public Queue {
		/** Vulkan queue handle */
		VkQueue queue;
	};

	export struct Device_VK final :public Device {
		/** virtual destructor */
		virtual ~Device_VK();
		/** destroy the device */
		virtual auto destroy() noexcept -> void override;
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
			-> std::promise<std::unique_ptr<ComputePipeline>> override;
		/** create a render pipeline on the device in async way */
		virtual auto createRenderPipelineAsync(RenderPipelineDescriptor const& desc) noexcept
			-> std::promise<std::unique_ptr<RenderPipeline>> override;
		// Create command encoders
		// ---------------------------
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
		auto getGraphicsQueue() noexcept -> Queue_VK& { return graphicsQueue; }
		/** get compute queue handle */
		auto getComputeQueue() noexcept -> Queue_VK& { return computeQueue; }
		/** get present queue handle */
		auto getPresentQueue() noexcept -> Queue_VK& { return presentQueue; }
		/** get the adapter from which this device was created */
		auto getAdapterVk() noexcept -> Adapter_VK*& { return adapter; }
	private:
		/** vulkan logical device handle */
		VkDevice device;
		/** various queue handles */
		Queue_VK graphicsQueue, computeQueue, presentQueue;
		/** the adapter from which this device was created */
		Adapter_VK* adapter = nullptr;
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
				Core::LogManager::Log(std::format("VULKAN :: VALIDATION :: %s", pCallbackData->pMessage));
			break;
		case VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT:
			Core::LogManager::Log(std::format("VULKAN :: VALIDATION :: %s", pCallbackData->pMessage));
			break;
		case VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT:
			Core::LogManager::Warning(std::format("VULKAN :: VALIDATION :: %s", pCallbackData->pMessage));
			break;
		case VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT:
			Core::LogManager::Error(std::format("VULKAN :: VALIDATION :: %s", pCallbackData->pMessage));
			break;
		case VK_DEBUG_UTILS_MESSAGE_SEVERITY_FLAG_BITS_MAX_ENUM_EXT:
			Core::LogManager::Error(std::format("VULKAN :: VALIDATION :: %s", pCallbackData->pMessage));
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

	auto findQueueFamilies(Context_VK* contextVk, VkPhysicalDevice& device) noexcept -> VkQueueFamilyIndices {
		VkQueueFamilyIndices indices;
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
		VkQueueFamilyIndices indices = findQueueFamilies(contextVk, device);
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
		vkGetDeviceQueue(device->getVkDevice(), queueFamilyIndices.graphicsFamily.value(), 0, &device->getGraphicsQueue().queue);
		vkGetDeviceQueue(device->getVkDevice(), queueFamilyIndices.presentFamily.value(), 0, &device->getPresentQueue().queue);
		vkGetDeviceQueue(device->getVkDevice(), queueFamilyIndices.computeFamily.value(), 0, &device->getComputeQueue().queue);
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

	}

	auto Device_VK::createBuffer(BufferDescriptor const& desc) noexcept -> std::unique_ptr<Buffer> {
		return nullptr;
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

	auto Device_VK::createBindGroupLayout(BindGroupLayoutDescriptor const& desc) noexcept -> std::unique_ptr<BindGroupLayout> {
		return nullptr;
	}

	auto Device_VK::createPipelineLayout(PipelineLayoutDescriptor const& desc) noexcept -> std::unique_ptr<PipelineLayout> {
		return nullptr;
	}

	auto Device_VK::createBindGroup(BindGroupDescriptor const& desc) noexcept -> std::unique_ptr<BindGroup> {
		return nullptr;
	}

	auto Device_VK::createShaderModule(ShaderModuleDescriptor const& desc) noexcept -> std::unique_ptr<ShaderModule> {
		return nullptr;
	}

	auto Device_VK::createComputePipeline(ComputePipelineDescriptor const& desc) noexcept -> std::unique_ptr<ComputePipeline> {
		return nullptr;
	}

	auto Device_VK::createRenderPipeline(RenderPipelineDescriptor const& desc) noexcept -> std::unique_ptr<RenderPipeline> {
		return nullptr;
	}

	auto Device_VK::createComputePipelineAsync(ComputePipelineDescriptor const& desc) noexcept
		-> std::promise<std::unique_ptr<ComputePipeline>> {
		std::promise<std::unique_ptr<ComputePipeline>> r;
		r.set_value(nullptr);
		return r;
	}

	auto Device_VK::createRenderPipelineAsync(RenderPipelineDescriptor const& desc) noexcept
		-> std::promise<std::unique_ptr<RenderPipeline>> {
		std::promise<std::unique_ptr<RenderPipeline>> r;
		r.set_value(nullptr);
		return r;
	}

	auto Device_VK::createCommandEncoder(CommandEncoderDescriptor const& desc) noexcept
		-> std::unique_ptr<CommandEncoder> {
		return nullptr;
	}

	auto Device_VK::createRenderBundleEncoder(CommandEncoderDescriptor const& desc) noexcept
		-> std::unique_ptr<RenderBundleEncoder> {
		return nullptr;
	}

	auto Device_VK::createQuerySet(QuerySetDescriptor const& desc) noexcept -> std::unique_ptr<QuerySet> {
		return nullptr;
	}

#pragma endregion

	// Initialization Interface
	// ===========================================================================
	// Buffers Interface

	export struct Buffer_VK :public Buffer {
		virtual ~Buffer_VK();
		// Readonly Attributes
		// ---------------------------
		/** readonly get buffer size on GPU */
		virtual auto size() const noexcept -> size_t override { return _size; }
		/** readonly get buffer usage flags on GPU */
		virtual auto bufferUsageFlags() const noexcept -> BufferUsagesFlags = 0;
		/** readonly get map state on GPU */
		virtual auto bufferMapState() const noexcept -> BufferMapState = 0;
		// Map methods
		// ---------------------------
		/** Maps the given range of the GPUBuffer */
		virtual auto mapAsync(MapModeFlags mode, size_t offset = 0, size_t size = 0) noexcept -> std::promise<bool> = 0;
		/** Returns an ArrayBuffer with the contents of the GPUBuffer in the given mapped range */
		virtual auto getMappedRange(size_t offset = 0, size_t size = 0) noexcept -> ArrayBuffer = 0;
		/** Unmaps the mapped range of the GPUBuffer and makes it’s contents available for use by the GPU again. */
		virtual auto unmap() noexcept -> void = 0;
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
	protected:
		/** vulkan buffer */
		VkBuffer buffer = {};
		/** vulkan buffer device memory */
		VkDeviceMemory bufferMemory = {};
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

	inline auto getVkBufferUsage(BufferUsagesFlags usage) noexcept -> VkBufferUsageFlags {
		uint32_t flags{};
		if ((uint32_t)usage & (uint32_t)BufferUsage::COPY_SRC)
			flags |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
		if ((uint32_t)usage & (uint32_t)BufferUsage::COPY_DST)
			flags |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
		if ((uint32_t)usage & (uint32_t)BufferUsage::INDEX)
			flags |= VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
		if ((uint32_t)usage & (uint32_t)BufferUsage::VERTEX)
			flags |= VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
		if ((uint32_t)usage & (uint32_t)BufferUsage::UNIFORM)
			flags |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
		if ((uint32_t)usage & (uint32_t)BufferUsage::STORAGE)
			flags |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
		if ((uint32_t)usage & (uint32_t)BufferUsage::INDIRECT)
			flags |= VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
		return (VkBufferUsageFlags)flags;
	}

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

	auto Buffer_VK::init(Device_VK* device, size_t size, BufferDescriptor const& desc) noexcept -> void {
		this->_size = size;
		this->device = device;
		createBuffer(
			_size,
			getVkBufferUsage(desc.usage),
			getVkBufferShareMode(desc.shareMode),
			getVkMemoryProperty(desc.memoryProperties), 
			buffer,
			bufferMemory,
			device
		);
	}

	auto Buffer_VK::destroy() const noexcept -> void {
		if (buffer)		  vkDestroyBuffer(device->getVkDevice(), buffer, nullptr);
		if (bufferMemory) vkFreeMemory(device->getVkDevice(), bufferMemory, nullptr);
	}

#pragma endregion

	// Buffers Interface
	// ===========================================================================
	// Textures/TextureViews Interface

	export struct Texture_VK :public Texture {
		// Texture Behaviors
		// ---------------------------
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

#pragma region VK_TEXTURE_IMPL

	Texture_VK::~Texture_VK() {
		destroy();
	}

	auto Texture_VK::createView(TextureViewDescriptor const& desc) noexcept -> std::unique_ptr<TextureView> {
		return nullptr;
	}

	auto Texture_VK::destroy() noexcept -> void {

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

	export struct TextureView_VK :public TextureView {
		/** Vulkan texture view */
		VkImageView imageView;
		/** Texture view descriptor */
		TextureViewDescriptor descriptor;
		/** The device that the pointed texture is created on */
		Device_VK* device = nullptr;
	};

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
	// Resource Binding Interface

	export struct BindGroupLayout_VK :public BindGroupLayout {
		
	};


	// Bundles Interface
	// ===========================================================================
	// Queue Interface

}