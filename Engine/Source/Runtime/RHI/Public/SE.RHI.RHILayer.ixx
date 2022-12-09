module;
#include <Memory>
export module SE.RHI:RHILayer;
import :Interface;
import :VK;
import SE.Core.System;
import SE.Platform.Window;

namespace SIByL::RHI
{
	export enum struct RHIBackend {
		None,
		OpenGL,
		Vulkan,
	};

	export struct RHILayerDescriptor {
		RHIBackend backend = RHIBackend::None;
		ContextExtensionsFlags extensions = {};
		Platform::Window* windowBinded = nullptr;
		bool useImGui = false;
	};

	export struct RHILayer :public Core::Layer {
		/** initialzier */
		RHILayer(RHILayerDescriptor const& desc);
		/** virtual destructor */
		virtual ~RHILayer();
		/** get rhi context */
		auto getContext() noexcept -> RHI::Context* { return context.get(); }
		/** get rhi adapter */
		auto getAdapter() noexcept -> RHI::Adapter* { return adapter.get(); }
		/** get rhi device */
		auto getDevice() noexcept -> RHI::Device* { return device.get(); }
		/** get swapChain device */
		auto getSwapChain() noexcept -> RHI::SwapChain* { return swapChain.get(); }
		/** get multi frame flights device */
		auto getMultiFrameFlights() noexcept -> RHI::MultiFrameFlights* { return multiFrameFlights.get(); }
		/** get descriptor */
		auto getRHILayerDescriptor() const noexcept -> RHILayerDescriptor const& { return desc; }
		/** get singleton */
		static auto get() noexcept ->RHILayer* { return singleton; }
	private:
		RHILayerDescriptor const desc;
		/** rhi context */
		std::unique_ptr<RHI::Context> context = nullptr;
		/** rhi adapter */
		std::unique_ptr<RHI::Adapter> adapter = nullptr;
		/** rhi device */
		std::unique_ptr<RHI::Device> device = nullptr;
		/** swapChain device */
		std::unique_ptr<RHI::SwapChain> swapChain = nullptr;
		/** multi frame flights device */
		std::unique_ptr<RHI::MultiFrameFlights> multiFrameFlights = nullptr;
		/** singleton */
		static RHILayer* singleton;
	};
	
	RHILayer* RHILayer::singleton = nullptr;

	RHILayer::RHILayer(RHILayerDescriptor const& desc) : desc(desc) {
		singleton = this;
		if (desc.backend == RHIBackend::Vulkan) {
			context = std::make_unique<Context_VK>();
			context->init(desc.windowBinded, desc.extensions);
			adapter = context->requestAdapter({});
			device = adapter->requestDevice();
			multiFrameFlights = device->createMultiFrameFlights({ 2, swapChain.get() });
			if (!desc.useImGui) {
				swapChain = device->createSwapChain({});
				desc.windowBinded->connectResizeEvent([&](size_t w, size_t h)->void {swapChain->recreate(); });
			}
		}
	}

	RHILayer::~RHILayer() {
		if (desc.backend == RHIBackend::Vulkan) {
			static_cast<Context_VK*>(context.get())->getVkSurfaceKHR() = {};
		}
	}
}