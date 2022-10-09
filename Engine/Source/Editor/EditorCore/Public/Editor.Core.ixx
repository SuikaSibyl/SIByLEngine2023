module;
#include <imgui.h>
export module Editor.Core;
import Core.System;

namespace SIByL::Editor 
{
	export struct ImGuiBackend {
		/** virtual destructor */
		virtual ~ImGuiBackend() = default;

		virtual auto setupPlatformBackend() noexcept -> void = 0;
		virtual auto uploadFonts() noexcept -> void = 0;
		virtual auto getWindowDPI() noexcept -> float = 0;
		//virtual auto onWindowResize(WindowResizeEvent& e) -> void = 0;

		virtual auto startNewFrame() -> void = 0;
		virtual auto render(ImDrawData* draw_data) -> void = 0;
		virtual auto present() -> void = 0;
	};

	export struct ImGuiLayer :public Core::Layer {
	public:
		virtual ~ImGuiLayer() = default;

		//auto onEvent(Event& e) -> void;
		//auto onWindowResize(WindowResizeEvent& e) -> bool;

		//auto createImImage(RHI::ISampler* sampler, RHI::ITextureView* view, RHI::ImageLayout layout) noexcept -> MemScope<ImImage>;
		//auto getImImage(GFX::Texture const& texture) noexcept -> ImImage*;

		auto startNewFrame() -> void;
		auto startGuiRecording() -> void;
		auto render() -> void;

		//RHI::API api;
		//ImImageLibrary imImageLibrary;
		//MemScope<RHI::ISampler> sampler;
		//MemScope<ImGuiBackend> backend;
	};
}