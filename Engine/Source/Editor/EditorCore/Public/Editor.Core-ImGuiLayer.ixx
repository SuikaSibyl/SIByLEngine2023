module;
#include <imgui.h>
#include <memory>
#include <functional>
#include <unordered_map>
export module Editor.Core:ImGuiLayer;
import Core.System;
import Core.Resource.RuntimeManage;
import RHI;
import RHI.RHILayer;
import :ImGuiBackend;
import :ImGuiBackend.VK;

namespace SIByL::Editor
{
	export struct ImGuiLayer :public Core::Layer {
	public:
		/** initialzier */
		ImGuiLayer(RHI::RHILayer* rhiLayer);
		/** virtual destructor*/
		virtual ~ImGuiLayer();
		/* get singleton */
		static auto get() noexcept -> ImGuiLayer* { return singleton; }
		//auto onEvent(Event& e) -> void;
		auto onWindowResize(size_t x, size_t y) -> void;

		//auto createImImage(RHI::ISampler* sampler, RHI::ITextureView* view, RHI::ImageLayout layout) noexcept -> MemScope<ImImage>;
		//auto getImImage(GFX::Texture const& texture) noexcept -> ImImage*;

		auto startNewFrame() -> void;
		auto startGuiRecording() -> void;
		auto render() -> void;

		auto getDPI() noexcept -> float { return dpi; }
		auto createImGuiTexture(RHI::Sampler* sampler, RHI::TextureView* view, 
			RHI::TextureLayout layout) noexcept -> std::unique_ptr<ImGuiTexture>;

		/** rhi layer */
		RHI::RHILayer* rhiLayer = nullptr;
		/** imgui backend */
		std::unique_ptr<ImGuiBackend> imguiBackend = nullptr;
		/** imgui texture pool */
		std::unordered_map<Core::GUID, std::unique_ptr<ImGuiTexture>> ImGuiTexturePool = {};
	private:
		float dpi;
		static ImGuiLayer* singleton;
	};

#pragma region IMGUI_LAYER_IMPL
	
	ImGuiLayer* ImGuiLayer::singleton = nullptr;

	ImGuiLayer::ImGuiLayer(RHI::RHILayer* rhiLayer)
		: rhiLayer(rhiLayer)
	{
		singleton = this;

		if (rhiLayer->getRHILayerDescriptor().backend == RHI::RHIBackend::Vulkan) {
			imguiBackend = std::make_unique<ImGuiBackend_VK>(rhiLayer);
		}
		
		// Setup Dear ImGui context
		IMGUI_CHECKVERSION();
		ImGui::CreateContext();

		ImGuiIO& io = ImGui::GetIO(); (void)io;
		io.BackendFlags |= ImGuiBackendFlags_HasMouseCursors;
		io.BackendFlags |= ImGuiBackendFlags_HasSetMousePos;
		io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;       // Enable Keyboard Controls
		io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;           // Enable Docking
		io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;         // Enable Multi-Viewport / Platform Windows

		// Setup Dear ImGui style
		ImGui::StyleColorsDark();
		//ImGui::StyleColorsClassic();

		dpi = imguiBackend->getWindowDPI();
		io.Fonts->AddFontFromFileTTF("../Engine/Binaries/Runtime/fonts/opensans/OpenSans-Bold.ttf", dpi * 15.0f);
		io.FontDefault = io.Fonts->AddFontFromFileTTF("../Engine/Binaries/Runtime/fonts/opensans/OpenSans-Regular.ttf", dpi * 15.0f);

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
		// When viewports are enabled we tweak WindowRounding/WindowBg so platform windows can look identical to regular ones.
		ImGuiStyle& style = ImGui::GetStyle();
		if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
			style.WindowRounding = 0.0f;
			style.Colors[ImGuiCol_WindowBg].w = 1.0f;
		}

		imguiBackend->setupPlatformBackend();

		// Load Fonts
		// - If no fonts are loaded, dear imgui will use the default font. You can also load multiple fonts and use ImGui::PushFont()/PopFont() to select them.
		// - AddFontFromFileTTF() will return the ImFont* so you can store it if you need to select the font among multiple.
		// - If the file cannot be loaded, the function will return NULL. Please handle those errors in your application (e.g. use an assertion, or display an error and quit).
		// - The fonts will be rasterized at a given size (w/ oversampling) and stored into a texture when calling ImFontAtlas::Build()/GetTexDataAsXXXX(), which ImGui_ImplXXXX_NewFrame below will call.
		// - Read 'docs/FONTS.md' for more instructions and details.
		// - Remember that in C/C++ if you want to include a backslash \ in a string literal you need to write a double backslash \\ !
		//io.Fonts->AddFontDefault();
		//io.Fonts->AddFontFromFileTTF("../../misc/fonts/Roboto-Medium.ttf", 16.0f);
		//io.Fonts->AddFontFromFileTTF("../../misc/fonts/Cousine-Regular.ttf", 15.0f);
		//io.Fonts->AddFontFromFileTTF("../../misc/fonts/DroidSans.ttf", 16.0f);
		//io.Fonts->AddFontFromFileTTF("../../misc/fonts/ProggyTiny.ttf", 10.0f);
		//ImFont* font = io.Fonts->AddFontFromFileTTF("c:\\Windows\\Fonts\\ArialUni.ttf", 18.0f, NULL, io.Fonts->GetGlyphRangesJapanese());
		//IM_ASSERT(font != NULL);

		imguiBackend->uploadFonts();
	}
	
	ImGuiLayer::~ImGuiLayer() {
		imguiBackend = nullptr;
		ImGui::DestroyContext();
	}

	auto ImGuiLayer::onWindowResize(size_t x, size_t y) -> void {
		imguiBackend->onWindowResize(x, y);
	}

	auto ImGuiLayer::startNewFrame() -> void {
		imguiBackend->startNewFrame();
	}

	auto ImGuiLayer::startGuiRecording() -> void {
		ImGui::NewFrame();
		// Using Docking space
		{
			static bool dockspaceOpen = true;
			static bool opt_fullscreen = true;
			static bool opt_padding = false;
			static ImGuiDockNodeFlags dockspace_flags = ImGuiDockNodeFlags_None;

			// We are using the ImGuiWindowFlags_NoDocking flag to make the parent window not dockable into,
			// because it would be confusing to have two docking targets within each others.
			ImGuiWindowFlags window_flags = ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoDocking;
			if (opt_fullscreen) {
				const ImGuiViewport* viewport = ImGui::GetMainViewport();
				ImGui::SetNextWindowPos(viewport->WorkPos);
				ImGui::SetNextWindowSize(viewport->WorkSize);
				ImGui::SetNextWindowViewport(viewport->ID);
				ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
				ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
				window_flags |= ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove;
				window_flags |= ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;
			}
			else {
				dockspace_flags &= ~ImGuiDockNodeFlags_PassthruCentralNode;
			}
			// When using ImGuiDockNodeFlags_PassthruCentralNode, DockSpace() will render our background
			// and handle the pass-thru hole, so we ask Begin() to not render a background.
			if (dockspace_flags & ImGuiDockNodeFlags_PassthruCentralNode)
				window_flags |= ImGuiWindowFlags_NoBackground;
			// Important: note that we proceed even if Begin() returns false (aka window is collapsed).
			// This is because we want to keep our DockSpace() active. If a DockSpace() is inactive,
			// all active windows docked into it will lose their parent and become undocked.
			// We cannot preserve the docking relationship between an active window and an inactive docking, otherwise
			// any change of dockspace/settings would lead to windows being stuck in limbo and never being visible.
			if (!opt_padding)
				ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
			ImGui::Begin("DockSpace Demo", &dockspaceOpen, window_flags);
			if (!opt_padding)
				ImGui::PopStyleVar();
			if (opt_fullscreen)
				ImGui::PopStyleVar(2);
			// Submit the DockSpace
			ImGuiIO& io = ImGui::GetIO();
			ImGuiStyle& style = ImGui::GetStyle();
			//style.WindowMinSize.x = 350.0f;
			if (io.ConfigFlags & ImGuiConfigFlags_DockingEnable) {
				ImGuiID dockspace_id = ImGui::GetID("MyDockSpace");
				ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), dockspace_flags);
			}
		}
	}

	auto ImGuiLayer::render() -> void {
		// End docking space
		ImGui::End();
		// Do render ImGui stuffs
		ImGui::Render();
		ImDrawData* main_draw_data = ImGui::GetDrawData();
		const bool main_is_minimized = (main_draw_data->DisplaySize.x <= 0.0f || main_draw_data->DisplaySize.y <= 0.0f);
		if (!main_is_minimized)
			imguiBackend->render(main_draw_data);
		ImGuiIO& io = ImGui::GetIO(); (void)io;
		// Update and Render additional Platform Windows
		if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
			ImGui::UpdatePlatformWindows();
			ImGui::RenderPlatformWindowsDefault();
		}
		// Present Main Platform Window
		if (!main_is_minimized)
			imguiBackend->present();
	}

	auto ImGuiLayer::createImGuiTexture(RHI::Sampler* sampler, RHI::TextureView* view,
		RHI::TextureLayout layout) noexcept -> std::unique_ptr<ImGuiTexture> {
		return imguiBackend->createImGuiTexture(sampler, view, layout);
	}

#pragma endregion
}