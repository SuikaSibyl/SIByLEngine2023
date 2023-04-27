module;
#include <string>
#include <vector>
#include <memory>
#include <imgui.h>
#include <typeinfo>
#include <unordered_map>
export module SE.Editor.Core:Framework;
import SE.Platform.Window;
import SE.Core.System;

namespace SIByL::Editor
{
	export struct Widget {
		/** virtual destructor */
		virtual ~Widget() = default;
		/** virtual draw gui*/
		virtual auto onDrawGui() noexcept -> void = 0;
		/** fetch common infomation */
		auto commonOnDrawGui() noexcept -> void {
			// get the screen pos
			info.windowPos = ImGui::GetWindowPos();
			// see whether it is hovered
			if (ImGui::IsWindowHovered())
				info.isHovered = true;
			else info.isHovered = false;
			if (ImGui::IsWindowFocused())
				info.isFocused = true;
			else info.isFocused = false;
		}
		struct WidgetInfo {
			ImVec2 windowPos;
			ImVec2 mousePos;
			bool isHovered;
			bool isFocused;
		} info;
	};

	export struct Fragment {
		/** virtual destructor */
		virtual ~Fragment() = default;
		/** virtual draw gui*/
		virtual auto onDrawGui(uint32_t flags, void* data) noexcept -> void = 0;
	};

	export struct EditorLayer :public Core::Layer {
		/** initialize */
		EditorLayer() { singleton = nullptr; }
		/** get singleton */
		static auto get() noexcept -> EditorLayer* { return singleton; }
		/** draw gui*/
		auto onDrawGui() noexcept -> void;
		/** register a widget to editor */
		template<class T>
		auto registerWidget() noexcept -> void {
			widgets[typeid(T).name()] = std::make_unique<T>();
		}
		/** get a widget registered in editor */
		template<class T>
		auto getWidget() noexcept -> T* {
			auto iter = widgets.find(typeid(T).name());
			if (iter == widgets.end()) return nullptr;
			else return static_cast<T*>(iter->second.get());
		}
		/** add a widget to editor */
		template<class T>
		auto addWidget(std::string const& name) noexcept -> void {
			mutable_widgets[name] = std::make_unique<T>();
		}
		/** remove a widget from editor */
		template<class T>
		auto removeWidget(std::string const& name) noexcept -> void {
			mutable_widgets.erase(name);
		}
	private:
		static EditorLayer* singleton;
		/** all the widgets registered */
		std::unordered_map<char const*, std::unique_ptr<Widget>> widgets = {};
		/** all the widgets registered */
		std::unordered_map<std::string, std::unique_ptr<Widget>> mutable_widgets = {};
	};
	
	EditorLayer* EditorLayer::singleton = nullptr;

	auto EditorLayer::onDrawGui() noexcept -> void {
		for (auto& iter : widgets) {
			iter.second->onDrawGui();
		}
	}
}