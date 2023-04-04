module;
#include <vector>
#include <string>
#include <unordered_map>
#include <cstdint>
#include <typeinfo>
#include <memory>
#include <functional>
#include <imgui.h>
#include <imgui_internal.h>
export module SE.Editor.GFX:InspectorWidget;
import :Utils;
import :ResourceViewer;
import SE.Editor.Core;
import SE.Core.ECS;
import SE.GFX;

namespace SIByL::Editor
{
	export struct InspectorWidget :public Widget {
		/** draw gui*/
		virtual auto onDrawGui() noexcept -> void override;
		/** set custom draw */
		auto setCustomDraw(std::function<void()> func) noexcept -> void { customDraw = func; }
		/** set empty */
		auto setEmpty() noexcept -> void;
		/** custom draw on inspector widget */
		std::function<void()> customDraw;
		/** resource viewer */
		ResourceViewer resourceViewer;
	};

	export struct CustomInspector {
		/** draw gui*/
		virtual auto onDrawGui() noexcept -> void = 0;
		/** register a widget to editor */
		template<class T>
		auto registerFragment() noexcept -> void {
			fragments[typeid(T).name()] = std::make_unique<T>();
			fragmentSequence.push_back(fragments[typeid(T).name()].get());
		}
		/** set insepctor widget to show this cutom one */
		auto setInspectorWidget(InspectorWidget* widget) noexcept -> void;
		/** get a widget registered in editor */
		template<class T>
		auto getFragment() noexcept -> T* {
			auto iter = fragments.find(typeid(T).name());
			if (iter == fragments.end()) return nullptr;
			else return static_cast<T*>(iter->second.get());
		}
	protected:
		/** fragment sequences to be drawn */
		std::vector<Fragment*> fragmentSequence = {};
		/** all the widgets registered */
		std::unordered_map<char const*, std::unique_ptr<Fragment>> fragments = {};
	};

	auto InspectorWidget::onDrawGui() noexcept -> void {
		ImGui::Begin("Inspector", 0, ImGuiWindowFlags_MenuBar);
		if (customDraw) customDraw();
		ImGui::End();
	}

	auto InspectorWidget::setEmpty() noexcept -> void {
		setCustomDraw([]() {});
	}

	auto CustomInspector::setInspectorWidget(InspectorWidget* widget) noexcept -> void {
		widget->setCustomDraw(std::bind(&CustomInspector::onDrawGui, this));
	}
}