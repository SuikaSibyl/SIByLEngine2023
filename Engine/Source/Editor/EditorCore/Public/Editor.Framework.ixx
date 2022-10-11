module;
#include <vector>
#include <memory>
#include <typeinfo>
#include <unordered_map>
export module Editor.Framework;
import Core.System;

namespace SIByL::Editor
{
	export struct Widget {
		/** virtual destructor */
		virtual ~Widget() = default;
		/** virtual draw gui*/
		virtual auto onDrawGui() noexcept -> void = 0;
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
	private:
		static EditorLayer* singleton;
		/** all the widgets registered */
		std::unordered_map<char const*, std::unique_ptr<Widget>> widgets = {};
	};
	
	EditorLayer* EditorLayer::singleton = nullptr;

	auto EditorLayer::onDrawGui() noexcept -> void {
		for (auto& iter : widgets) {
			iter.second->onDrawGui();
		}
	}
}