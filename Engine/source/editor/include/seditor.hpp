#pragma once
#include <memory>
#include <string>
#include <imgui.h>
#include <unordered_map>
#include <se.editor.hpp>

namespace se::editor {
struct SIByL_API EditorContext {
  /** initialize editor */
  static auto initialize() noexcept -> void;
  /** draw gui*/
  static auto onDrawGui() noexcept -> void;
  /** register a widget to editor */ template <class T>
  static auto registerWidget() noexcept -> void { widgets[typeid(T).name()] = std::make_unique<T>(); }
  /** add a widget to editor */ template <class T>
  static auto addWidget(std::string const& name) noexcept -> void { widgets[name] = std::make_unique<T>(); }
  /** remove a widget from editor */ template <class T>
  static auto removeWidget(std::string const& name) noexcept -> void { widgets.erase(name); }
  /** get a widget registered in editor */ template <class T>
  static auto getWidget() noexcept -> T* {
    auto iter = widgets.find(std::string(typeid(T).name()));
    if (iter == widgets.end()) return nullptr;
    else return static_cast<T*>(iter->second.get()); }
  /** all the widgets registered */
  static std::unordered_map<std::string, std::unique_ptr<Widget>> widgets;
};
}