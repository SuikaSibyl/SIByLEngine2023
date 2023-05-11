#include <SE.Editor.Core-Framework.hpp>
import SE.Platform.Window;

namespace SIByL::Editor {
EditorLayer* EditorLayer::singleton = nullptr;

auto EditorLayer::onDrawGui() noexcept -> void {
  for (auto& iter : widgets) {
    iter.second->onDrawGui();
  }
}
}  // namespace SIByL::Editor