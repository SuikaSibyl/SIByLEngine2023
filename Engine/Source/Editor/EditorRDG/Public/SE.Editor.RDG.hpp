#pragma once
#include <SE.RDG.hpp>

namespace SIByL::Editor {
SE_EXPORT struct RDGViewerWidget : public Widget {
  RDG::Pipeline* pipeline = nullptr;
  /** draw gui*/
  virtual auto onDrawGui() noexcept -> void override;
};
}  // namespace SIByL::Editor
