#pragma once
#include <memory>
#include <string>
#include <imgui.h>
#include <unordered_map>
#include "seditor.hpp"
#include <se.gfx.hpp>
#include <se.rdg.hpp>

namespace se::editor {
struct SIByL_API RDGViewerWidget : public Widget {
  /** draw gui*/
  virtual auto onDrawGui() noexcept -> void override;
  rdg::Pipeline* pipeline = nullptr;
};
}