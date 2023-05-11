#pragma once

#include <glad/glad.h>
#include <glfw3.h>
#include <memory>
#include <string>
#include <SE.RHI-Interface.hpp>
#include <Print/SE.Core.Log.hpp>
import SE.Platform.Window;

namespace SIByL::RHI {
SE_EXPORT struct Context_GL final : public Context {
  virtual auto init(Platform::Window* window = nullptr,
                    ContextExtensionsFlags ext = 0) noexcept -> bool override;

  virtual auto requestAdapter(RequestAdapterOptions const& options) noexcept
      -> std::unique_ptr<Adapter> override;

  virtual auto getBindedWindow() const noexcept -> Platform::Window* override {
    return bindedWindow;
  }

 private:
  Platform::Window* bindedWindow = nullptr;
};

SE_EXPORT struct Adapter_GL final : public Adapter {};

#pragma region DEVICE_PARTIAL_IMPL

auto Context_GL::init(Platform::Window* window,
                      ContextExtensionsFlags ext) noexcept -> bool {
  if (window == nullptr ||
      window->getVendor() == Platform::WindowVendor::WIN_64) {
    if (!gladLoadGL()) {
      Core::LogManager::Error("Context_GL Init Error with Win64 window!");
      return false;
    }
  } else if (window->getVendor() == Platform::WindowVendor::GLFW) {
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) return false;
  }

  // Core::Root::get()->window->windowResizeSignal.connect([](size_t x, size_t
  // y)->void { 	CommandList::SetViewportSize(x, y)();
  //	});

  return true;
}

#pragma endregion

}  // namespace SIByL::RHI