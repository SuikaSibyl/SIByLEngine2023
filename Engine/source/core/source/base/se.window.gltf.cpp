#define DLIB_EXPORT
#include <se.core.hpp>
#undef DLIB_EXPORT
#include <format>
#include <glfw/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <glfw/glfw3native.h>

namespace se {
struct InputGLFW : public input {
  InputGLFW(window* attached_window);
  virtual auto isKeyPressed(CodeEnum const& keycode) noexcept -> bool override;
  virtual auto isMouseButtonPressed(CodeEnum const& button) noexcept -> bool override;
  virtual auto getMousePosition(int button) noexcept -> std::pair<float, float> override;
  virtual auto getMouseX() noexcept -> float override;
  virtual auto getMouseY() noexcept -> float override;
  virtual auto getMouseScrollX() noexcept -> float override { float tmp = scrollX; scrollX = 0; return tmp; }
  virtual auto getMouseScrollY() noexcept -> float override { float tmp = scrollY; scrollY = 0; return tmp; }
  virtual auto disableCursor() noexcept -> void override;
  virtual auto enableCursor() noexcept -> void override;
  virtual auto decodeCodeEnum(CodeEnum const& code) noexcept -> int override;
  float scrollX = 0;
  float scrollY = 0;
private:
  window* attached_window;
};

struct Window_GLFW :public window {
  Window_GLFW(window::WindowOptions const& option);
  // ---------------------------------
  // Life Cycle
  // ---------------------------------
  /** intialize created window */
  virtual auto init() noexcept -> bool override;
  /** return whether the window is still runniong or has been closed */
  virtual auto isRunning() noexcept -> bool override;
  /** fetch window events */
  virtual auto fetchEvents() noexcept -> int override;
  /** flush window contents immediately */
  virtual auto invalid() noexcept -> void override;
  /** should be called when frame ends */
  virtual auto endFrame() noexcept -> void override;
  /** destroy window */
  virtual auto destroy() noexcept -> void override;
  // ---------------------------------
  // Event Based Behaviors
  // ---------------------------------
  /** resizie the window */
  virtual auto resize(size_t x, size_t y) noexcept -> void override;
  /** bind a block of CPU bitmap data to be drawn on the window */
  virtual auto bindPaintingBitmapRGB8(size_t width, size_t height, char* data) noexcept -> void override;
  /** connect resize signal events */
  virtual auto connectResizeEvent(std::function<void(size_t, size_t)> const& func) noexcept -> void override;
  // ---------------------------------
  // Fetch Properties
  // ---------------------------------
  /** return the high DPI value */
  virtual auto getHighDPI() noexcept -> float override;
  /** return vendor */
  virtual auto getVendor() noexcept -> window::Vendor override { return window::Vendor::GLFW; }
  /** return window handle */
  virtual auto getHandle() noexcept -> void* override { return (void*)wndHandle; }
  /* return window framebuffer size */
  virtual auto getFramebufferSize(int* width, int* height) noexcept -> void override;
  /** return window input */
  virtual auto getInput() noexcept -> input* override;
  // ---------------------------------
  // System Functional
  // ---------------------------------
  /** open a local file using browser */
  virtual auto openFile(const char* filter) noexcept -> std::string override;
  /** save a local file using browser */
  virtual auto saveFile(const char* filter, std::string const& name = {}) noexcept -> std::string override;
private:
  std::wstring const uniName;
  bool shouldQuit = false;
  GLFWwindow* wndHandle = nullptr;
  InputGLFW input;
  se::signal<size_t, size_t> onResizeSignal;
  uint32_t width, height;
  window::Properties const properties;
};

InputGLFW::InputGLFW(window* attached_window)
  :attached_window(attached_window) {}

auto InputGLFW::isKeyPressed(CodeEnum const& keycode) noexcept -> bool {
  auto window = static_cast<GLFWwindow*>(attached_window->getHandle());
  auto state = glfwGetKey(window, keycode.GLFWCode);
  return state == GLFW_PRESS || state == GLFW_REPEAT;
}

auto InputGLFW::isMouseButtonPressed(CodeEnum const& button) noexcept -> bool {
  auto window = static_cast<GLFWwindow*>(attached_window->getHandle());
  auto state = glfwGetMouseButton(window, button.GLFWCode);
  return state == GLFW_PRESS;
}

auto InputGLFW::getMousePosition(int button) noexcept -> std::pair<float, float> {
  auto window = static_cast<GLFWwindow*>(attached_window->getHandle());
  double xpos, ypos;
  glfwGetCursorPos(window, &xpos, &ypos);
  return { (float)xpos, (float)ypos };
}

auto InputGLFW::getMouseX() noexcept -> float {
  auto window = static_cast<GLFWwindow*>(attached_window->getHandle());
  double xpos, ypos;
  glfwGetCursorPos(window, &xpos, &ypos);
  return (float)xpos;
}

auto InputGLFW::getMouseY() noexcept -> float {
  auto window = static_cast<GLFWwindow*>(attached_window->getHandle());
  double xpos, ypos;
  glfwGetCursorPos(window, &xpos, &ypos);
  return (float)ypos;
}

auto InputGLFW::disableCursor() noexcept -> void {
  auto window = static_cast<GLFWwindow*>(attached_window->getHandle());
  glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
}

auto InputGLFW::enableCursor() noexcept -> void {
  auto window = static_cast<GLFWwindow*>(attached_window->getHandle());
  glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
}

auto InputGLFW::decodeCodeEnum(CodeEnum const& code) noexcept -> int {
  return code.GLFWCode;
}

static bool gGLFWInitialized = false;
static int	gGLFWWindowCount = 0;

auto GLFWErrorCallback(int error, const char* description) -> void {
  root::print::error(std::format("GLFW Error ({}): {}", error, description));
}

Window_GLFW::Window_GLFW(WindowOptions const& option)
  : uniName(option.title), width(option.width), height(option.height)
  , properties(option.properties), input(this) { init(); }

auto Window_GLFW::init() noexcept -> bool {
  if (!gGLFWInitialized) {
    if (!glfwInit()) {
      root::print::error("GLFW :: Initialization failed");
    }
  	glfwSetErrorCallback(GLFWErrorCallback);
  	gGLFWInitialized = true;
  }
  gGLFWWindowCount++;
  // Context hint selection
  if (properties == window::Properties::OPENGL_CONTEX) {
  	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
  } else if (properties == window::Properties::VULKAN_CONTEX) {
  	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  }
  wndHandle = glfwCreateWindow(width, height, platform::string_cast(uniName).c_str(), nullptr, nullptr);
  glfwSetWindowUserPointer(wndHandle, this);
  // create context if need
  if (properties == window::Properties::OPENGL_CONTEX)
  	glfwMakeContextCurrent(wndHandle);
  // Set GLFW Callbacks
  glfwSetWindowSizeCallback(wndHandle, [](GLFWwindow* window, int width, int height) {
  	Window_GLFW* this_window = (Window_GLFW*)glfwGetWindowUserPointer(window);
  this_window->onResizeSignal.emit(width, height);
  });
  return true;
}

auto Window_GLFW::fetchEvents() noexcept -> int {
  if (glfwWindowShouldClose(wndHandle)) shouldQuit = true;
  glfwPollEvents();
  return 0;
}

auto Window_GLFW::invalid() noexcept -> void {
  se::root::print::error("Error|TODO :: Window_GLFW does not support func { invalid() } for now!");
}

auto Window_GLFW::endFrame() noexcept -> void {
  if (properties == window::Properties::OPENGL_CONTEX)
	glfwSwapBuffers(wndHandle);
}

auto Window_GLFW::destroy() noexcept -> void {
  glfwDestroyWindow(wndHandle);
  gGLFWWindowCount--;
  if (gGLFWWindowCount <= 0) {
  	glfwTerminate();
  	gGLFWInitialized = false;
  }
}

auto Window_GLFW::isRunning() noexcept -> bool {
  return !shouldQuit;
}

auto Window_GLFW::resize(size_t x, size_t y) noexcept -> void {
  se::root::print::error("Error|TODO :: Window_GLFW does not support func { resize(size_t x, size_t y) } for now!");
}

auto Window_GLFW::bindPaintingBitmapRGB8(size_t width, size_t height, char* data) noexcept -> void {
  se::root::print::error("Error|TODO :: Window_GLFW does not support func { bindPaintingBitmapRGB8 } for now!");
}

auto Window_GLFW::connectResizeEvent(std::function<void(size_t, size_t)> const& func) noexcept -> void {
  onResizeSignal.connect(func);
}

auto Window_GLFW::getHighDPI() noexcept -> float {
  float xscale, yscale;
  GLFWmonitor* primary = glfwGetPrimaryMonitor();
  glfwGetMonitorContentScale(primary, &xscale, &yscale);
  return xscale;
}

auto Window_GLFW::getFramebufferSize(int* width, int* height) noexcept -> void {
  glfwGetFramebufferSize(wndHandle, width, height);
}

auto Window_GLFW::getInput() noexcept -> se::input* {
  return &input;
}

auto Window_GLFW::openFile(const char* filter) noexcept -> std::string {
  HWND handle = glfwGetWin32Window(wndHandle);
  return platform::openFile(filter, &handle);
}

auto Window_GLFW::saveFile(const char* filter, std::string const& name) noexcept -> std::string {
  HWND handle = glfwGetWin32Window(wndHandle);
  return platform::saveFile(filter, name , &handle);
}

auto window::create(WindowOptions const& options) noexcept -> std::unique_ptr<window> {
  switch (options.vendor) {
  case window::Vendor::GLFW: {
    std::unique_ptr<window> ret = std::make_unique<Window_GLFW>(options);
    uint32_t glfwExtensionCount = 0;
    char const** glfwExtensions;
    glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
    return ret;
  }
  case window::Vendor::WIN_64:
    se::root::print::error("core :: Win64 window is not implemented yet for current version");
  	return nullptr;
  default:
  	break;
  }
  return nullptr;
}
}