module;
#include <string>
#include <glfw3.h>
#include <format>
#include <Windows.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <glfw3native.h>
module Platform.Window:WindowGLFW;
import Platform.Window;
import :Window;
import Core.Log;
import Core.Event;
import Core.String;

namespace SIByL::Platform
{
	static bool gGLFWInitialized = false;
	static int	gGLFWWindowCount = 0;

	auto GLFWErrorCallback(int error, const char* description) -> void
	{
		Core::LogManager::Error(std::format("GLFW Error ({}): {}", error, description));
	}

	Window_GLFW::Window_GLFW(WindowOptions const& option)
		:uniName(option.title), width(option.width), height(option.height), properties(option.properties)
	{ init(); }

	auto Window_GLFW::init() noexcept -> bool {
		if (!gGLFWInitialized) {
			int success = glfwInit();
			glfwSetErrorCallback(GLFWErrorCallback);
			gGLFWInitialized = true;
		}
		gGLFWWindowCount++;

		// Context hint selection
		if (properties & WindowProperties::OPENGL_CONTEX) {
			glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
			glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
			glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
			glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
		}
		else if (properties & WindowProperties::VULKAN_CONTEX) {
			glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		}

		wndHandle = glfwCreateWindow(width, height, Core::string_cast(uniName).c_str(), nullptr, nullptr);
		glfwSetWindowUserPointer(wndHandle, this);

		// create context if need
		if (properties & WindowProperties::OPENGL_CONTEX)
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
		Core::LogManager::Error("Error|TODO :: Window_GLFW does not support func { invalid() } for now!");
	}
	
	auto Window_GLFW::endFrame() noexcept -> void {
		if (properties & WindowProperties::OPENGL_CONTEX)
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
		Core::LogManager::Error("Error|TODO :: Window_GLFW does not support func { resize(size_t x, size_t y) } for now!");
	}

	auto Window_GLFW::bindPaintingBitmapRGB8(size_t width, size_t height, char* data) noexcept -> void {
		Core::LogManager::Error("Error|TODO :: Window_GLFW does not support func { bindPaintingBitmapRGB8 } for now!");
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

	auto Window_GLFW::openFile(const char* filter) noexcept -> std::string {
		OPENFILENAMEA ofn;
		CHAR szFile[260] = { 0 };
		ZeroMemory(&ofn, sizeof(OPENFILENAME));
		ofn.lStructSize = sizeof(OPENFILENAME);
		ofn.hwndOwner = glfwGetWin32Window(wndHandle);
		ofn.lpstrFile = szFile;
		ofn.nMaxFile = sizeof(szFile);
		ofn.lpstrFilter = filter;
		ofn.nFilterIndex = 1;
		ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;
		if (GetOpenFileNameA(&ofn) == TRUE) {
			return ofn.lpstrFile;
		}
		return std::string();
	}

	auto Window_GLFW::saveFile(const char* filter, std::string const& name) noexcept -> std::string {
		OPENFILENAMEA ofn;
		CHAR szFile[260] = { 0 };
		memcpy(szFile, name.c_str(), name.size() + 1);
		ZeroMemory(&ofn, sizeof(OPENFILENAME));
		ofn.lStructSize = sizeof(OPENFILENAME);
		ofn.hwndOwner = glfwGetWin32Window(wndHandle);
		ofn.lpstrFile = szFile;
		ofn.nMaxFile = sizeof(szFile);
		ofn.lpstrFilter = filter;
		ofn.nFilterIndex = 1;
		ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;
		if (GetSaveFileNameA(&ofn) == TRUE) {
			return ofn.lpstrFile;
		}
		return std::string();
	}
}