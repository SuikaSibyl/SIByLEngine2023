module;
#include <string>
#include <glfw3.h>
#include <format>
module Platform.Window:WindowGLFW;
import Platform.Window;
import :Window;
import Core.Log;
import Core.Event;

namespace SIByL::Platform
{
	static bool gGLFWInitialized = false;
	static int	gGLFWWindowCount = 0;

	auto GLFWErrorCallback(int error, const char* description) -> void
	{
		Core::LogManager::Error(std::format("GLFW Error ({}): {}", error, description));
	}

	Window_GLFW::Window_GLFW(std::wstring const& unique_name, uint32_t width, uint32_t height)
		:uniName(unique_name), width(width), height(height)
	{}

	auto Window_GLFW::create() noexcept -> bool {
		if (!gGLFWInitialized) {
			int success = glfwInit();
			glfwSetErrorCallback(GLFWErrorCallback);
			gGLFWInitialized = true;
		}
		gGLFWWindowCount++;
		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		wndHandle = glfwCreateWindow(width, height, "Hello", nullptr, nullptr);
		glfwSetWindowUserPointer(wndHandle, this);
		return true;
	}

	auto Window_GLFW::run() noexcept -> int {
		glfwPollEvents();
		return 0;
	}

	//auto Window_GLFW::invalid() noexcept -> void;
	//auto Window_GLFW::resize(size_t x, size_t y) noexcept -> void;
	
	auto Window_GLFW::destroy() noexcept -> void {
		glfwDestroyWindow(wndHandle);
		gGLFWWindowCount--;

		if (gGLFWWindowCount <= 0)
		{
			glfwTerminate();
			gGLFWInitialized = false;
		}

	}

	auto Window_GLFW::isRunning() noexcept -> bool {
		return !shouldQuit;
	}

	//export auto paintRGB8Bitmap(HDC& hdc, size_t width, size_t height, char* data) -> void;
}