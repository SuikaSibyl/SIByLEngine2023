module;
#include <string>
#include <glfw3.h>
#include <format>
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

}