module;
#include <string>
#include <glfw3.h>
export module SE.Platform.Window:WindowGLFW;
import :Window;
import SE.Core.Event;

namespace SIByL::Platform
{
	export struct InputGLFW : public Input {
		InputGLFW(Window* attached_window);

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
		Window* attached_window;
	};

	export struct Window_GLFW :public Window {
		Window_GLFW(WindowOptions const& option);
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
		virtual auto getVendor() noexcept -> WindowVendor { return WindowVendor::GLFW; }
		/** return window handle */
		virtual auto getHandle() noexcept -> void* { return (void*)wndHandle; }
		/* return window framebuffer size */
		virtual auto getFramebufferSize(int* width, int* height) noexcept -> void override;
		/** return window input */
		virtual auto getInput() noexcept -> Input* override;

		// ---------------------------------
		// System Functional
		// ---------------------------------
		/** open a local file using browser */
		virtual auto openFile(const char* filter) noexcept -> std::string override;
		/** save a local file using browser */
		virtual auto saveFile(const char* filter, std::string const& name = {}) noexcept -> std::string override;

	private:
		std::wstring const	uniName;
		bool				shouldQuit = false;
		GLFWwindow*			wndHandle = nullptr;
		InputGLFW			input;
		Core::EventSignal<size_t, size_t> onResizeSignal;
		uint32_t			width, height;
		WindowProperties const properties;
	};

#pragma region GLTF_INPUT_IMPL

	InputGLFW::InputGLFW(Window* attached_window)
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

#pragma endregion
}