module;
#include <string>
#include <glfw3.h>
export module Platform.Window:WindowGLFW;
import :Window;
import Core.Event;

namespace SIByL::Platform
{
	export struct Window_GLFW :public Window
	{
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

		// ---------------------------------
		// Fetch Properties
		// ---------------------------------
		/** return the high DPI value */
		virtual auto getHighDPI() noexcept -> float override { return 1.f; }
		/** return vendor */
		virtual auto getVendor() noexcept -> WindowVendor { return WindowVendor::GLFW; }
		/** return window handle */
		virtual auto getHandle() noexcept -> void* { return (void*)wndHandle; }
		/* return window framebuffer size */
		virtual auto getFramebufferSize(int* width, int* height) noexcept -> void override;

	private:
		std::wstring const	uniName;
		bool				shouldQuit = false;
		GLFWwindow*			wndHandle = nullptr;
		Core::EventSignal<size_t, size_t> onResizeSignal;
		uint32_t			width, height;
		WindowProperties const properties;
	};

	//export auto paintRGB8Bitmap(HDC& hdc, size_t width, size_t height, char* data) -> void;
}