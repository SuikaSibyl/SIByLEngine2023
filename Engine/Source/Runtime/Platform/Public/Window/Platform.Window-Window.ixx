module;
#include <string>
export module Platform.Window:Window;
import Core.Memory;

namespace SIByL::Platform
{
	export enum struct WindowVendor {
		GLFW,
		WIN_64,
	};

	namespace FlagEnum {
		export enum WindowProperties {
			OPENGL_CONTEX = 1 << 0,
			VULKAN_CONTEX = 1 << 1,
		};
	}
	export using WindowProperties = FlagEnum::WindowProperties;

	export struct WindowOptions {
		WindowVendor vendor;
		std::wstring title;
		uint32_t width, height;
		WindowProperties properties = static_cast<WindowProperties>(0);
	};

	export struct Window
	{
		/** create a window with options */
		static auto create(WindowOptions const& options) noexcept -> Scope<Window>;

		// ---------------------------------
		// Life Cycle
		// ---------------------------------
		/** intialize created window */
		virtual auto init() noexcept -> bool = 0;
		/** return whether the window is still runniong or has been closed */
		virtual auto isRunning() noexcept -> bool = 0;
		/** fetch window events */
		virtual auto fetchEvents() noexcept -> int = 0;
		/** flush window contents immediately */
		virtual auto invalid() noexcept -> void = 0;
		/** should be called when frame ends */
		virtual auto endFrame() noexcept -> void = 0;
		/** destroy window */
		virtual auto destroy() noexcept -> void = 0;

		// ---------------------------------
		// Event Based Behaviors
		// ---------------------------------
		/** resizie the window */
		virtual auto resize(size_t x, size_t y) noexcept -> void = 0;
		/* bind a block of CPU bitmap data to be drawn on the window */
		virtual auto bindPaintingBitmapRGB8(size_t width, size_t height, char* data) noexcept -> void = 0;

		// ---------------------------------
		// Fetch Properties
		// ---------------------------------
		/** return the high DPI value */
		virtual auto getHighDPI() noexcept -> float = 0;
		/** return vendor */
		virtual auto getVendor() noexcept -> WindowVendor = 0;
		/** return window handle */
		virtual auto getHandle() noexcept -> void* = 0;
		/* return window framebuffer size */
		virtual auto getFramebufferSize(int* width, int* height) noexcept -> void = 0;
	};
}