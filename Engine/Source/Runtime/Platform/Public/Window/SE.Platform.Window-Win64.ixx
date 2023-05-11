module;
#include <string>
#include <windows.h>
#include <Event/SE.Core.Event.hpp>
export module SE.Platform.Window:WindowWin64;
import :Window;

namespace SIByL::Platform
{
	export struct Window_Win64 :public Window {
		/** constructor */
		Window_Win64(WindowOptions const& option);
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
		virtual auto getVendor() noexcept -> WindowVendor { return WindowVendor::WIN_64; }
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
		HWND				wndHandle;
		HINSTANCE			instanceHandle;
		bool				shouldQuit = false;
		uint32_t			width, height;
		WindowProperties const properties;

		friend LRESULT CALLBACK StaticWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);
		auto wndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)noexcept -> LRESULT;

		// ---------------------------------
		// Signals
		// ---------------------------------
		Core::EventSignal<HDC&>				onPaintSignal;
		Core::EventSignal<size_t, size_t>	onResizeSignal;
	};

	export auto paintRGB8Bitmap(HDC& hdc, size_t width, size_t height, char* data) -> void;
}