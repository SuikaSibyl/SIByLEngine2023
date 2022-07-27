module;
#include <string>
#include <windows.h>
export module Platform.Window:WindowWin64;
import :Window;
import Core.Event;

namespace SIByL::Platform
{
	export struct Window_Win64 :public Window
	{
		Window_Win64(std::wstring const& unique_name);

		auto create() noexcept -> bool;
		auto run() noexcept -> int;
		auto invalid() noexcept -> void;
		auto resize(size_t x, size_t y) noexcept -> void;
		auto destroy() noexcept -> void;
		auto isRunning() noexcept -> bool;
		auto getHighDPI() noexcept -> float;

		Core::EventSignal<HDC&> onPaintSignal;

	private:
		std::wstring const	uniName;
		HWND				wndHandle;
		HINSTANCE			instanceHandle;
		bool				shouldQuit = false;

		friend LRESULT CALLBACK StaticWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);
		auto wndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)noexcept -> LRESULT;

		Core::EventSignal<size_t, size_t> onResizeSignal;
	};

	export auto paintRGB8Bitmap(HDC& hdc, size_t width, size_t height, char* data) -> void;
}