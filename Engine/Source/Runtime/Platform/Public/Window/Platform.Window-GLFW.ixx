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
		Window_GLFW(std::wstring const& unique_name, uint32_t width, uint32_t height);

		auto create() noexcept -> bool;
		auto run() noexcept -> int;
		auto invalid() noexcept -> void;
		auto resize(size_t x, size_t y) noexcept -> void;
		auto destroy() noexcept -> void;
		auto isRunning() noexcept -> bool;

		//Core::EventSignal<HDC&> onPaintSignal;

	private:
		std::wstring const	uniName;
		bool				shouldQuit = false;
		GLFWwindow*			wndHandle = nullptr;
		Core::EventSignal<size_t, size_t> onResizeSignal;
		uint32_t width, height;
	};

	//export auto paintRGB8Bitmap(HDC& hdc, size_t width, size_t height, char* data) -> void;
}