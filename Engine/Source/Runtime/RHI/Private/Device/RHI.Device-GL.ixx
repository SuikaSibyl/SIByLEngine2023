module;
#include <glad/glad.h>
#include <glfw3.h>
export module RHI.Device:GL;
import :Interface;
import Core.Log;

namespace SIByL::RHI
{
	export struct Context_GL final :public Context
	{
		virtual auto init(Platform::Window* window = nullptr) noexcept -> bool override {
			if (window == nullptr || window->getVendor() == Platform::WindowVendor::WIN_64) {
				if (!gladLoadGL()) {
					Core::LogManager::Error("Context_GL Init Error with Win64 window!");
					return false;
				}
			}
			else if (window->getVendor() == Platform::WindowVendor::GLFW) {
				if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
					return false;
			}

			//Core::Root::get()->window->windowResizeSignal.connect([](size_t x, size_t y)->void {
			//	CommandList::SetViewportSize(x, y)();
			//	});

			return true;
		}

	};
}