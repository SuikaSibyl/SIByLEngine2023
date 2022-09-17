module;
#include <memory>
module Platform.Window:Window;
import Platform.Window;
import Core.Memory;

namespace SIByL::Platform
{
	auto Window::create(WindowOptions const& options) noexcept -> Scope<Window> {
		switch (options.vendor)
		{
		case WindowVendor::GLFW:
			return std::make_unique<Window_GLFW>(options);
			break;
		case WindowVendor::WIN_64:
			return std::make_unique<Window_Win64>(options);
			break;
		default:
			break;
		}
		return nullptr;
	}
}