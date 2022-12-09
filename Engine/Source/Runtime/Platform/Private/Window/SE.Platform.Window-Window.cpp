module;
#include <memory>
module SE.Platform.Window:Window;
import SE.Platform.Window;
import SE.Core.Memory;

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