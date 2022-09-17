export module RHI.Device:VK;
import :Interface;

namespace SIByL::RHI
{
	export struct Context_VK final :public Context
	{
		virtual auto init(Platform::Window* window = nullptr) noexcept -> bool override {

			return true;
		}
	};

	export struct Adapter_VK final :public Adapter
	{
		virtual auto requestDevice() const noexcept -> Scope<Device> override;
	};

	export struct Device_VK final :public Device
	{
	};
}