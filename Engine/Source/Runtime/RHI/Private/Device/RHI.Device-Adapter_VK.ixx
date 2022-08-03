export module RHI.Device:Adapter_VK;
import Core.Memory;
import :Adapter;

namespace SIByL::RHI
{
	export struct Adapter_VK final :public Adapter
	{
		virtual auto requestDevice() const noexcept -> Scope<Device> override;
	};
}