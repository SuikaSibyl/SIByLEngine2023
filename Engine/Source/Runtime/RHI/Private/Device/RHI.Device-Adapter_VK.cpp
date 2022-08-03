module RHI.Device:Adapter_VK;
import RHI.Device;
import Core.Memory;
import :Adapter;

namespace SIByL::RHI
{
	auto Adapter_VK::requestDevice() const noexcept -> Scope<Device> {
		return nullptr;
	}
}