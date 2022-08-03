export module RHI.Device:Adapter;
import Core.Memory;

namespace SIByL::RHI
{
	export struct Device;
	/**
	* Describes the physical properties of a given GPU.
	*/
	export struct Adapter
	{
		virtual auto requestDevice() const noexcept -> Scope<Device> = 0;
	};
}