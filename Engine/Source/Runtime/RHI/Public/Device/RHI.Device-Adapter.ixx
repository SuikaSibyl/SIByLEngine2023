export module RHI.Device:Adapter;

namespace SIByL::RHI
{
	/**
	* Describes the physical properties of a given GPU.
	*/
	export struct Adapter
	{
		auto requestDevice();
	};
}