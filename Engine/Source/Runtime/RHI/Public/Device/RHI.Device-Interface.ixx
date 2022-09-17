export module RHI.Device:Interface;
import Core.Memory;
import Platform.Window;

namespace SIByL::RHI
{
	struct Context;
	struct Adapter;
	struct Device;

	export struct Context
	{
		virtual auto init(Platform::Window* window = nullptr) noexcept -> bool = 0;

		//virtual auto requestAdapter() noexcept -> Scope<Adapter> = 0;
	};

	/**
	* Describes the physical properties of a given GPU.
	*/
	export struct Adapter
	{
		virtual auto requestDevice() const noexcept -> Scope<Device> = 0;
	};

	/**
	* Device is a logical instantiation of an adapter, through which internal objects are created.
	* Is the exclusive owner of all internal objects created from it.
	*/
	export struct Device
	{
		/** the adapter from which this device was created */
		Adapter* adapter;

		/** the features which can be used on this device */

	};
}