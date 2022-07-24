export module RHI.Device:Device;

namespace SIByL::RHI
{
	export struct Adapter;
	/*
	* A logical instantiation of an adapter, through which internal objects are created.
	* Is the exclusive owner of all internal objects created from it.
	*/
	export struct Device
	{
		/** the adapter from which this device was created */
		Adapter* adapter;

		/** the features which can be used on this device */

	};
}