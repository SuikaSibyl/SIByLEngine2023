export module Tracer.Medium:MediumInterface;
import :Medium;

namespace SIByL::Tracer
{
	/*
	* Geometric Primitive represents the boundary between two different types of scattering media.
	* It holds a MediumInterface, which in turn holds pointers to one Medium for interior
	* and exterior medium.
	* A nullptr could be used to indicate a vaccum.
	*/
	export struct MediumInterface
	{
		MediumInterface() :inside(nullptr), outside(nullptr) {}
		MediumInterface(Medium const* medium) :inside(medium), outside(medium) {}
		MediumInterface(Medium const* inside, Medium const* outside) :inside(inside), outside(outside) {}

		/** checks whether a particular instance marks a transition between two distinct media*/
		auto isMediumTransition() const noexcept -> bool { return inside != outside; }

		Medium const* inside, *outside;
	};
}