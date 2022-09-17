module;
#include <cmath>
export module Tracer.Medium;
import Math.Vector;
import Math.Trigonometric;
import Tracer.Ray;

namespace SIByL::Tracer
{
	export struct PhaseFunction {
		virtual auto p(Math::vec3 const& wo, Math::vec3 const& wi) const noexcept -> float = 0;
	};

	inline auto phaseHG(float cosTheta, float g) {
		float denom = 1 + g * g + 2 * g * cosTheta;
		return Math::float_Inv4Pi * (1 - g * g) / (denom * std::sqrt(denom));
	}

	export struct HenyeyGreenstein :public PhaseFunction {
		virtual auto p(Math::vec3 const& wo, Math::vec3 const& wi) const noexcept -> float override {
			return phaseHG(Math::dot(wo, wi), g);
		}

	private:
		float const g;
	};

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

		Medium const* inside, * outside;
	};
}