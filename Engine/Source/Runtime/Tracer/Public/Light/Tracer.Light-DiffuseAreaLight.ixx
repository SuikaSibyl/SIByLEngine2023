module;
#include <string>
export module Tracer.Light:DiffuseAreaLight;
import SE.Math.Misc;
import SE.Math.Geometric;
import Tracer.Interactable;
import Tracer.Medium;
import Tracer.Spectrum;
import Tracer.Shape;

namespace SIByL::Tracer
{
	export struct DiffuseAreaLight :public AreaLight
	{
		virtual auto L(Interaction const& intr, Math::vec3 const& w) const noexcept -> Spectrum override {
			return Math::dot(intr.n, w) > 0.f ? Lemit : Spectrum(0.f);
		}

		virtual auto power() const noexcept -> Spectrum override {
			return Lemit * area * Math::float_Pi;
		}

	protected:
		Spectrum const Lemit;
		Shape* shape;
		float const area;
	};
}