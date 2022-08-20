export module Tracer.Interactable:Light.AreaLight;
import :Light;
import Math.Vector;
import Tracer.Spectrum;

namespace SIByL::Tracer
{
	export struct AreaLight :public Light
	{
		virtual auto L(Interaction const& intr, Math::vec3 const& w) const noexcept -> Spectrum = 0;
	};
}