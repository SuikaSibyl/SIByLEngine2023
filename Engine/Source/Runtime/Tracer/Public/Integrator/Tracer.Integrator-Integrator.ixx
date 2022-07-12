export module Tracer.Integrator:Integrator;
import GFX.Scene;

namespace SIByL::Tracer
{
	export struct Integrator
	{
		virtual auto render(GFX::Scene const& scene) noexcept -> void = 0;
	};
}