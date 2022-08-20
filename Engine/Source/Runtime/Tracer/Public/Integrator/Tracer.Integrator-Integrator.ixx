export module Tracer.Integrator:Integrator;
import Tracer.Interactable;

namespace SIByL::Tracer
{
	export struct Integrator
	{
		virtual auto render(Scene const& scene) noexcept -> void = 0;
	};
}