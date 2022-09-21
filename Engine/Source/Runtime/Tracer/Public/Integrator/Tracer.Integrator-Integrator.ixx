export module Tracer.Integrator:Integrator;
import Tracer.Interactable;

namespace SIByL::Tracer
{
	/**
	* An abstract base for rendering an image or more generally
	* by computing the radiance on each pixel using various strategies.
	*/
	export struct Integrator
	{
		/** Get a reference to the Scene to compute an image of the scene or more generally */
		virtual auto render(Scene const& scene) noexcept -> void = 0;
	};
}