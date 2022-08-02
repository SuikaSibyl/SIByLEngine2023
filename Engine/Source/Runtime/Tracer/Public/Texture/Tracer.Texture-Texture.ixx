export module Tracer.Texture:Texture;
import Tracer.Interactions;

namespace SIByL::Tracer
{
	export template <class T>
	struct Texture
	{
		virtual auto evaluate(SurfaceInteraction const&) const noexcept -> T = 0;
	};
}