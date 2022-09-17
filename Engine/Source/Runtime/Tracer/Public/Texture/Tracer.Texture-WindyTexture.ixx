export module Tracer.Texture:WindyTexture;
import Tracer.Interactable;

namespace SIByL::Tracer
{
	export template <class T>
	struct WindyTexture :public Texture<T>
	{
		virtual auto evaluate(SurfaceInteraction const&) const noexcept -> T override;
	};
}