export module Tracer.Texture:MixTexture;
import Tracer.Interactable;

namespace SIByL::Tracer
{
	export template <class T>
	struct MixTexture :public Texture<T>
	{
		virtual auto evaluate(SurfaceInteraction const&) const noexcept -> T override;
	};
}