export module Tracer.Texture:ImageTexture;
import Tracer.Interactable;

namespace SIByL::Tracer
{
	export template <class T>
		struct ImageTexture :public Texture<T>
	{
		virtual auto evaluate(SurfaceInteraction const&) const noexcept -> T override;
	};
}