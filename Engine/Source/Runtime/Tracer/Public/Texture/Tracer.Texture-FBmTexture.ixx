export module Tracer.Texture:FBmTexture;
import Tracer.Interactable;

namespace SIByL::Tracer
{
	export template <class T>
		struct FBmTexture :public Texture<T>
	{
		virtual auto evaluate(SurfaceInteraction const&) const noexcept -> T override;
	};
}