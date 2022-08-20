export module Tracer.Texture:UVTexture;
import :Texture;
import Tracer.Interactable;

namespace SIByL::Tracer
{
	export template <class T>
	struct UVTexture :public Texture<T>
	{
		virtual auto evaluate(SurfaceInteraction const&) const noexcept -> T override;
	};
}