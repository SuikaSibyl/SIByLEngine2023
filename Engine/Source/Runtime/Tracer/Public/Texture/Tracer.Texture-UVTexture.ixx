export module Tracer.Texture:UVTexture;
import :Texture;
import Tracer.Interactions;

namespace SIByL::Tracer
{
	export template <class T>
	struct UVTexture :public Texture<T>
	{
		virtual auto evaluate(SurfaceInteraction const&) const noexcept -> T override;
	};
}