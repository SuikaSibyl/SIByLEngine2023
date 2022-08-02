export module Tracer.Texture:Checkerboard2DTexture;
import :Texture;
import Tracer.Interactions;

namespace SIByL::Tracer
{
	export template <class T>
		struct Checkerboard2DTexture :public Texture<T>
	{
		virtual auto evaluate(SurfaceInteraction const&) const noexcept -> T override;
	};
}