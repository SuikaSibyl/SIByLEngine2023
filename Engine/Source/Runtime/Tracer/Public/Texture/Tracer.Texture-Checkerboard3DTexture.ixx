export module Tracer.Texture:Checkerboard3DTexture;
import :Texture;
import Tracer.Interactable;

namespace SIByL::Tracer
{
	export template <class T>
		struct Checkerboard3DTexture :public Texture<T>
	{
		virtual auto evaluate(SurfaceInteraction const&) const noexcept -> T override;
	};
}