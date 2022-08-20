export module Tracer.Texture:MarbleTexture;
import :Texture;
import Tracer.Interactable;

namespace SIByL::Tracer
{
	export template <class T>
		struct MarbleTexture :public Texture<T>
	{
		virtual auto evaluate(SurfaceInteraction const&) const noexcept -> T override;
	};
}