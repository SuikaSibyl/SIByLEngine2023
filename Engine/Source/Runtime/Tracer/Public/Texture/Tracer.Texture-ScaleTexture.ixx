export module Tracer.Texture:ScaleTexture;
import :Texture;
import Tracer.Interactable;

namespace SIByL::Tracer
{
	export template <class T>
	struct ScaleTexture :public Texture<T>
	{
		virtual auto evaluate(SurfaceInteraction const&) const noexcept -> T override;
	};
}