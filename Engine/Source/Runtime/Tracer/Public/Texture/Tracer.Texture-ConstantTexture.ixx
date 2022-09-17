export module Tracer.Texture:ConstantTexture;
import Tracer.Interactable;

namespace SIByL::Tracer
{
	export template <class T>
		struct ConstantTexture :public Texture<T>
	{
		virtual auto evaluate(SurfaceInteraction const&) const noexcept -> T override;
	};
}