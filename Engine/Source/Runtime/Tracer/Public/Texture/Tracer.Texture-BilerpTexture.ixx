export module Tracer.Texture:BilerpTexture;
import :Texture;
import Tracer.Interactable;

namespace SIByL::Tracer
{
	export template <class T>
		struct BilerpTexture :public Texture<T>
	{
		virtual auto evaluate(SurfaceInteraction const&) const noexcept -> T override;
	};
}