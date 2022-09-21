export module Tracer.Texture:ConstantTexture;
import Tracer.Interactable;

namespace SIByL::Tracer
{
	/** Returns the same value no matter where it is evaluated. */
	export template <class T>
	struct ConstantTexture :public Texture<T> {
		/** Initialize with the constant value to be returned anywhere. */
		ConstantTexture(T const& value) :value(value) {}
		/** Evaluate texture's value at the point being shaded of the SurfaceInteraction.*/
		virtual auto evaluate(SurfaceInteraction const&) const noexcept -> T override { return value; }
	private:
		/** The constant value */
		T value;
	};
}