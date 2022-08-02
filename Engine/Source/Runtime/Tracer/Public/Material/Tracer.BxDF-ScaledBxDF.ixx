module;
export module Tracer.BxDF:ScaledBxDF;
import :BxDF;
import Math.Vector;
import Math.Geometry;
import Tracer.Spectrum;

namespace SIByL::Tracer
{
	/**
	* ScaledBxDF takes a given BxDF and scale its contribution with a Spectrum value.
	* It is used by MixMaterial, which creates BSDFs based on a weighted combination 
	* of two other materials.
	*/
	export struct ScaledBxDF :public BxDF
	{
	private:
		BxDF* bxdf;
		Spectrum scale;
	};
}