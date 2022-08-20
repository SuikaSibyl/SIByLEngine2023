export module Tracer.Filter:GaussianFilter;
import :Filter;
import Math.Vector;
import Math.Geometry;

namespace SIByL::Tracer
{
	export struct GaussianFilter :public Filter
	{
		virtual auto evaluate(Math::point2 const& p) const noexcept -> float override;
	};
}