export module Tracer.Filter:BoxFilter;
import :Filter;
import Math.Vector;
import Math.Geometry;

namespace SIByL::Tracer
{
	export struct BoxFilter :public Filter
	{
		BoxFilter(Math::vec2 const& radius) :Filter(radius) {}

		virtual auto evaluate(Math::point2 const& p) const noexcept -> float override { return 1; }
	};
}