export module Tracer.Filter:Filter;
import Math.Vector;
import Math.Geometry;

namespace SIByL::Tracer
{
	export struct Filter
	{
		Math::vec2 radius, invRadius;

		virtual auto evaluate(Math::point2 const& p) noexcept -> float = 0;
	};
}