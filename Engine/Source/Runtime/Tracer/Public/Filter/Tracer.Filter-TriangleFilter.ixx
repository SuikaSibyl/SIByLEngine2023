export module Tracer.Filter:TriangleFilter;
import :Filter;
import Math.Vector;
import Math.Geometry;

namespace SIByL::Tracer
{
	export struct TriangleFilter :public Filter
	{
		virtual auto evaluate(Math::point2 const& p) noexcept -> float override;
	};
}