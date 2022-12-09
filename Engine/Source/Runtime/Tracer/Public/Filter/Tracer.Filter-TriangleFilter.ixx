export module Tracer.Filter:TriangleFilter;
import :Filter;
import SE.Math.Geometric;

namespace SIByL::Tracer
{
	export struct TriangleFilter :public Filter
	{
		virtual auto evaluate(Math::point2 const& p) const noexcept -> float override;
	};
}