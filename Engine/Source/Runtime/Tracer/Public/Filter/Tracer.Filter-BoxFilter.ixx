export module Tracer.Filter:BoxFilter;
import :Filter;
import Math.Vector;
import Math.Geometry;

namespace SIByL::Tracer
{
	export struct BoxFilter :public Filter
	{
		virtual auto evaluate(Math::point2 const& p) noexcept -> float override;
	};
}