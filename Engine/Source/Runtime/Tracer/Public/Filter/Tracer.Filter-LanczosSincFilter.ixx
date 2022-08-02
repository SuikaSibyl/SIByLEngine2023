export module Tracer.Filter:LanczosSincFilter;
import :Filter;
import Math.Vector;
import Math.Geometry;

namespace SIByL::Tracer
{
	export struct LanczosSincFilter :public Filter
	{
		virtual auto evaluate(Math::point2 const& p) noexcept -> float override;
	};
}