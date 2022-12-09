export module Tracer.Filter:LanczosSincFilter;
import :Filter;
import SE.Math.Geometric;

namespace SIByL::Tracer
{
	export struct LanczosSincFilter :public Filter
	{
		virtual auto evaluate(Math::point2 const& p) const noexcept -> float override;
	};
}