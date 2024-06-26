export module Tracer.Filter:MitchellFilter;
import :Filter;
import SE.Math.Geometric;

namespace SIByL::Tracer
{
	export struct MitchellFilter :public Filter
	{
		virtual auto evaluate(Math::point2 const& p) const noexcept -> float override;
	};
}