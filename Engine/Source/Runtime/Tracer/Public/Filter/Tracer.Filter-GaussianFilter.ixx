export module Tracer.Filter:GaussianFilter;
import :Filter;
import SE.Math.Geometric;

namespace SIByL::Tracer
{
	export struct GaussianFilter :public Filter
	{
		virtual auto evaluate(Math::point2 const& p) const noexcept -> float override;
	};
}