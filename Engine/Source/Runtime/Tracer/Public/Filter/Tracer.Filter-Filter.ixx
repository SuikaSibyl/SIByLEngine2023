export module Tracer.Filter:Filter;
import SE.Math.Geometric;

namespace SIByL::Tracer
{
	export struct Filter
	{
		Filter(Math::vec2 const& radius)
			: radius(radius)
			, invRadius(Math::vec2{ 1.f / radius.x,1.f / radius.y })
		{}

		virtual auto evaluate(Math::point2 const& p) const noexcept -> float = 0;

		Math::vec2 const radius, invRadius;
	};
}