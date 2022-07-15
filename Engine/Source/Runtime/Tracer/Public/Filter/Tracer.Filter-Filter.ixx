export module Tracer.Filter:Filter;
import Math.Vector;

namespace SIByL::Tracer
{
	export struct Filter
	{
		Math::vec2 radius, invRadius;
	};
}