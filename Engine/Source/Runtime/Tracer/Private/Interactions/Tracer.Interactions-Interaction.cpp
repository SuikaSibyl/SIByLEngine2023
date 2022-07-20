module Tracer.Interactions:Interaction;
import Tracer.Interactions;
import Math.Vector;
import Math.Geometry;
import Tracer.Medium;

namespace SIByL::Tracer
{
	auto Interaction::isSurfaceInteraction() const noexcept -> bool {
		return n != Math::normal3();
	}
}