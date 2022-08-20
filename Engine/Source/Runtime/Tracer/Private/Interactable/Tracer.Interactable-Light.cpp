module Tracer.Interactable:Light;
import Math.Vector;
import Math.Geometry;
import Math.Transform;
import Tracer.Medium;
import Tracer.Spectrum;
import Tracer.Ray;
import Tracer.Interactable;

namespace SIByL::Tracer
{
	auto VisibilityTester::unoccluded(Scene const& scene) const noexcept -> bool {
		return !scene.intersectP(_p0.spawnRayTo(_p1));
	}
}