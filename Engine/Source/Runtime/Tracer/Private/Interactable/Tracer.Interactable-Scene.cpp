module;
#include <vector>
module Tracer.Interactable:Scene;
import Tracer.Interactable;
import Math.Geometry;
import Tracer.Ray;

namespace SIByL::Tracer
{
	Scene::Scene(Primitive* aggregate, std::vector<Light*> const& lights)
		:lights(lights), aggregate(aggregate)
	{
		worldBound = aggregate->worldBound();
		for (auto const& light : lights)
			light->preprocess(*this);
	}
	
	auto Scene::getWorldBound() const noexcept -> Math::bounds3 const& {
		return worldBound;
	}
	
	auto Scene::intersect(Ray const& ray, SurfaceInteraction* isect) const noexcept -> bool {
		return aggregate->intersect(ray, isect);
	}
	
	auto Scene::intersectP(Ray const& ray) const noexcept -> bool {
		return aggregate->intersectP(ray);
	}
}