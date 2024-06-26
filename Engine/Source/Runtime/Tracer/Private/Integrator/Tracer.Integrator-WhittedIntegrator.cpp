module;
#include <cmath>
module Tracer.Integrator:WhittedIntegrator;
import Tracer.Integrator;
import SE.Core.Memory;
import SE.Math.Geometric;
import SE.Parallelism;
import Tracer.Ray;
import Tracer.Camera;
import Tracer.Sampler;
import Tracer.Spectrum;
import Tracer.Interactable;

namespace SIByL::Tracer
{
	WhittedIntegrator::WhittedIntegrator(int maxDepth, Camera const* camera, Sampler* sampler)
		: SamplerIntegrator(camera, sampler), maxDepth(maxDepth) {}

	auto WhittedIntegrator::Li(RayDifferential const& ray, Scene const& scene, Sampler& sampler, 
		Core::MemoryArena& arena, int depth) const noexcept -> Spectrum 
	{
		Spectrum L(0.f);
		// find closet ray intersection or return background radiance
		SurfaceInteraction isect;
		if (!scene.intersect(ray, &isect)) {
			// ray may carry radiance due to light sources 
			// that don't have associated geometry, like InfiniteAreaLight
			for (auto const& light : scene.lights)
				L += light->Le(ray);
			return L;
		}
		// compute emitted and reflected light at ray intersection point
		//  initialize common variables for Whitted integrator
		Math::normal3 const n = isect.shading.n;
		Math::vec3 const wo = isect.wo;
		//  compute scattering functions for surface interaction
		isect.computeScatteringFunctions(ray, arena);
		//  compute emitted light if ray hit an area light source
		L += isect.Le(wo);
		//  add contribution of each light source
		for (auto const& light : scene.lights) {
			Math::vec3 wi;
			float pdf;
			VisibilityTester visibility;
			Spectrum Li = light->sample_Li(isect, sampler.get2D(), &wi, &pdf, &visibility);
			if (Li.isBlack() || pdf == 0) continue;
			Spectrum f = isect.bsdf->f(wo, wi);
			if (!f.isBlack() && visibility.unoccluded(scene)) {
				L += f * Li * Math::absDot(wi, n) / pdf;
			}				
			//Spectrum normal;
			//normal[0] = std::abs(wi.x);
			//normal[1] = std::abs(wi.y);
			//normal[2] = std::abs(wi.z);
			//L += normal;
		}
		if (depth + 1 < maxDepth) {
			// trace rays for specular reflection and refraction
			L += specularReflect(ray, isect, scene, sampler, arena, depth);
			L += specularTransmit(ray, isect, scene, sampler, arena, depth);
		}
		return L;
	}

}