module;
#include <algorithm>
export module Tracer.Integrator:PathIntegrator;
import :Integrator;
import :SamplerIntegrator;
import :Utility;
import SE.Core.Memory;
import SE.Math.Misc;
import Tracer.Ray;
import Tracer.Sampler;
import Tracer.Interactable;

namespace SIByL::Tracer
{
	export struct PathIntegrator : public SamplerIntegrator
	{
		PathIntegrator(int maxDepth, Camera const* camera, Sampler* sampler)
			:SamplerIntegrator(camera, sampler), maxDepth(maxDepth) {}

		virtual auto Li(RayDifferential const& r, Scene const& scene, Sampler& sampler,
			Core::MemoryArena& arena, int depth = 0) const noexcept -> Spectrum override 
		{
			Spectrum		L(0.f);					// L holds the radiance value from the running total of ΣP(p_i).
			Spectrum		beta(1.f);				// beta holds the path throughput weight.
			RayDifferential ray(r);					// ray holds the next ray to be traced to extend the path one more vertex.
			bool			specularBounce = false; // records if the last outgoing path direction sampled was due to specular reflection.
			// exntending the path and do integrate
			for (int bounces = 0;; ++bounces) {
				// Find next path vertex and accumulate contribution
				//  Intersect ray with sceneand store intersection in isect
				SurfaceInteraction isect;
				bool const foundIntersection = scene.intersect(ray, &isect);
				//  Possibly add emitted light at intersection
				//		The emission is usually ignored, since the previous path vertex performed a
				//		direct illumination estimate that already accounted for its effect. There are
				//		two exceptions:
				//		1. initial intersection point of camera rays
				//		2. sampled direction from the last path vertex was from a specular BSDF component
				if (bounces == 0 || specularBounce) {
					// Add emitted light at path vertex or from the environment
					if (foundIntersection)				// + radiance emitted by the current path vertex
						L += beta * isect.Le(-ray.d);
					else								// + radiance emitted by infinite area light sources
						for (auto const& light : scene.lights)
							L += beta * light->Le(ray);
				}
				//  Terminate path if ray escaped or maxDepth was reached
				if (!foundIntersection || bounces >= maxDepth)
					break;
				//  Compute scattering functions and skip over medium boundaries
				isect.computeScatteringFunctions(ray, arena, true);
				if (!isect.bsdf) { // if the current surface has no effect on light, like media divider
					ray = isect.spawnRay(ray.d); // simply skips over such surfaces
					bounces--;
					continue;
				}
				//  Sample illumination from lights to find path contribution
				L += beta * uniformSampleOneLight(isect, scene, arena, sampler, false);
				//  Sample BSDF to get new path direction
				Math::vec3 wo = -ray.d, wi;
				float pdf;
				BxDF::Type flags;
				Spectrum f = isect.bsdf->sample_f(wo, &wi, sampler.get2D(), &pdf, BxDF::BSDF_ALL, &flags);
				if (f.isBlack() || pdf == 0.f)
					break;
				beta *= f * Math::absDot(wi, isect.shading.n) / pdf;
				specularBounce = (flags & BxDF::BSDF_SPECULAR) != 0;
				ray = isect.spawnRay(wi);
				//  Account for subsurface scattering, if applicable
				//  Possibly terminate the path with Russian roulette
				if (bounces > 3) {
					float q = std::max((float).05, 1 - beta.y());
					if (sampler.get1D() < q)
						break;
					beta /= 1 - q;
				}
			}
			return L;
		}

	private:
		/** works besides Russian roulette */
		int const maxDepth;
	};
}