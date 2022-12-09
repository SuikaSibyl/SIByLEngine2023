export module Tracer.Integrator:WhittedIntegrator;
import :Integrator;
import :SamplerIntegrator;
import SE.Core.Memory;
import SE.Math.Misc;
import SE.Parallelism;
import Tracer.Ray;
import Tracer.Camera;
import Tracer.Sampler;
import Tracer.Spectrum;
import Tracer.Interactable;

namespace SIByL::Tracer
{
	/**
	* A simple implementation of SamplerIntegrator with Whitted's ray-tracing algorithm.
	* It accurately computes reflected and transmitted light from specular surfaces, but
	* it doesn't account for other types of indirect lighting effects.
	*/
	export struct WhittedIntegrator :public SamplerIntegrator
	{
		WhittedIntegrator(int maxDepth, Camera const* camera, Sampler* sample);

		/** Given a ray, determine the amount of light arriving at the image plane along the ray. */
		virtual auto Li(RayDifferential const& ray, Scene const& scene, Sampler& sampler,
			Core::MemoryArena& arena, int depth = 0) const noexcept -> Spectrum override;

	private:
		int const maxDepth;
	};
}