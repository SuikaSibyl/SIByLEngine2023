export module Tracer.Integrator:WhittedIntegrator;
import :Integrator;
import :SamplerIntegrator;
import Core.Memory;
import Parallelism.Parallel;
import Tracer.Ray;
import Tracer.Camera;
import Tracer.Sampler;
import Tracer.Spectrum;
import Tracer.Interactable;

namespace SIByL::Tracer
{
	export struct WhittedIntegrator :public SamplerIntegrator
	{
		WhittedIntegrator(int maxDepth, Camera const* camera, Sampler* sample);

		virtual auto Li(RayDifferential const& ray, Scene const& scene, Sampler& sampler, Core::MemoryArena& arena, int depth = 0) const noexcept -> Spectrum override;

	private:
		int const maxDepth;
	};
}