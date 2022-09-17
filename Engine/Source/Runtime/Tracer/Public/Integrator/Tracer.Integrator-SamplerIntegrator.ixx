export module Tracer.Integrator:SamplerIntegrator;
import :Integrator;
import Core.Memory;
import Math.Limits;
import Parallelism.Parallel;
import Tracer.Ray;
import Tracer.Camera;
import Tracer.Sampler;
import Tracer.Spectrum;
import Tracer.Interactable;

namespace SIByL::Tracer
{
	/**
	* The rendering process is driven by a stream of samples from a Sampler.
	*/
	export struct SamplerIntegrator :public Integrator
	{
		SamplerIntegrator(Camera const* camera, Sampler* sampler)
			: camera(camera), sampler(sampler) {}

		/**
		* Optinally called after Scene has been fully initialized and gives the integrator a chance 
		* to do scene-dependent computation, such as allocating additional data structures
		* that are dependent on the number of lights in the scene, or precomputing a rough
		* reprensentation of the distribution of radiance in the scene.
		*/
		virtual auto preprocess(Scene const& scene) noexcept -> void {}

		virtual auto Li(RayDifferential const& ray, Scene const& scene, Sampler& sampler, Core::MemoryArena& arena, int depth = 0) const noexcept -> Spectrum = 0;
		/** the rendering loop */
		virtual auto render(Scene const& scene) noexcept -> void override;

		auto specularReflect(RayDifferential const& ray, SurfaceInteraction const& isect, Scene const& scene, Sampler& sampler, Core::MemoryArena& arena, int depth) const noexcept -> Spectrum;
		auto specularTransmit(RayDifferential const& ray, SurfaceInteraction const& isect, Scene const& scene, Sampler& sampler, Core::MemoryArena& arena, int depth) const noexcept -> Spectrum;
	private:
		/**
		* Responsible for choosing the points on the image plane from which rays are traced,
		* and supply the sample positions used by integrators for estimating the value of the
		* light transport integral.
		*/
		Sampler* sampler;

		/** Control the viewing and lens parameters such as position, orientation, focus, and field of view. */
		Camera const* camera;
	};
}