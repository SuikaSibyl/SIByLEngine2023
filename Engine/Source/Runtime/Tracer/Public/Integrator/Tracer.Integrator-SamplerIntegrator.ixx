export module Tracer.Integrator:SamplerIntegrator;
import :Integrator;
import GFX.Scene;
import GFX.Camera;
import Tracer.Sampler;

namespace SIByL::Tracer
{
	export struct SamplerIntegrator :public Integrator
	{
		SamplerIntegrator(GFX::Camera* camera, Sampler* sampler)
			: camera(camera), sampler(sampler) {}

		virtual auto preprocess(GFX::Scene const& scene) noexcept -> void {}
		virtual auto render(GFX::Scene const& scene) noexcept -> void override;

	private:
		Sampler* sampler;
		GFX::Camera* camera;
	};

	auto SamplerIntegrator::render(GFX::Scene const& scene) noexcept -> void
	{
		preprocess(scene);
		// render image tiles in parallel
		// - Compute number of tiles, nTiles, to use for parallel rendering 28
		
		//ParallelFor2D(
		//	[&](Point2i tile) {
		//			Render section of image corresponding to tile 28
		//	}, nTiles);
		
		// save final image after rendering
	}

}