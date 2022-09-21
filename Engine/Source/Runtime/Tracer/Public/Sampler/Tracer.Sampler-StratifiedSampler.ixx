export module Tracer.Sampler:StratifiedSampler;
import :PixelSampler;
import Math.Geometry;
import Math.Random;
import Tracer.Base;

namespace SIByL::Tracer
{
	export struct StratifiedSampler :public PixelSampler
	{
		StratifiedSampler(int xPixelSamples, int yPixelSamples, bool jitterSamples, int nSampledDimensions);

		virtual auto startPixel(Math::ipoint2 const& p) noexcept -> void override;

		/** Get a new instance of an initial Sampler */
		virtual auto clone(int seed) noexcept -> Scope<Sampler> override;
		
		int const xPixelSamples, yPixelSamples;
		bool const jitterSamples;
	};

	export inline auto stratifiedSample1D(float* samp, int nSamples, Math::RNG& rng, bool jitter) noexcept -> void;
	export inline auto stratifiedSample2D(Math::point2* samp, int nx, int ny, Math::RNG& rng, bool jitter) noexcept -> void;
}