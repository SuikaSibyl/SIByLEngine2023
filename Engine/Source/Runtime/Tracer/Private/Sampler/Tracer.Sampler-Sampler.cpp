module;
#include <cstdint>
module Tracer.Sampler:Sampler;
import Tracer.Sampler;
import Math.Geometry;
import Tracer.Camera;

namespace SIByL::Tracer
{
	auto Sampler::getCameraSample(Math::ipoint2 const& pRaster) noexcept -> CameraSample {
		CameraSample cs;
		cs.pFilm = (Math::point2)pRaster + get2D();
		cs.time = get1D();
		cs.pLens = get2D();
		return cs;
	}

	auto Sampler::request1DArray(int n) noexcept -> void {
	}

	auto Sampler::request2DArray(int n) noexcept -> void {
	}

	auto Sampler::roundCount(int n) const noexcept -> int {
		return n;
	}
}