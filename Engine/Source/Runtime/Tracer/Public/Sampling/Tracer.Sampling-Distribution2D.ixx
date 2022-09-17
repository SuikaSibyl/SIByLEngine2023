module;
#include <vector>
#include <memory>
export module Tracer.Sampling:Distribution2D;
import :Distribution1D;
import Math.Geometry;

namespace SIByL::Tracer
{
	struct Distribution2D
	{
		Distribution2D(float const* func, int nu, int nv);

		auto sampleContinuous(Math::point2 const& u, float* pdf) const noexcept -> Math::point2;

		auto pdf(Math::point2 const& p) const noexcept -> float;

	private:
		std::vector<std::unique_ptr<Distribution1D>> pConditionalV;
		std::unique_ptr<Distribution1D> pMarginal;
	};
}