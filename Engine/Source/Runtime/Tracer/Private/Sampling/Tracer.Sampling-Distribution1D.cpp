module;
#include <vector>
module Tracer.Sampling:Distribution1D;
import Tracer.Sampling;
import SE.Math.Misc;

namespace SIByL::Tracer
{
	Distribution1D::Distribution1D(float const* f, int n)
		: func(f, f + n), cdf(n + 1) 
	{
		// Compute integral of step function at x_i
		cdf[0] = 0;
		for (int i = 1; i < n + 1; ++i)
			cdf[i] = cdf[i - 1] + func[i - 1] / n;
		// Transform step function integral into CDF
		funcInt = cdf[n];
		if (funcInt == 0) {
			for (int i = 1; i < n + 1; ++i)
				cdf[i] = float(i) / float(n);
		}
		else
			for (int i = 1; i < n + 1; ++i)
				cdf[i] /= funcInt;
	}

	auto Distribution1D::sampleContinuous(float u, float* pdf, int* off) const noexcept -> float {
		// Find surrounding CDF segmentsand offset
		int offset = Math::findInterval(cdf.size(),
			[&](int index) { return cdf[index] <= u; });
		if (off) *off = offset;
		// Compute offset along CDF segment
		float du = u - cdf[offset];
		if ((cdf[offset + 1] - cdf[offset]) > 0)
			du /= (cdf[offset + 1] - cdf[offset]);
		// Compute PDF for sampled offset
		if (pdf) *pdf = func[offset] / funcInt;
		// Return x ∈[0, 1) corresponding to sample
		return (offset + du) / count();
	}

	auto Distribution1D::sampleDiscrete(float u, float* pdf, float* uRemapped) const noexcept -> int {
		// Find surrounding CDF segmentsand offset
		int offset = Math::findInterval(cdf.size(),
			[&](int index) { return cdf[index] <= u; });
		if (pdf) *pdf = func[offset] / (funcInt * count());
		if (uRemapped)
			*uRemapped = (u - cdf[offset]) / (cdf[offset + 1] - cdf[offset]);
		return offset;
	}

	auto Distribution1D::discretePDF(int index) const noexcept -> float {
		return func[index] / (funcInt * count());
	}
}