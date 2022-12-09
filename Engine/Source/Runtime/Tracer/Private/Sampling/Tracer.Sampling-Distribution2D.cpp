module;
#include <vector>
#include <memory>
module Tracer.Sampling:Distribution2D;
import Tracer.Sampling;
import SE.Math.Misc;
import SE.Math.Geometric;

namespace SIByL::Tracer
{
	Distribution2D::Distribution2D(float const* func, int nu, int nv) {
		for (int v = 0; v < nv; ++v) {
			// Compute conditional sampling distribution for ˜v
			pConditionalV.emplace_back(new Distribution1D(&func[v * nu], nu));
		}
		// Compute marginal sampling distribution p[˜v]
		std::vector<float> marginalFunc;
		for (int v = 0; v < nv; ++v)
			marginalFunc.push_back(pConditionalV[v]->funcInt);
		pMarginal.reset(new Distribution1D(&marginalFunc[0], nv));
	}

	auto Distribution2D::sampleContinuous(Math::point2 const& u, float* pdf) const noexcept -> Math::point2 {
		float pdfs[2];
		int v;
		float d1 = pMarginal->sampleContinuous(u[1], &pdfs[1], &v);
		float d0 = pConditionalV[v]->sampleContinuous(u[0], &pdfs[0]);
		*pdf = pdfs[0] * pdfs[1];
		return Math::point2(d0, d1);
	}

	auto Distribution2D::pdf(Math::point2 const& p) const noexcept -> float {
		int iu = Math::clamp(int(p[0] * pConditionalV[0]->count()), 0, pConditionalV[0]->count() - 1);
		int iv = Math::clamp(int(p[1] * pMarginal->count()), 0, pMarginal->count() - 1);
		return pConditionalV[iv]->func[iu] / pMarginal->funcInt;
	}
}