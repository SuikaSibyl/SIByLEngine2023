module;
#include <cmath>
export module Tracer.Medium:HomogeneousMedium;
import SE.Math.Misc;
import Tracer.Ray;
import Tracer.Spectrum;

namespace SIByL::Tracer
{
	export struct HomogeneousMedium :public Medium
	{
		HomogeneousMedium(Spectrum const& sigma_a, Spectrum const& sigma_s, float g)
			:sigma_a(sigma_a), sigma_s(sigma_s), sigma_t(sigma_s + sigma_a), g(g) {}

		virtual auto Tr(Ray const& ray, Sampler& sampler) const noexcept -> Spectrum override {
			return exp(-sigma_t * std::min(ray.tMax * ray.d.length(), Math::float_max));
		}

	private:
		Spectrum const sigma_a, sigma_s, sigma_t;
		float const g;
	};
}