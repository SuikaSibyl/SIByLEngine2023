module;
#include <string>
export module Tracer.Ray:Medium;
import :Ray;
import :Sampler;
import Tracer.Spectrum;

namespace SIByL::Tracer
{
	export struct Medium
	{
		virtual auto Tr(Ray const& ray, Sampler& sampler) const noexcept -> Spectrum = 0;
	};

	export inline auto getMediumScatteringProperties(std::string const& name, Spectrum* sigma_a, Spectrum* sigma_s) noexcept -> bool {
		// TODO
		return false;
	}
}