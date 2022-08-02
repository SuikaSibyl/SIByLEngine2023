module;
#include <cstdint>
module Tracer.BxDF:BxDF;
import Tracer.BxDF;
import Math.Vector;
import Tracer.Spectrum;

namespace SIByL::Tracer
{
	auto BxDF::matchFlags(Type t) const noexcept -> bool {
		return ((uint32_t)type & (uint32_t)t) == (uint32_t)type;
	}
}