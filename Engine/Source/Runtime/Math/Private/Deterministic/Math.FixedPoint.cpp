module;
#include <cstdint>
#include <functional>
#include <intrin.h>
module Math.FixedPoint;

namespace SIByL::Math
{
	// -----------------------
	// fixed32_t impl
	// -----------------------

	fixed32_t::fixed32_t(int32_t value, bool useRawInit)
		:raw([&]() {return useRawInit ? value : value << fractionBitCount; }())
	{}

	auto fixed32_t::operator+(fixed32_t const& rhs) const -> fixed32_t {
		return fixed32_t(this->raw + rhs.raw, true);
	}

	auto fixed32_t::operator-(fixed32_t const& rhs) const -> fixed32_t {
		return fixed32_t(this->raw - rhs.raw, true);
	}

	auto fixed32_t::operator*(fixed32_t const& rhs) const -> fixed32_t {
		return fixed32_t((int64_t(this->raw) * int64_t(rhs.raw)) >> fractionBitCount, true);
	}

	auto fixed32_t::operator/(fixed32_t const& rhs) const -> fixed32_t {
		return fixed32_t((int64_t(this->raw) << fractionBitCount) / int64_t(rhs.raw), true);
	}

	fixed64_t::fixed64_t(int64_t value, bool useRawInit)
		:raw([&]() {return useRawInit ? value : value << fractionBitCount; }())
	{}


	// -----------------------
	// fixed64_t impl
	// -----------------------

	auto fixed64_t::operator+(fixed64_t const& rhs) const->fixed64_t {
		return fixed64_t(this->raw + rhs.raw, true);
	}

	auto fixed64_t::operator-(fixed64_t const& rhs) const->fixed64_t {
		return fixed64_t(this->raw - rhs.raw, true);
	}

	auto fixed64_t::operator*(fixed64_t const& rhs) const->fixed64_t {
		int64_t product128_high = 0;
		int64_t const product128_low = _mul128(this->raw, rhs.raw, &product128_high);
		return fixed64_t((product128_high << wholeBitCount) | (product128_low >> fractionBitCount), true);
	}

	auto fixed64_t::operator/(fixed64_t const& rhs) const->fixed64_t {
		int64_t const highDividend = int64_t(this->raw) >> wholeBitCount;
		int64_t const lowDividend = int64_t(this->raw) << fractionBitCount;
		int64_t remainder64 = 0;
		int64_t const quotient64 = _div128(highDividend, lowDividend, rhs.raw, &remainder64);
		return fixed64_t(quotient64, true);
	}
}