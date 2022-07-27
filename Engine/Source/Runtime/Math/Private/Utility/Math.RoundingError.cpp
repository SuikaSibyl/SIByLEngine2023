module;
#include <cstdint>
#include <cmath>
module Math.RoundingError;

namespace SIByL::Math
{
	inline auto floatToBits(float f) noexcept -> uint32_t {
		uint32_t bits = *reinterpret_cast<uint32_t*>(&f);
		return bits;
	}

	inline auto bitsToFloat(uint32_t b) noexcept -> float {
		float f = *reinterpret_cast<float*>(&b);
		return f;
	}

	inline auto nextFloatUp(float v) noexcept -> float {
		// Hanlde infinity and negative zero
		if (std::isinf(v) && v > 0.f)
			return v;
		if (v == -0.f)
			return 0.f;
		// Advance v to next higher float
		uint32_t ui = floatToBits(v);
		if (v >= 0) ++ui;
		else        --ui;
		return bitsToFloat(ui);
	}

	inline auto nextFloatDown(float v) noexcept -> float {
		// Hanlde infinity and positive zero
		if (std::isinf(v) && v < 0.f)
			return v;
		if (v == +0.f)
			return -0.f;
		// Advance v to next higher float
		uint32_t ui = floatToBits(v);
		if (v >= 0) --ui;
		else        ++ui;
		return bitsToFloat(ui);
	}
	
	inline auto gamma(int n) noexcept -> float {
		return (n * MachineEpsilon) / (1 - n * MachineEpsilon);
	}

	FloatWithErrBound::FloatWithErrBound(float v) : v(v) {
		low = high = v;
#ifdef _DEBUG
		ld = v;
#endif
	}

	FloatWithErrBound::FloatWithErrBound(float v, float err) 
		: v(v)
	{
		if (err == 0.f)
			low = high = v;
		else {
			low = nextFloatDown(v - err);
			high = nextFloatUp(v + err);
		}
#ifdef _DEBUG
		ld = v;
#endif
	}

	auto FloatWithErrBound::operator+(FloatWithErrBound f) const->FloatWithErrBound {
		FloatWithErrBound r;
		r.v = v + f.v;
		r.low = nextFloatDown(lowerBound() + f.lowerBound());
		r.high = nextFloatUp(upperBound() + f.upperBound());
#ifdef _DEBUG
		r.ld = ld + f.ld;
#endif
		return r;
	}

	auto FloatWithErrBound::operator-(FloatWithErrBound f) const->FloatWithErrBound {
		FloatWithErrBound r;
		r.v = v - f.v;
		r.low = nextFloatDown(lowerBound() - f.upperBound());
		r.high = nextFloatUp(upperBound() - f.lowerBound());
#ifdef _DEBUG
		r.ld = ld - f.ld;
#endif
		return r;
	}

	auto FloatWithErrBound::operator*(FloatWithErrBound f) const->FloatWithErrBound {
		FloatWithErrBound r;
		r.v = v * f.v;
		float prod[4] = {
			lowerBound() * f.lowerBound(),
			lowerBound() * f.upperBound(),
			upperBound() * f.lowerBound(),
			upperBound() * f.upperBound(),
		};
		r.low = nextFloatDown(std::min(
			std::min(prod[0], prod[1]),
			std::min(prod[2], prod[3])));
		r.high = nextFloatUp(std::max(
			std::max(prod[0], prod[1]), 
			std::max(prod[2], prod[3])));

#ifdef _DEBUG
		r.ld = ld * f.ld;
#endif
		return r;
	}

	auto FloatWithErrBound::operator/(FloatWithErrBound f) const->FloatWithErrBound {
		FloatWithErrBound r;
		r.v = v / f.v;
		float div[4] = {
			lowerBound() / f.lowerBound(),
			lowerBound() / f.upperBound(),
			upperBound() / f.lowerBound(),
			upperBound() / f.upperBound(),
		};
		r.low = nextFloatDown(std::min(
			std::min(div[0], div[1]),
			std::min(div[2], div[3])));
		r.high = nextFloatUp(std::max(
			std::max(div[0], div[1]),
			std::max(div[2], div[3])));

#ifdef _DEBUG
		r.ld = ld / f.ld;
#endif
		return r;
	}
	auto FloatWithErrBound::operator-() const -> FloatWithErrBound {
		FloatWithErrBound r;
		r.v = -v;
		r.low = -high;
		r.high = -low;
#ifdef _DEBUG
		r.ld = -ld;
#endif
		return r;
	}

	auto FloatWithErrBound::operator==(FloatWithErrBound f) const -> bool {
		return v == f.v;
	}

	auto FloatWithErrBound::getAbsoluteError() const noexcept -> float { 
		return nextFloatUp(std::max(std::abs(high - v), std::abs(v - low)));
	}

	auto FloatWithErrBound::upperBound() const noexcept -> float {
		return high;
	}

	auto FloatWithErrBound::lowerBound() const noexcept -> float {
		return low;
	}
}