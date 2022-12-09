module;
#include <cstdint>
#include <numeric>
export module SE.Math.Misc:RoundingError;

namespace SIByL::Math
{
	export inline auto floatToBits(float f) noexcept -> uint32_t;
	export inline auto bitsToFloat(uint32_t b) noexcept -> float;

	/** bump a floating-point value up to the next greater reprensentable floating-point value */
	export inline auto nextFloatUp(float v) noexcept -> float;
	/** bump a floating-point value down to the next smaller reprensentable floating-point value */
	export inline auto nextFloatDown(float v) noexcept -> float;

	/** machine epsilon is a conservative relative error for floating-point rounding */
	export inline float MachineEpsilon = std::numeric_limits<float>::epsilon() * 0.5f;
	/* tight bound of (1����)^n given by Higham */
	export inline auto gamma(int n) noexcept -> float;
	
	/**
	* Each time a floating-point operation is performed. we also compute terms that
	* compute intervals to  compute a running bound on the error.
	* Provide all of the regular arithmetic operations on floats while computing
	* these error bounds.
	*/
	export struct FloatWithErrBound
	{
		FloatWithErrBound() = default;
		FloatWithErrBound(float v);
		FloatWithErrBound(float v, float err);

		auto operator+(FloatWithErrBound f) const -> FloatWithErrBound;
		auto operator-(FloatWithErrBound f) const -> FloatWithErrBound;
		auto operator*(FloatWithErrBound f) const -> FloatWithErrBound;
		auto operator/(FloatWithErrBound f) const -> FloatWithErrBound;

		auto operator-() const -> FloatWithErrBound;
		auto operator==(FloatWithErrBound f) const -> bool;

		explicit operator float() const { return v; }
		explicit operator double() const { return (double)v; }
		auto getAbsoluteError() const noexcept -> float;
		auto upperBound() const noexcept -> float;
		auto lowerBound() const noexcept -> float;

	private:
		float v;
		float low, high;
#ifdef _DEBUG
		long double ld;
#endif
	};

	export using efloat = FloatWithErrBound;
}