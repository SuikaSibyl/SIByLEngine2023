export module Math.Common;

namespace SIByL::Math
{
	export template <class T>
		inline auto clamp(T const& val, T const& min, T const& max) noexcept -> T
	{
		if (val < min) return min;
		else if (val > max) return max;
		else return val;
	}

	export template <class T>
		inline auto lerp(float t, T const& a, T const& b) noexcept -> T
	{
		return a * (1 - t) + b * t;
	}
}