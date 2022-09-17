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

    export template <typename Predicate>
        int findInterval(int size, const Predicate& pred) {
        int first = 0, len = size;
        while (len > 0) {
            int half = len >> 1, middle = first + half;
            // Bisect range based on value of _pred_ at _middle_
            if (pred(middle)) {
                first = middle + 1;
                len -= half + 1;
            }
            else
                len = half;
        }
        return Math::clamp(first - 1, 0, size - 2);
    }
}