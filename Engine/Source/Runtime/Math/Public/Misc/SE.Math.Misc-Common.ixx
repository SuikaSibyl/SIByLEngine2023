module;
#include <cmath>
#include <intrin.h>
export module SE.Math.Misc:Common;

namespace SIByL::Math
{
	export template <class T>
	inline auto clamp(T const& val, T const& min, T const& max) noexcept -> T {
		if (val < min) return min;
		else if (val > max) return max;
		else return val;
	}

	export template <class T>
	inline auto mod(T a, T b) noexcept -> T {
        T result = a - (a / b) * b;
        return (T)((result < 0) ? result + b : result);
	}
    
	export template <>
	inline auto mod(float a, float b) noexcept -> float {
        return std::fmod(a, b);
	}
    
    export inline auto log2(float x) noexcept -> float {
        float const invLog2 = 1.442695040888963387004650940071f;
        return std::log(x) * invLog2;
    }

    export inline auto ctz(uint32_t value) noexcept -> uint32_t {
        unsigned long trailing_zero = 0;
        if (_BitScanForward(&trailing_zero, value))
            return trailing_zero;
        // This is undefined, I better choose 32 than 0
        else return 32;
    }

    export inline auto clz(uint32_t value)-> uint32_t {
        unsigned long leading_zero = 0;
        if (_BitScanReverse(&leading_zero, value))
            return 31 - leading_zero;
            // Same remarks as above
        else return 32;
    }

    export inline auto log2Int(uint32_t v) noexcept -> int {
        return 31 - clz(v);
    }

    export template <class T>
    inline auto isPowerOf2(T v) noexcept -> T {
        return v && !(v & (v - 1));
    }

    export inline auto roundUpPow2(int32_t v) noexcept -> int32_t {
        --v;
        v |= v >> 1;    v |= v >> 2;
        v |= v >> 4;    v |= v >> 8;
        v |= v >> 16;
        return v + 1;
    }

    export inline auto countTrailingZeros(uint32_t v) noexcept -> int {
        return ctz(v);
    }

	export template <class T>
	inline auto lerp(float t, T const& a, T const& b) noexcept -> T {
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

    /** Rounds a up tp multiple of b. */
    export auto alignUp(uint32_t a, uint32_t b) -> uint32_t {
        uint32_t res = a + b - 1;
        return res - res % b;
    }
}