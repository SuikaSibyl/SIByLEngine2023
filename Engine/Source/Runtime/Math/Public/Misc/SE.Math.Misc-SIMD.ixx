module;
#include <xmmintrin.h>
export module SE.Math.Misc:SIMD;

namespace SIByL::Math
{
#define SHUFFLE_PARAM(x,y,z,w) \
	((x) | ((y)<<2) | ((z)<<4) | ((w)<<6))

	export inline auto _mm_replicate_x_ps(__m128 const& v) noexcept -> __m128 {
		return _mm_shuffle_ps(v, v, SHUFFLE_PARAM(0, 0, 0, 0));
	}

	export inline auto _mm_replicate_y_ps(__m128 const& v) noexcept -> __m128 {
		return _mm_shuffle_ps(v, v, SHUFFLE_PARAM(1, 1, 1, 1));
	}

	export inline auto _mm_replicate_z_ps(__m128 const& v) noexcept -> __m128 {
		return _mm_shuffle_ps(v, v, SHUFFLE_PARAM(2, 2, 2, 2));
	}
	export inline auto _mm_replicate_w_ps(__m128 const& v) noexcept -> __m128 {
		return _mm_shuffle_ps(v, v, SHUFFLE_PARAM(3, 3, 3, 3));
	}
}