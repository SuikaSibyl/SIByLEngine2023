#pragma once
#include <intrin.h>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <xmmintrin.h>
#include <algorithm>
#include <common_config.hpp>

namespace SIByL::Math {
SE_EXPORT template <class T>
inline auto clamp(T const& val, T const& min, T const& max) noexcept -> T {
  if (val < min)
    return min;
  else if (val > max)
    return max;
  else
    return val;
}

SE_EXPORT template <class T>
inline auto mod(T a, T b) noexcept -> T {
  T result = a - (a / b) * b;
  return (T)((result < 0) ? result + b : result);
}

SE_EXPORT template <>
inline auto mod(float a, float b) noexcept -> float {
  return std::fmod(a, b);
}

//SE_EXPORT inline auto log2(float x) noexcept -> float {
//  float const invLog2 = 1.442695040888963387004650940071f;
//  return std::log(x) * invLog2;
//}

SE_EXPORT inline auto ctz(uint32_t value) noexcept -> uint32_t {
  unsigned long trailing_zero = 0;
  if (_BitScanForward(&trailing_zero, value)) return trailing_zero;
  // This is undefined, I better choose 32 than 0
  else
    return 32;
}

SE_EXPORT inline auto clz(uint32_t value) -> uint32_t {
  unsigned long leading_zero = 0;
  if (_BitScanReverse(&leading_zero, value)) return 31 - leading_zero;
  // Same remarks as above
  else
    return 32;
}

SE_EXPORT inline auto log2Int(uint32_t v) noexcept -> int { return 31 - clz(v); }

SE_EXPORT template <class T>
inline auto isPowerOf2(T v) noexcept -> T {
  return v && !(v & (v - 1));
}

SE_EXPORT inline auto roundUpPow2(int32_t v) noexcept -> int32_t {
  --v;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  return v + 1;
}

SE_EXPORT inline auto countTrailingZeros(uint32_t v) noexcept -> int {
  return ctz(v);
}

SE_EXPORT template <class T>
inline auto lerp(float t, T const& a, T const& b) noexcept -> T {
  return a * (1 - t) + b * t;
}

SE_EXPORT template <typename Predicate>
int findInterval(int size, const Predicate& pred) {
  int first = 0, len = size;
  while (len > 0) {
    int half = len >> 1, middle = first + half;
    // Bisect range based on value of _pred_ at _middle_
    if (pred(middle)) {
      first = middle + 1;
      len -= half + 1;
    } else
      len = half;
  }
  return Math::clamp(first - 1, 0, size - 2);
}

/** Rounds a up tp multiple of b. */
SE_EXPORT auto alignUp(uint32_t a, uint32_t b) -> uint32_t;
}  // namespace SIByL::Math

namespace SIByL::Math {
SE_EXPORT inline auto floatToBits(float f) noexcept -> uint32_t;
SE_EXPORT inline auto bitsToFloat(uint32_t b) noexcept -> float;

/** bump a floating-point value up to the next greater reprensentable
 * floating-point value */
SE_EXPORT inline auto nextFloatUp(float v) noexcept -> float;
/** bump a floating-point value down to the next smaller reprensentable
 * floating-point value */
SE_EXPORT inline auto nextFloatDown(float v) noexcept -> float;

/** machine epsilon is a conservative relative error for floating-point rounding
 */
SE_EXPORT inline float MachineEpsilon =
    std::numeric_limits<float>::epsilon() * 0.5f;
/* tight bound of (1 ~~)^n given by Higham */
SE_EXPORT inline auto gamma(int n) noexcept -> float;

/**
 * Each time a floating-point operation is performed. we also compute terms that
 * compute intervals to  compute a running bound on the error.
 * Provide all of the regular arithmetic operations on floats while computing
 * these error bounds.
 */
SE_EXPORT struct FloatWithErrBound {
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

SE_EXPORT using efloat = FloatWithErrBound;
}  // namespace SIByL::Math

namespace SIByL::Math {
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
  if (std::isinf(v) && v > 0.f) return v;
  if (v == -0.f) return 0.f;
  // Advance v to next higher float
  uint32_t ui = floatToBits(v);
  if (v >= 0)
    ++ui;
  else
    --ui;
  return bitsToFloat(ui);
}

inline auto nextFloatDown(float v) noexcept -> float {
  // Hanlde infinity and positive zero
  if (std::isinf(v) && v < 0.f) return v;
  if (v == +0.f) return -0.f;
  // Advance v to next higher float
  uint32_t ui = floatToBits(v);
  if (v >= 0)
    --ui;
  else
    ++ui;
  return bitsToFloat(ui);
}

inline auto gamma(int n) noexcept -> float {
  return (n * MachineEpsilon) / (1 - n * MachineEpsilon);
}
}  // namespace SIByL::Math

namespace SIByL::Math {
#define SHUFFLE_PARAM(x, y, z, w) ((x) | ((y) << 2) | ((z) << 4) | ((w) << 6))

SE_EXPORT inline auto _mm_replicate_x_ps(__m128 const& v) noexcept -> __m128 {
  return _mm_shuffle_ps(v, v, SHUFFLE_PARAM(0, 0, 0, 0));
}

SE_EXPORT inline auto _mm_replicate_y_ps(__m128 const& v) noexcept -> __m128 {
  return _mm_shuffle_ps(v, v, SHUFFLE_PARAM(1, 1, 1, 1));
}

SE_EXPORT inline auto _mm_replicate_z_ps(__m128 const& v) noexcept -> __m128 {
  return _mm_shuffle_ps(v, v, SHUFFLE_PARAM(2, 2, 2, 2));
}
SE_EXPORT inline auto _mm_replicate_w_ps(__m128 const& v) noexcept -> __m128 {
  return _mm_shuffle_ps(v, v, SHUFFLE_PARAM(3, 3, 3, 3));
}
}  // namespace SIByL::Math

namespace SIByL::Math {
/*
 * An implementation of the PCG pseudo-random number generator (O'Neill 2014)
 * to generate pseudo-random numbers
 */
SE_EXPORT struct RNG {
 public:
  RNG();

  /** takes a single argument that selects a sequence of pseudo-random values.*/
  explicit RNG(uint64_t sequenceIndex) { setSequence(sequenceIndex); }

  auto setSequence(uint64_t sequenceIndex) noexcept -> void;

  /** returns a pseudo-random number in range [0, 2^32 - 1]*/
  auto uniformUInt32() noexcept -> uint32_t;

  /** returns a value uniformedly distributed in range [0, b - 1] */
  auto uniformUInt32(uint32_t b) noexcept -> uint32_t;

  /** returns a pseudo-random floating-point number in range [0, 1) */
  auto uniformFloat() noexcept -> float;

  template <class Iterator>
  auto shuffle(Iterator begin, Iterator end) noexcept -> void;

  auto advance(int64_t idelta) noexcept -> void;
  auto operator-(const RNG& other) const -> int64_t;

 private:
  uint64_t state{}, inc{};
};

template <class Iterator>
auto RNG::shuffle(Iterator begin, Iterator end) noexcept -> void {
  for (Iterator it = end - 1; it > begin; --it)
    std::iter_swap(it, begin + uniformUInt32((uint32_t)(it - begin + 1)));
}

}  // namespace SIByL::Math

#define PCG32_DEFAULT_STATE 0x853c49e6748fea9bULL
#define PCG32_DEFAULT_STREAM 0xda3e39cb94b95bdbULL
#define PCG32_MULT 0x5851f42d4c957f2dULL

namespace SIByL::Math {
inline float oneMinusEpsilon = 0x1.fffffep-1;
}  // namespace SIByL::Math

namespace SIByL::Math {
SE_EXPORT inline float float_Pi = 3.14159265358979323846f;
SE_EXPORT inline double double_Pi = 3.14159265358979323846f;

SE_EXPORT inline float float_InvPi = 1.f / float_Pi;
SE_EXPORT inline float float_Inv2Pi = 1.f / (2 * float_Pi);
SE_EXPORT inline float float_Inv4Pi = 1.f / (4 * float_Pi);
SE_EXPORT inline float float_PiOver2 = float_Pi / 2;
SE_EXPORT inline float float_PiOver4 = float_Pi / 4;

SE_EXPORT inline double double_InvPi = 1. / double_Pi;
SE_EXPORT inline double double_Inv2Pi = 1. / (2 * double_Pi);
SE_EXPORT inline double double_Inv4Pi = 1. / (4 * double_Pi);
SE_EXPORT inline double double_PiOver2 = double_Pi / 2;
SE_EXPORT inline double double_PiOver4 = double_Pi / 4;

SE_EXPORT inline auto radians(float deg) noexcept -> float {
  return (float_Pi / 180) * deg;
}
SE_EXPORT inline auto radians(double deg) noexcept -> double {
  return (double_Pi / 180) * deg;
}
}  // namespace SIByL::Math

namespace SIByL::Math {
SE_EXPORT constexpr inline float float_infinity =
    std::numeric_limits<float>::infinity();
SE_EXPORT constexpr inline float float_min = std::numeric_limits<float>::min();
SE_EXPORT constexpr inline float float_max = std::numeric_limits<float>::max();
SE_EXPORT constexpr inline float one_minus_epsilon = 0x1.fffffep-1;

SE_EXPORT constexpr inline uint32_t uint32_max =
    std::numeric_limits<uint32_t>::max();

SE_EXPORT constexpr inline float shadow_epsilon = 0.0001f;

SE_EXPORT struct Limits {
  static float float_min;
};
}  // namespace SIByL::Math


namespace SIByL::Math {
SE_EXPORT inline auto quadratic(float a, float b, float c, float& t0,
                             float& t1) noexcept -> bool;
SE_EXPORT inline auto quadratic(efloat a, efloat b, efloat c, efloat& t0,
                             efloat& t1) noexcept -> bool;

SE_EXPORT inline auto solveLinearSystem2x2(float const A[2][2], const float B[2],
                                        float* x0, float* x1) noexcept -> bool;
}  // namespace SIByL::Math

namespace SIByL::Math {
inline auto quadratic(float a, float b, float c, float& t0, float& t1) noexcept
    -> bool {
  // Find quadratic discriminant
  double discrim = (double)b * (double)b - 4 * (double)a * (double)c;
  if (discrim < 0) return false;
  double rootDiscrim = std::sqrt(discrim);
  // Compute quadratic t value
  double q;
  if (b < 0)
    q = -.5 * (b - rootDiscrim);
  else
    q = -.5 * (b + rootDiscrim);
  t0 = static_cast<float>(q / a);
  t1 = static_cast<float>(c / q);
  if (t0 > t1) std::swap(t0, t1);
  return true;
}

inline auto quadratic(efloat a, efloat b, efloat c, efloat& t0,
                      efloat& t1) noexcept -> bool {
  // Find quadratic discriminant
  double discrim = (double)b * (double)b - 4 * (double)a * (double)c;
  if (discrim < 0) return false;
  double rootDiscrim = std::sqrt(discrim);

  efloat floatRootDiscrim(static_cast<float>(rootDiscrim),
                          static_cast<float>(MachineEpsilon * rootDiscrim));

  // Compute quadratic t value
  efloat q;
  if ((float)b < 0)
    q = efloat(-.5f) * (b - floatRootDiscrim);
  else
    q = efloat(-.5f) * (b + floatRootDiscrim);
  t0 = q / a;
  t1 = c / q;
  if ((float)t0 > (float)t1) std::swap(t0, t1);
  return true;
}

inline auto solveLinearSystem2x2(float const A[2][2], const float B[2],
                                 float* x0, float* x1) noexcept -> bool {
  float det = A[0][0] * A[1][1] - A[0][1] * A[1][0];
  if (std::abs(det) < 1e-10f) return false;
  *x0 = (A[1][1] * B[0] - A[0][1] * B[1]) / det;
  *x1 = (A[0][0] * B[1] - A[1][0] * B[0]) / det;
  if (std::isnan(*x0) || std::isnan(*x1)) return false;
  return true;
}


const inline double FEQ_EPS = 1e-6;
const inline double FEQ_EPS2 = 1e-12;

inline bool float_equal(double a, double b, double e = FEQ_EPS) {
  return fabs(a - b) < e;
}
inline bool float_equal2(double a, double b, double e = FEQ_EPS2) {
  return fabs(a - b) < e;
}

}  // namespace SIByL::Math