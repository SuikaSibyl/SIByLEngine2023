module;
#include <intrin.h>

#include <cstdint>
#include <functional>
module SE.Math.FixedPoint;

#ifdef _MSC_VER__
#pragma intrinsic(_mul128)
#endif

namespace SIByL::Math {
// -----------------------
// fixed32_t impl
// -----------------------

fixed32_t::fixed32_t(int32_t value, bool useRawInit)
    : raw([&]() { return useRawInit ? value : value << fractionBitCount; }()) {}

auto fixed32_t::operator+(fixed32_t const& rhs) const -> fixed32_t {
  return fixed32_t(this->raw + rhs.raw, true);
}

auto fixed32_t::operator-(fixed32_t const& rhs) const -> fixed32_t {
  return fixed32_t(this->raw - rhs.raw, true);
}

auto fixed32_t::operator*(fixed32_t const& rhs) const -> fixed32_t {
  return fixed32_t((int64_t(this->raw) * int64_t(rhs.raw)) >> fractionBitCount,
                   true);
}

auto fixed32_t::operator/(fixed32_t const& rhs) const -> fixed32_t {
  return fixed32_t((int64_t(this->raw) << fractionBitCount) / int64_t(rhs.raw),
                   true);
}

fixed64_t::fixed64_t(int64_t value, bool useRawInit)
    : raw([&]() { return useRawInit ? value : value << fractionBitCount; }()) {}

// -----------------------
// fixed64_t impl
// -----------------------

auto fixed64_t::operator+(fixed64_t const& rhs) const -> fixed64_t {
  return fixed64_t(this->raw + rhs.raw, true);
}

auto fixed64_t::operator-(fixed64_t const& rhs) const -> fixed64_t {
  return fixed64_t(this->raw - rhs.raw, true);
}

auto fixed64_t::operator*(fixed64_t const& rhs) const -> fixed64_t {
  int64_t product128_high = 0;
  int64_t const product128_low = _mul128(this->raw, rhs.raw, &product128_high);
  return fixed64_t(
      (product128_high << wholeBitCount) | (product128_low >> fractionBitCount),
      true);
}

#ifdef __clang__
int64_t div128by64(int64_t hi_64, int64_t lo_64, uint64_t y) {
  __int128 hi = hi_64;
  __int128 lo = lo_64;
  __int128 x = (hi << 64) + lo_64;
  int64_t quotient = x / y;
  return quotient;
}
auto fixed64_t::operator/(fixed64_t const& rhs) const -> fixed64_t {
  int64_t const highDividend = int64_t(this->raw) >> wholeBitCount;
  int64_t const lowDividend = int64_t(this->raw) << fractionBitCount;
  int64_t const quotient64 = div128by64(highDividend, lowDividend, rhs.raw);
  return fixed64_t(quotient64, true);
}
#elif
auto fixed64_t::operator/(fixed64_t const& rhs) const -> fixed64_t {
  int64_t const highDividend = int64_t(this->raw) >> wholeBitCount;
  int64_t const lowDividend = int64_t(this->raw) << fractionBitCount;
  int64_t remainder64 = 0;
  int64_t const quotient64 =
      _div128(highDividend, lowDividend, rhs.raw, &remainder64);
  return fixed64_t(quotient64, true);
}
#endif

}  // namespace SIByL::Math