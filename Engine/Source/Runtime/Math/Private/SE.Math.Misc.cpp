#include <algorithm>
#include <SE.Math.Misc.hpp>

namespace SIByL::Math {
float Limits::float_min = 0.f;

auto alignUp(uint32_t a, uint32_t b) -> uint32_t {
  uint32_t res = a + b - 1;
  return res - res % b;
}

FloatWithErrBound::FloatWithErrBound(float v) : v(v) {
  low = high = v;
#ifdef _DEBUG
  ld = v;
#endif
}

FloatWithErrBound::FloatWithErrBound(float v, float err) : v(v) {
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

auto FloatWithErrBound::operator+(FloatWithErrBound f) const
    -> FloatWithErrBound {
  FloatWithErrBound r;
  r.v = v + f.v;
  r.low = nextFloatDown(lowerBound() + f.lowerBound());
  r.high = nextFloatUp(upperBound() + f.upperBound());
#ifdef _DEBUG
  r.ld = ld + f.ld;
#endif
  return r;
}

auto FloatWithErrBound::operator-(FloatWithErrBound f) const
    -> FloatWithErrBound {
  FloatWithErrBound r;
  r.v = v - f.v;
  r.low = nextFloatDown(lowerBound() - f.upperBound());
  r.high = nextFloatUp(upperBound() - f.lowerBound());
#ifdef _DEBUG
  r.ld = ld - f.ld;
#endif
  return r;
}

auto FloatWithErrBound::operator*(FloatWithErrBound f) const
    -> FloatWithErrBound {
  FloatWithErrBound r;
  r.v = v * f.v;
  float prod[4] = {
      lowerBound() * f.lowerBound(),
      lowerBound() * f.upperBound(),
      upperBound() * f.lowerBound(),
      upperBound() * f.upperBound(),
  };
  r.low = nextFloatDown(
      std::min(std::min(prod[0], prod[1]), std::min(prod[2], prod[3])));
  r.high = nextFloatUp(
      std::max(std::max(prod[0], prod[1]), std::max(prod[2], prod[3])));

#ifdef _DEBUG
  r.ld = ld * f.ld;
#endif
  return r;
}

auto FloatWithErrBound::operator/(FloatWithErrBound f) const
    -> FloatWithErrBound {
  FloatWithErrBound r;
  r.v = v / f.v;
  float div[4] = {
      lowerBound() / f.lowerBound(),
      lowerBound() / f.upperBound(),
      upperBound() / f.lowerBound(),
      upperBound() / f.upperBound(),
  };
  r.low = nextFloatDown(
      std::min(std::min(div[0], div[1]), std::min(div[2], div[3])));
  r.high =
      nextFloatUp(std::max(std::max(div[0], div[1]), std::max(div[2], div[3])));

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

auto FloatWithErrBound::upperBound() const noexcept -> float { return high; }

auto FloatWithErrBound::lowerBound() const noexcept -> float { return low; }

RNG::RNG() : state(PCG32_DEFAULT_STATE), inc(PCG32_DEFAULT_STREAM) {}

auto RNG::setSequence(uint64_t sequenceIndex) noexcept -> void {
  state = 0u;
  inc = (sequenceIndex << 1u) | 1u;
  uniformUInt32();
  state += PCG32_DEFAULT_STATE;
  uniformUInt32();
}

auto RNG::uniformUInt32() noexcept -> uint32_t {
  uint64_t oldstate = state;
  state = oldstate * PCG32_MULT + inc;
  uint32_t xorshifted = (uint32_t)(((oldstate >> 18u) ^ oldstate) >> 27u);
  uint32_t rot = (uint32_t)(oldstate >> 59u);
  return (xorshifted >> rot) | (xorshifted << ((~rot + 1u) & 31));
}

auto RNG::uniformUInt32(uint32_t b) noexcept -> uint32_t {
  // computes the above remainder 2^32 mod b
  uint32_t threshold = (~b + 1u) % b;  // (2^32-1 - 1 + 1) % b
  while (true) {
    uint32_t r = uniformUInt32();
    if (r >= threshold) return r % b;
  }
}

auto RNG::uniformFloat() noexcept -> float {
  return std::min(oneMinusEpsilon, uniformUInt32() * 0x1p-32f);
}

auto RNG::advance(int64_t idelta) noexcept -> void {
  uint64_t cur_mult = PCG32_MULT, cur_plus = inc, acc_mult = 1u, acc_plus = 0u,
           delta = (uint64_t)idelta;
  while (delta > 0) {
    if (delta & 1) {
      acc_mult *= cur_mult;
      acc_plus = acc_plus * cur_mult + cur_plus;
    }
    cur_plus = (cur_mult + 1) * cur_plus;
    cur_mult *= cur_mult;
    delta /= 2;
  }
  state = acc_mult * state + acc_plus;
}

auto RNG::operator-(const RNG& other) const -> int64_t {
  uint64_t cur_mult = PCG32_MULT, cur_plus = inc, cur_state = other.state,
           the_bit = 1u, distance = 0u;
  while (state != cur_state) {
    if ((state & the_bit) != (cur_state & the_bit)) {
      cur_state = cur_state * cur_mult + cur_plus;
      distance |= the_bit;
    }
    the_bit <<= 1;
    cur_plus = (cur_mult + 1ULL) * cur_plus;
    cur_mult *= cur_mult;
  }
  return (int64_t)distance;
}
}  // namespace SIByL::Math