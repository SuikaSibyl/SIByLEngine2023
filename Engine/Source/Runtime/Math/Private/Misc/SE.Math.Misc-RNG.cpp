module;
#include <cstdint>
#include <algorithm>
#include <cmath>
module SE.Math.Misc:RNG;

#define PCG32_DEFAULT_STATE 0x853c49e6748fea9bULL
#define PCG32_DEFAULT_STREAM 0xda3e39cb94b95bdbULL
#define PCG32_MULT 0x5851f42d4c957f2dULL

namespace SIByL::Math
{
	inline float oneMinusEpsilon = 0x1.fffffep-1;

	RNG::RNG() :state(PCG32_DEFAULT_STATE), inc(PCG32_DEFAULT_STREAM) {}

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
		uint32_t threshold = (~b + 1u) % b; // (2^32-1 - 1 + 1) % b
		while (true) {
			uint32_t r = uniformUInt32();
			if (r >= threshold) return r % b;
		}
	}

	auto RNG::uniformFloat() noexcept -> float {
		return std::min(oneMinusEpsilon, uniformUInt32() * 0x1p-32f);
	}

	auto RNG::advance(int64_t idelta) noexcept -> void {
		uint64_t cur_mult = PCG32_MULT, cur_plus = inc, acc_mult = 1u,
			acc_plus = 0u, delta = (uint64_t)idelta;
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

}
