module;
#include <atomic>
module Parallelism.Atomic:AtomicFloat;
import Parallelism.Atomic;

namespace SIByL::Parallelism
{
	auto floatToBits(float v) noexcept -> uint32_t
	{
		return *(reinterpret_cast<uint32_t*>(&v));
	}
	auto bitsToFloat(uint32_t v) noexcept -> float
	{
		return *(reinterpret_cast<float*>(&v));
	}

	AtomicFloat::AtomicFloat(float v) { bits = floatToBits(v); }

	AtomicFloat::operator float() const { return bitsToFloat(bits); }
	auto AtomicFloat::operator=(float v) -> float { bits = floatToBits(v); return v; }

	auto AtomicFloat::add(float v)->void
	{
		uint32_t oldBits = bits, newBits;
		do {
			newBits = floatToBits(bitsToFloat(oldBits) + v);
		} while (!bits.compare_exchange_weak(oldBits, newBits));
	}

}