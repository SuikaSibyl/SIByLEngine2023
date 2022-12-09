module;
#include <cstdint>
#include <algorithm>
export module SE.Math.Misc:RNG;

namespace SIByL::Math
{
	/*
	* An implementation of the PCG pseudo-random number generator (O'Neill 2014)
	* to generate pseudo-random numbers
	*/
	export struct RNG
	{
	public:
		RNG();

		/** takes a single argument that selects a sequence of pseudo-random values.*/
		RNG(uint64_t sequenceIndex) { setSequence(sequenceIndex); }

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
		auto operator-(const RNG& other) const->int64_t;

	private:
		uint64_t state, inc;
	};

	template <class Iterator>
	auto RNG::shuffle(Iterator begin, Iterator end) noexcept -> void {
		for (Iterator it = end - 1; it > begin; --it)
			std::iter_swap(it,
				begin + uniformFloat((uint32_t)(it - begin + 1)));
	}

}
