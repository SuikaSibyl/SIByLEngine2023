module;
#include <cstdint>
export module Math.FixedPoint;

namespace SIByL::Math 
{
	export struct fixed32_t {
		/** @param useRawInit: if true, directly set raw = value */
		fixed32_t(int32_t value, bool useRawInit = false);

		auto operator+(fixed32_t const& rhs) const -> fixed32_t;
		auto operator-(fixed32_t const& rhs) const -> fixed32_t;
		auto operator*(fixed32_t const& rhs) const -> fixed32_t;
		auto operator/(fixed32_t const& rhs) const -> fixed32_t;

		/**
		* raw data value of fixed32_t,
		* early 22 bits are whole part,
		* last 10 bits are fraction part
		*/
		int32_t raw;
		/** Number of bits used to represent whole part */
		static constexpr const int32_t wholeBitCount = 22;
		/** Number of bits used to represent fraction part */
		static constexpr const int32_t fractionBitCount = 10;
	};


	export struct fixed64_t {
		/** @param useRawInit: if true, directly set raw = value */
		fixed64_t(int64_t value, bool useRawInit = false);

		auto operator+(fixed64_t const& rhs) const -> fixed64_t;
		auto operator-(fixed64_t const& rhs) const -> fixed64_t;
		auto operator*(fixed64_t const& rhs) const -> fixed64_t;
		auto operator/(fixed64_t const& rhs) const -> fixed64_t;

		/**
		* raw data value of fixed64_t,
		* early 32 bits are whole part,
		* last 32 bits are fraction part
		*/
		int64_t raw;
		/** Number of bits used to represent whole part */
		static constexpr const int32_t wholeBitCount = 32;
		/** Number of bits used to represent fraction part */
		static constexpr const int32_t fractionBitCount = 32;
	};
}