module;
#include <atomic>
export module Parallelism.Atomic:AtomicFloat;

namespace SIByL::Parallelism
{
	export struct AtomicFloat
	{
	public:
		explicit AtomicFloat(float v = 0);

		operator float() const;
		auto operator=(float v) -> float;

		auto add(float v)->void;

	private:
		std::atomic<uint32_t> bits;
	};
}