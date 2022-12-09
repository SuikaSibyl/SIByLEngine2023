module;
#include <functional>
export module SE.Parallelism:ParallelFor;
import SE.Platform.Misc;
import SE.Math.Geometric;

namespace SIByL::Parallelism
{
	export inline auto ParallelFor(std::function<void(int)> const& func, int count, int chunckSize) noexcept -> void;
	export inline auto ParallelFor2D(std::function<void(Math::ipoint2)> const& func, Math::ipoint2 const& count) noexcept -> void;

	export inline auto clearThreadPool() noexcept -> void;
}