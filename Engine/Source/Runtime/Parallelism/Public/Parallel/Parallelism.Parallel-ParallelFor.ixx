module;
#include <functional>
export module Parallelism.Parallel:ParallelFor;
import Platform.System;
import Math.Vector;
import Math.Geometry;

namespace SIByL::Parallelism
{
	export inline auto ParallelFor(std::function<void(int)> const& func, int count, int chunckSize) noexcept -> void;
	export inline auto ParallelFor2D(std::function<void(Math::ipoint2)> const& func, Math::ipoint2 const& count) noexcept -> void;

	export inline auto clearThreadPool() noexcept -> void;
}