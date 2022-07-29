module;
#include <functional>
export module Parallelism.Parallel:ParallelFor;
import Platform.System;

namespace SIByL::Parallelism
{
	export inline auto ParallelFor(std::function<void(int)> const& func, int count, int chunckSize) noexcept -> void;

	export inline auto clearThreadPool() noexcept -> void;
}