module;
#include <vector>
#include <functional>
#include <thread>
export module Parallelism.Parallel:ParallelFor;

namespace SIByL::Parallelism
{
	export auto ParallelFor(std::function<void(int)> const& func, int count, int chunckSize)
	{
		static std::vector<std::thread> threads;
		static bool shutdownThreads = false;

		// Run iterations immediately if not using threads or if count is small
		if (count < chunckSize)
		{
			for (size_t i = 0; i < count; i++)
				func(i);
			return;
		}
		// launch worker threads if needed
		if (threads.size() == 0)
		{
			//ThreadIndex = 0;
			//for (size_t i = 0; i < NumSystemCores - 1; ++i)
			//	threads.push_back(std::thread(workerThreadFunc, i + 1));
		}

		// create and enqueue ParallelForLoop for this loop

		// notify worker threads of work to be done

		// help out with parallel loop iterations in the current thread

	};
}