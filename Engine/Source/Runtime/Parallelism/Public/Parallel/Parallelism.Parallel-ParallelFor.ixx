module;
#include <vector>
#include <functional>
#include <thread>
export module Parallelism.Parallel:ParallelFor;
import Platform.System;

namespace SIByL::Parallelism
{
	/**
	* Encapsulate the relevant information about a parallel for loop body.
	* Including the function to run, the number of iterations, and which iterations are already done.
	*/
	export struct ParallelForLoop
	{
		ParallelForLoop(std::function<void(int)> func1D, int64_t maxIndex, int chunkSize, int profilerState);

	private:
		std::function<void(int)> const func1D;
		int64_t const maxIndex;
		int const chunkSize, const profilerState;

		int64_t nextIndex = 0;
		int activeWorkers = 0;
		ParallelForLoop* next = nullptr;
	};

	export inline auto ParallelFor(std::function<void(int)> const& func, int count, int chunckSize)
	{
		static std::vector<std::thread> threads;
		static bool shutdownThreads = false;
		extern thread_local int ThreadIndex;
		// Run iterations immediately if not using threads or if count is small
		if (count < chunckSize)
		{
			for (size_t i = 0; i < count; i++)
				func(i);
			return;
		}
		//// launch worker threads if needed
		//if (threads.size() == 0)
		//{
		//	ThreadIndex = 0;
		//	for (size_t i = 0; i < Platform::getNumSystemCores() - 1; ++i)
		//		threads.push_back(std::thread(workerThreadFunc, i + 1));
		//}

		// create and enqueue ParallelForLoop for this loop

		// notify worker threads of work to be done

		// help out with parallel loop iterations in the current thread

	};
}