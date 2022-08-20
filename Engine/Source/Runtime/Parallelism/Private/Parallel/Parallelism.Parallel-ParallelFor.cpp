module;
#include <vector>
#include <functional>
#include <thread>
#include <mutex>
#include <cstdint>
#include <iostream>
module Parallelism.Parallel:ParallelFor;
import Parallelism.Parallel;
import Platform.System;
import Math.Vector;
import Math.Geometry;

namespace SIByL::Parallelism
{
	/**
	* Encapsulate the relevant information about a parallel for loop body.
	* Including the function to run, the number of iterations, and which iterations are already done.
	*/
	struct ParallelForLoop
	{
		ParallelForLoop(std::function<void(int)> func1D, int64_t maxIndex, int chunkSize, uint64_t profilerState);
		ParallelForLoop(std::function<void(Math::ipoint2)> func2D, Math::ipoint2 const& count, uint64_t profilerState);

		/**
		* Only finished when the index has been advanced to the end of the loop's range
		* and there are no threads currently working on it.
		*/
		auto finished() const noexcept -> bool;

		std::function<void(int)> const func1D;
		std::function<void(Math::ipoint2)> const func2D;
		int64_t const maxIndex;
		int const chunkSize;
		uint64_t const profilerState;
		int nX;

		/** tracks the next loop index to be executed */
		int64_t nextIndex = 0;
		/** records how many worker threads are currently running iterations of the loop*/
		int activeWorkers = 0;
		/** maintain the linked list of nested loops */
		ParallelForLoop* next = nullptr;
	};

	ParallelForLoop::ParallelForLoop(std::function<void(int)> func1D, int64_t maxIndex, int chunkSize, uint64_t profilerState)
		:func1D(func1D), maxIndex(maxIndex), chunkSize(chunkSize), profilerState(profilerState)
	{}

	ParallelForLoop::ParallelForLoop(std::function<void(Math::ipoint2)> func2D, Math::ipoint2 const& count, uint64_t profilerState)
		:func2D(func2D), maxIndex(count.x* count.y), chunkSize(1), profilerState(profilerState)
	{	nX = count.x; }

	auto ParallelForLoop::finished() const noexcept -> bool {
		return nextIndex >= maxIndex && activeWorkers == 0;
	}

	// threads don't terminate after returns, instead they wait on
	// a condition variable that signals more worls.
	inline std::vector<std::thread> threads;
	inline volatile bool shutdownThreads = false;
	// a thread-local storage should be allocated for determine which thread they are
	thread_local int ThreadIndex;

	// Holds a pointer to the head of a list of parallel for loops that aren't yet finished.
	inline ParallelForLoop* workList = nullptr;
	// Must always be held when accessing workList or values stored in the ParallelForLoop objects held in it.
	inline std::mutex workListMutex;

	inline std::condition_variable workListCondition;

	inline auto workerThreadFunc(int tIndex) noexcept -> void {
		ThreadIndex = tIndex;
		std::unique_lock<std::mutex> lock(workListMutex);
		while (!shutdownThreads) {
			if (!workList) {
				// sleep until there are more tasks to run
				workListCondition.wait(lock);
			}
			else { 
				// get work from workList and run loop iterations
				ParallelForLoop& loop = *workList;
				// run a chunk of loop iterations for loop
				// - find the set of loop iterations to run next
				int64_t indexStart = loop.nextIndex;
				int64_t indexEnd = std::min(indexStart + loop.chunkSize, loop.maxIndex);
				// - update loop to reflect iterations this thread will run
				loop.nextIndex = indexEnd;
				if (loop.nextIndex == loop.maxIndex)
					workList = loop.next;
				loop.activeWorkers++;
				// - run loop indicies in [indexStart, indexEnd]
				lock.unlock();
				for (int index = indexStart; index < indexEnd; ++index) {
					if (loop.func1D) {
						loop.func1D(index);
					}
					// handle other types of loops
				}
				lock.lock();
				// - update loop to reflect completion of itertaions
				--loop.activeWorkers;

				if (loop.finished())
					workListCondition.notify_all();
			}
		}
		// report thread statistics at worker thread exit
	}

	inline auto ParallelFor(std::function<void(int)> const& func, int count, int chunckSize) noexcept -> void
	{
		// Run iterations immediately if not using threads or if count is small
		if (count < chunckSize) {
			for (size_t i = 0; i < count; i++)
				func(i);
			return;
		}
		// launch worker threads if needed
		if (threads.size() == 0) {
			ThreadIndex = 0;
			for (size_t i = 0; i < Platform::getNumSystemCores() - 1; ++i)
				threads.push_back(std::thread(workerThreadFunc, i + 1));
		}
		// create and enqueue ParallelForLoop for this loop
		ParallelForLoop loop(func, count, chunckSize, 0);
		workListMutex.lock();
		loop.next = workList;
		workList = &loop;
		workListMutex.unlock();

		// notify worker threads of work to be done
		std::unique_lock<std::mutex> lock(workListMutex);
		workListCondition.notify_all();
		// help out with parallel loop iterations in the current thread
		while (!loop.finished()) {
			// run a chunck of loop iterations for loop
			// - find the set of loop iterations to run next
			int64_t indexStart = loop.nextIndex;
			int64_t indexEnd = std::min(indexStart + loop.chunkSize, loop.maxIndex);
			// - update loop to reflect iterations this thread will run
			loop.nextIndex = indexEnd;
			if (loop.nextIndex == loop.maxIndex)
				workList = loop.next;
			loop.activeWorkers++;
			// - run loop indicies in [indexStart, indexEnd]
			lock.unlock();
			for (int index = indexStart; index < indexEnd; ++index) {
				if (loop.func1D) {
					loop.func1D(index);
				}
				// handle other types of loops
			}
			lock.lock();
			// - update loop to reflect completion of itertaions
			--loop.activeWorkers;
		}
	};
	
	inline auto ParallelFor2D(std::function<void(Math::ipoint2)> const& func, Math::ipoint2 const& count) noexcept -> void {
		// Run iterations immediately if not using threads or if count is small
		if (count.x * count.y <= 1) {
			for (int j = 0; j < count.y; ++j)
				for (int i = 0; i < count.x; ++i)
					func(Math::ipoint2{ i,j });
			return;
		}
		// launch worker threads if needed
		if (threads.size() == 0) {
			ThreadIndex = 0;
			for (size_t i = 0; i < Platform::getNumSystemCores() - 1; ++i)
				threads.push_back(std::thread(workerThreadFunc, i + 1));
		}
		// create and enqueue ParallelForLoop for this loop
		ParallelForLoop loop(func, count, 0);
		workListMutex.lock();
		loop.next = workList;
		workList = &loop;
		workListMutex.unlock();

		// notify worker threads of work to be done
		std::unique_lock<std::mutex> lock(workListMutex);
		workListCondition.notify_all();
		// help out with parallel loop iterations in the current thread
		while (!loop.finished()) {
			// run a chunck of loop iterations for loop
			// - find the set of loop iterations to run next
			int64_t indexStart = loop.nextIndex;
			int64_t indexEnd = std::min(indexStart + loop.chunkSize, loop.maxIndex);
			// - update loop to reflect iterations this thread will run
			loop.nextIndex = indexEnd;
			if (loop.nextIndex == loop.maxIndex)
				workList = loop.next;
			loop.activeWorkers++;
			// - run loop indicies in [indexStart, indexEnd]
			lock.unlock();
			for (int index = indexStart; index < indexEnd; ++index) {
				if (loop.func2D) {
					loop.func2D(Math::ipoint2{ index % loop.nX,index / loop.nX });
				}
				// handle other types of loops
			}
			lock.lock();
			// - update loop to reflect completion of itertaions
			--loop.activeWorkers;
		}
	}

	inline auto clearThreadPool() noexcept -> void {
		shutdownThreads = true;
		workListCondition.notify_all();

		for (auto& thread : threads)
			thread.join();
		threads.clear();
		shutdownThreads = false;
	}
}