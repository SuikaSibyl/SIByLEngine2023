module;
#include <chrono>
#include <string>
#include <iostream>
export module Core.Timer;

namespace SIByL::Core
{
	export struct Timer {
		Timer();
		/** update the timer */
		auto update() noexcept -> void;
		/** get the delta time */
		auto deltaTime() noexcept -> double;
		/** get the total time */
		auto totalTime() noexcept -> double;
	private:
		std::chrono::steady_clock::time_point startTimePoint;
		std::chrono::steady_clock::time_point prevTimePoint;
		double _deltaTime = 0.f;
	};

#pragma region TIMER_IMPL

	Timer::Timer() {
		startTimePoint = std::chrono::steady_clock::now();
		prevTimePoint = startTimePoint;
	}

	auto Timer::update() noexcept -> void {
		auto now = std::chrono::steady_clock::now();
		uint64_t deltaTimeCount = std::chrono::duration<double, std::micro>(now - prevTimePoint).count();
		_deltaTime = 0.000001 * deltaTimeCount;
		prevTimePoint = now;
	}

	auto Timer::deltaTime() noexcept -> double {
		return _deltaTime;
	}

	auto Timer::totalTime() noexcept -> double {
		uint64_t totalTimeCount = std::chrono::duration<double, std::micro>(prevTimePoint - startTimePoint).count();
		return 0.000001 * totalTimeCount;
	}

#pragma endregion
}