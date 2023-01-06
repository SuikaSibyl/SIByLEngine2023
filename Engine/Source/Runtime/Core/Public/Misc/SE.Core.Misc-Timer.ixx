module;
#include <chrono>
#include <string>
#include <iostream>
export module SE.Core.Misc:Timer;

namespace SIByL::Core
{
	/** Timer for the engine.
	* Tick every frame to get delta time. */
	export struct Timer {
		Timer();
		/** update the timer */
		inline auto update() noexcept -> void;
		/** get the delta time */
		inline auto deltaTime() noexcept -> double;
		/** get the total time */
		inline auto totalTime() noexcept -> double;
	private:
		/** start time point record */
		std::chrono::steady_clock::time_point startTimePoint;
		/** previous time point record */
		std::chrono::steady_clock::time_point prevTimePoint;
		/** delta time between this and prev tick */
		double _deltaTime = 0.f;
	};

	/** A world time point record. */
	export struct WorldTimePoint {
		/** get current world time point */
		static inline auto get() noexcept -> WorldTimePoint;
		/** output the current time point to a string */
		inline auto to_string() noexcept -> std::string;
		std::chrono::years y;
		std::chrono::days d;
		std::chrono::hours h;
		std::chrono::minutes m;
		std::chrono::seconds s;
	};

#pragma region TIMER_IMPL

	Timer::Timer() {
		startTimePoint = std::chrono::steady_clock::now();
		prevTimePoint = startTimePoint;
	}

	inline auto Timer::update() noexcept -> void {
		auto const now = std::chrono::steady_clock::now();
		uint64_t const deltaTimeCount = uint64_t(std::chrono::duration<double, std::micro>(now - prevTimePoint).count());
		_deltaTime = 0.000001 * deltaTimeCount;
		prevTimePoint = now;
	}

	inline auto Timer::deltaTime() noexcept -> double {
		return _deltaTime;
	}

	inline auto Timer::totalTime() noexcept -> double {
		uint64_t const totalTimeCount = uint64_t(std::chrono::duration<double, std::micro>(prevTimePoint - startTimePoint).count());
		return 0.000001 * totalTimeCount;
	}

#pragma endregion

#pragma region WORLD_TIME_POINT_IMPL

	inline auto WorldTimePoint::get() noexcept -> WorldTimePoint {
		WorldTimePoint wtp;
		using namespace std;
		using namespace std::chrono;
		typedef duration<int, ratio_multiply<hours::period, ratio<24> >::type> days;
		system_clock::time_point now = system_clock::now();
		system_clock::duration tp = now.time_since_epoch();
		wtp.y = duration_cast<years>(tp);
		tp -= wtp.y;
		wtp.d = duration_cast<days>(tp);
		tp -= wtp.d;
		wtp.h = duration_cast<hours>(tp);
		tp -= wtp.h;
		wtp.m = duration_cast<minutes>(tp);
		tp -= wtp.m;
		wtp.s = duration_cast<seconds>(tp);
		tp -= wtp.s;
		return wtp;
	}

	inline auto WorldTimePoint::to_string() noexcept -> std::string {
		std::string str;
		str += std::to_string(y.count());
		str += std::to_string(d.count());
		str += std::to_string(h.count());
		str += std::to_string(m.count());
		str += std::to_string(s.count());
		return str;
	}

#pragma endregion
}