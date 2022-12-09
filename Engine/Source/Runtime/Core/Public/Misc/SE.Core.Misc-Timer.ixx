module;
#include <chrono>
#include <string>
#include <iostream>
export module SE.Core.Misc:Timer;

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

	export struct WorldTimePoint {
		static auto get() noexcept -> WorldTimePoint;
		auto to_string() noexcept -> std::string;
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

	auto Timer::update() noexcept -> void {
		auto now = std::chrono::steady_clock::now();
		uint64_t deltaTimeCount = uint64_t(std::chrono::duration<double, std::micro>(now - prevTimePoint).count());
		_deltaTime = 0.000001 * deltaTimeCount;
		prevTimePoint = now;
	}

	auto Timer::deltaTime() noexcept -> double {
		return _deltaTime;
	}

	auto Timer::totalTime() noexcept -> double {
		uint64_t totalTimeCount = uint64_t(std::chrono::duration<double, std::micro>(prevTimePoint - startTimePoint).count());
		return 0.000001 * totalTimeCount;
	}

#pragma endregion

	auto WorldTimePoint::get() noexcept -> WorldTimePoint {
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

	auto WorldTimePoint::to_string() noexcept -> std::string {
		std::string str;
		str += std::to_string(y.count());
		str += std::to_string(d.count());
		str += std::to_string(h.count());
		str += std::to_string(m.count());
		str += std::to_string(s.count());
		return str;
	}
}