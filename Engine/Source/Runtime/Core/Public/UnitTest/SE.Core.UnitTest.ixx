module;
#include <format>
#include <string>
#include <vector>
#include <unordered_map>
#include <functional>
export module SE.Core.UnitTest;
import SE.Core.Log;
import SE.Core.System;

namespace SIByL::Core
{
	export using UnitTest = bool;

	export struct UnitTestManager :public Manager {

		static auto run() noexcept -> void {
			for (auto test : allTests) {
				Core::LogManager::Correct(std::format(""));
				testcase_passed = testcase_tested = 0;
				test.second();
				if (testcase_passed == testcase_tested) {
					Core::LogManager::Correct(std::format("Unit Test {0}: all ({1}, {2}) assertion Passed", test.first, testcase_passed, testcase_tested));
				}
				else {
					Core::LogManager::Error(std::format("Unit Test {0}: only ({1}, {2}) assertion Passed", test.first, testcase_passed, testcase_tested));
				}
			}
		}

		static auto Assert(bool pass, char const* desc) noexcept -> void {
			++testcase_tested;
			if (pass) {
				Core::LogManager::Correct("\t Passed: " + std::string(desc));
				++testcase_passed;
			}
			else
				Core::LogManager::Error("\t Failed: " + std::string(desc));
		}

		static auto NewTestCase(char const* desc, std::function<void()> const& function) noexcept -> UnitTest {
			allTests[desc] = function;
			return true;
		}

		static std::unordered_map<char const*, std::function<void()>> allTests;

	private:
		static size_t testcase_passed;
		static size_t testcase_tested;
	};

	size_t UnitTestManager::testcase_passed = 0;
	size_t UnitTestManager::testcase_tested = 0;
	std::unordered_map<char const*, std::function<void()>> UnitTestManager::allTests = {};
}