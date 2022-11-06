module;
#include <string>
#include <Windows.h>
export module Platform.System:Win64;

namespace SIByL::Platform
{
	export inline auto getNumSystemCores() noexcept -> int {
		SYSTEM_INFO sysInfo;
		GetSystemInfo(&sysInfo);
		return sysInfo.dwNumberOfProcessors;
	}
}