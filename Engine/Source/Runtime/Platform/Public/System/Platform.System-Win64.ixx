module;
#include <string>
#ifdef _WIN32
#include <Windows.h>
#endif
export module Platform.System:Win64;

#ifdef _WIN32
namespace SIByL::Platform
{
	export inline auto getNumSystemCores() noexcept -> int {
		SYSTEM_INFO sysInfo;
		GetSystemInfo(&sysInfo);
		return sysInfo.dwNumberOfProcessors;
	}
}
#endif