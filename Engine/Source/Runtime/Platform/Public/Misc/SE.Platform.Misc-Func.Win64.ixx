module;
#include <string>
#include <codecvt>
#ifdef _WIN32
#include <Windows.h>
#endif
export module SE.Platform.Misc:Func.Win64;

#ifdef _WIN32
namespace SIByL::Platform
{
    /** get number of CPU cores */
	export inline auto getNumSystemCores() noexcept -> int {
		SYSTEM_INFO sysInfo;
		GetSystemInfo(&sysInfo);
		return sysInfo.dwNumberOfProcessors;
	}

    /** convert utf-8 to utf-16 */
    export inline auto string_cast(const std::string& utf8str) noexcept -> std::wstring {
        if (utf8str.empty()) return L"";
        int char_count = MultiByteToWideChar(CP_UTF8, 0, &utf8str[0], (int)utf8str.size(), NULL, 0);
        std::wstring conv(char_count, 0);
        MultiByteToWideChar(CP_UTF8, 0, &utf8str[0], (int)utf8str.size(), &conv[0], char_count);
        return conv;
    }

    /** convert utf-16 to utf-8 */
    export inline auto string_cast(const std::wstring& utf16str) noexcept -> std::string {
        if (utf16str.empty()) return "";
        int char_count = WideCharToMultiByte(CP_UTF8, 0, &utf16str[0], (int)utf16str.size(), NULL, 0, NULL, NULL);
        std::string conv(char_count, 0);
        WideCharToMultiByte(CP_UTF8, 0, &utf16str[0], (int)utf16str.size(), &conv[0], char_count, NULL, NULL);
        return conv;
    }
}
#endif