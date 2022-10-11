module;
#include <cstdint>
#include <string>
#include <filesystem>
export module Core.Resource.RuntimeManage:GUID;

namespace SIByL::Core
{
	/** globally unique identifier */
	export using GUID = uint64_t;

	export inline auto hashGUID(std::filesystem::path const& path) noexcept -> GUID;
}