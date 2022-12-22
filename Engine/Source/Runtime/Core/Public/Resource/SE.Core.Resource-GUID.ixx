module;
#include <cstdint>
#include <string>
#include <filesystem>
export module SE.Core.Resource:GUID;

namespace SIByL::Core
{
	/** globally unique identifier */
	export using GUID = uint64_t;
	export inline GUID INVALID_GUID = uint64_t(-1);

	export inline auto hashGUID(std::filesystem::path const& path) noexcept -> GUID;
}