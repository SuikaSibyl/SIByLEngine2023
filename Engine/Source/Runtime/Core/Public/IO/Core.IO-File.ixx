module;
#include <filesystem>
export module Core.IO:FileSystem;

namespace SIByL::Core
{
	export using filepath = std::filesystem::path;

}