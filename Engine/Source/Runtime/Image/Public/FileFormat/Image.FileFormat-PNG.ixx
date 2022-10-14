module;
#include <filesystem>
export module Image.FileFormat:PNG;
import Core.Memory;
import Image.Color;
import Image.Image;

namespace SIByL::Image
{
	export struct PNG {
		static auto fromPNG(std::filesystem::path const& path) noexcept -> std::unique_ptr<Image<COLOR_R8G8B8A8_UINT>>;
	};
}