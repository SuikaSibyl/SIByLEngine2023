module;
#include <filesystem>
export module Image.FileFormat:JPEG;
import Core.Memory;
import Image.Color;
import Image.Image;

namespace SIByL::Image
{
	export struct JPEG {
		static auto toJPEG(Image<COLOR_R8G8B8_UINT> const& i) noexcept -> Core::Buffer;
		static auto toJPEG(Image<COLOR_R8G8B8A8_UINT> const& i) noexcept -> Core::Buffer;

		static auto fromJPEG(std::filesystem::path const& path) noexcept -> std::unique_ptr<Image<COLOR_R8G8B8A8_UINT>>;
	};
}