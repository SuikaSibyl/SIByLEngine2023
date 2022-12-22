module;
#include <format>
#include <string>
#include <filesystem>
export module SE.Image:Loader;
import :Color;
import :Image;
import :PFM;
import :PPM;
import :PNG;
import :JPEG;
import :HDR;
import SE.Core.Log;
import SE.Core.Resource;

namespace SIByL
{
	export struct ImageLoader {
		static auto load_rgba8(std::filesystem::path const& path) noexcept -> std::unique_ptr<Image::Image<Image::COLOR_R8G8B8A8_UINT>> {
			if (path.extension() == ".jpg" || path.extension() == ".JPG" || path.extension() == ".JPEG")
				return Image::JPEG::fromJPEG(path);
			else if (path.extension() == ".png" || path.extension() == ".PNG")
				return Image::PNG::fromPNG(path);
			else {
				Core::LogManager::Error(std::format("Image :: Image Loader failed when loading {0}, \
					as format extension {1} not supported. ", path.string(), path.extension().string()));
			}
			return nullptr;
		}
	};
}