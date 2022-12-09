module;
#include <stb_image.h>
#include <filesystem>
export module SE.Image:PNG;
import :Color;
import :Image;
import SE.Core.Log;
import SE.Core.Memory;

namespace SIByL::Image
{
	export struct PNG {
		static auto fromPNG(std::filesystem::path const& path) noexcept -> std::unique_ptr<Image<COLOR_R8G8B8A8_UINT>>;
	};

    auto PNG::fromPNG(std::filesystem::path const& path) noexcept -> std::unique_ptr<Image<COLOR_R8G8B8A8_UINT>> {
        int texWidth, texHeight, texChannels;
        stbi_uc* pixels = stbi_load(path.string().c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
        if (!pixels) {
            Core::LogManager::Error("Image :: failed to load texture image!");
            return nullptr;
        }
        std::unique_ptr<Image<COLOR_R8G8B8A8_UINT>> image = std::make_unique<Image<COLOR_R8G8B8A8_UINT>>(texWidth, texHeight);
        image->data = Core::Buffer(texWidth * texHeight * sizeof(COLOR_R8G8B8A8_UINT));
        memcpy(image->data.data, pixels, texWidth * texHeight * sizeof(COLOR_R8G8B8A8_UINT));
        stbi_image_free(pixels);
        return std::move(image);
    }
}