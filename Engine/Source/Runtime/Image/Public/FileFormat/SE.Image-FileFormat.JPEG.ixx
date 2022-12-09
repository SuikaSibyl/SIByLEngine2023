module;
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#include <filesystem>
export module SE.Image:JPEG;
import :Color;
import :Image;
import SE.Core.Log;
import SE.Core.Memory;

namespace SIByL::Image
{
	export struct JPEG {
		static auto toJPEG(Image<COLOR_R8G8B8_UINT> const& i) noexcept -> Core::Buffer;
		static auto toJPEG(Image<COLOR_R8G8B8A8_UINT> const& i) noexcept -> Core::Buffer;

		static auto fromJPEG(std::filesystem::path const& path) noexcept -> std::unique_ptr<Image<COLOR_R8G8B8A8_UINT>>;
	};

#pragma region IMAGE_JPEG_IMPL

    auto JPEG::fromJPEG(std::filesystem::path const& path) noexcept -> std::unique_ptr<Image<COLOR_R8G8B8A8_UINT>> {
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

#pragma endregion
}