module;
#include <stb_image.h>
#include <filesystem>
module Image.FileFormat:PNG;
import Core.Log;
import Core.Memory;
import Image.Color;
import Image.Image;

namespace SIByL::Image
{
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