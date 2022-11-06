module;
#pragma warning(disable:4996)
#include <filesystem>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
export module Image.FileFormat:HDR;
import Core.Memory;
import Image.Color;
import Image.Image;

namespace SIByL::Image
{
	export struct HDR {
		//static auto toHDR(Image<COLOR_R8G8B8_UINT> const& i) noexcept -> Core::Buffer;
		static auto writeHDR(std::filesystem::path const& path, uint32_t width, uint32_t height, uint32_t channel, float* data) noexcept -> void;
		//static auto fromJPEG(std::filesystem::path const& path) noexcept -> std::unique_ptr<Image<COLOR_R8G8B8A8_UINT>>;
	
	};

	auto HDR::writeHDR(std::filesystem::path const& path, uint32_t width, uint32_t height, uint32_t channel, float* data) noexcept -> void {
		stbi_write_hdr(path.string().c_str(), width, height, channel, reinterpret_cast<float*>(data));
	}

}