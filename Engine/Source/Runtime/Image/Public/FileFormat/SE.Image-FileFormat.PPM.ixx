module;
#include <ctime>
#include <string>
#pragma warning(disable:4996)
#include <stb_image_write.h>
export module SE.Image:PPM;
import :Color;
import :Image;
import SE.Core.Memory;
import SE.Core.IO;
import SE.Core.Log;

namespace SIByL::Image
{
	export struct PPM {
		static auto toPPM(Image<COLOR_R8G8B8_UINT> const& i) noexcept -> Core::Buffer;

		static auto writePPM(char const* path, uint32_t width, uint32_t height, uint32_t channel, float* data) noexcept -> void;
	};

#pragma region IMAGE_PPM_IMPL

	auto PPM::toPPM(Image<COLOR_R8G8B8_UINT> const& i) noexcept -> Core::Buffer {
		std::string width = std::to_string(i.width);
		std::string height = std::to_string(i.height);
		std::string max_value = "255";

		Core::Buffer PPMBuffer(width.length() + height.length() +
			max_value.length() + 6 + i.data.size);

		Core::BufferStream stream = PPMBuffer.stream();
		stream << std::string{ "P6" } << "\n";
		stream << width << " " << height << "\n";
		stream << max_value << '\0';
		stream << i.data;
		return PPMBuffer;
	}

	auto PPM::writePPM(char const* path, uint32_t width, uint32_t height, uint32_t channel, float* data) noexcept -> void {

		if (channel != 3) {
			Core::LogManager::Error("Image :: PPM :: PPM does not support alpha channel.");
			return;
		}

		std::string swidth = std::to_string(width);
		std::string sheight = std::to_string(height);
		std::string max_value = "255";

		Core::Buffer PPMPrefixBuffer(swidth.length() + sheight.length() + max_value.length() + 6);
		Core::BufferStream stream = PPMPrefixBuffer.stream();
		stream << std::string{ "P6" } << "\n";
		stream << width << " " << height << "\n";
		stream << max_value << '\0';

		Core::Buffer proxy;
		proxy.data = data;
		proxy.size = width * height * channel * sizeof(uint8_t);

		Core::syncWriteFile(path, std::vector<Core::Buffer*>{&PPMPrefixBuffer, & proxy});

		proxy.data = nullptr;
		proxy.size = 0;
	}

#pragma endregion

}