module;
#include <string>
export module SE.Image:PPM;
import :Color;
import :Image;
import SE.Core.Memory;

namespace SIByL::Image
{
	export struct PPM {
		static auto toPPM(Image<COLOR_R8G8B8_UINT> const& i) noexcept -> Core::Buffer;
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

#pragma endregion

}