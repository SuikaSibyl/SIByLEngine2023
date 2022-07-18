module;
#include <string>
module Image.FileFormat:PPM;
import Image.FileFormat;
import Core.Memory;
import Image.Color;
import Image.Image;

namespace SIByL::Image
{
	auto PPM::toPPM(Image<COLOR_R8G8B8_UINT> const& i) noexcept -> Core::Buffer
	{
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
}