module;
#include <string>
module Image.FileFormat:PFM;
import Image.FileFormat;
import Core.Memory;
import Image.Color;
import Image.Image;

namespace SIByL::Image
{
	auto PFM::toPFM(Image<COLOR_R32G32B32_FLOAT> const& i) noexcept -> Core::Buffer
	{
		std::string width = std::to_string(i.width);
		std::string height = std::to_string(i.height);
		std::string byte_order = "-1.0";

		Core::Buffer PFMBuffer(width.length() + height.length() +
			byte_order.length() + 6 + i.data.size);

		Core::BufferStream stream = PFMBuffer.stream();
		stream << std::string{ "PF" } << "\n";
		stream << width << " " << height << "\n";
		stream << byte_order << "\n";
		stream << i.data;
		return PFMBuffer;
	}
}