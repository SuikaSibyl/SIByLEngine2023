export module Image.FileFormat:PPM;
import Core.Memory;
import Image.Color;
import Image.Image;

namespace SIByL::Image
{
	export class PPM
	{
	public:
		static auto toPPM(Image<COLOR_R8G8B8_UINT> const& i) noexcept -> Core::Buffer;
	};
}