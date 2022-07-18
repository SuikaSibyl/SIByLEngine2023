export module Image.FileFormat:PFM;
import Core.Memory;
import Image.Color;
import Image.Image;

namespace SIByL::Image
{
	export class PFM
	{
	public:
		static auto toPFM(Image<COLOR_R32G32B32_FLOAT> const& i) noexcept -> Core::Buffer;
	};
}