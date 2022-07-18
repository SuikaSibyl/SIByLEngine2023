module;
#include <cstdint>
export module Image.Color;
import Math.Vector;

namespace SIByL::Image
{
	export enum struct ColorType {
		COLOR_R8G8B8_UNORM,
		COLOR_R8G8B8A8_UNORM,
		COLOR_R32G32B32_FLOAT,
	};

	export using COLOR_R8G8B8_UNORM = Math::Vector3<uint8_t>;
	export using COLOR_R8G8B8A8_UNORM = Math::Vector4<uint8_t>;
	export using COLOR_R32G32B32_FLOAT = Math::Vector3<float>;
}
