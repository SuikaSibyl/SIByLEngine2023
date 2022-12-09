module;
#include <cstdint>
export module SE.Image:Color;
import SE.Math.Geometric;

namespace SIByL::Image
{
	export enum struct ColorType {
		COLOR_R8G8B8_UINT,
		COLOR_R8G8B8A8_UINT,
		COLOR_R32G32B32_FLOAT,
	};

	export struct COLOR_R8G8B8_UINT :public Math::Vector3<uint8_t> {};
	export struct COLOR_R8G8B8A8_UINT :public Math::Vector4<uint8_t> {};
	export struct COLOR_R32G32B32_FLOAT :public Math::Vector3<float> {};
}
