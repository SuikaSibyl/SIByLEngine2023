module;
#include <memory>
export module Image.Image;
import Image.Color;
import Core.Memory;

namespace SIByL::Image
{
	export template <class ColorStruct>
	struct Image {
		Image(size_t width, size_t height, size_t channel = 4);
		auto operator[](size_t i) -> ColorStruct*;

		size_t width, height, channel;
		ColorType type;
		Core::Buffer data;
	};

	template <class ColorStruct>
	Image<ColorStruct>::Image(size_t width, size_t height, size_t channel)
		:width(width), height(height), channel(channel)
	{
		data = Core::Buffer(width * height * sizeof(ColorStruct));
		memset(data.data, 0, data.size);
	}

	template <class ColorStruct>
	auto Image<ColorStruct>::operator[](size_t i)->ColorStruct* {
		return &(reinterpret_cast<ColorStruct*>(data.data)[i * width]);
	}

}