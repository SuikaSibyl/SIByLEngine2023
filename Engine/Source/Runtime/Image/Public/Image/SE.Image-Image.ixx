module;
#include <vector>
#include <memory>
export module SE.Image:Image;
import :Color;
import SE.Core.Memory;
import SE.RHI;

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

	export struct Texture_Host {
		RHI::Extend3D extend;
		RHI::TextureFormat format;
		RHI::TextureDimension dimension;
		Core::Buffer buffer;
		uint32_t mip_levels = 0;
		uint32_t array_layers = 0;
		uint32_t data_offset = 0;
		uint32_t data_size = 0;

		auto getDescriptor() noexcept -> RHI::TextureDescriptor {
			return RHI::TextureDescriptor{
				extend, mip_levels, 1, dimension,
				format,
				(uint32_t)RHI::TextureUsage::COPY_DST | (uint32_t)RHI::TextureUsage::TEXTURE_BINDING, {format}
			};
		}

		auto getData() noexcept -> char const* {
			return &(static_cast<char const*>(buffer.data)[data_offset]);
		}

		struct SubResource {
			uint32_t mip;
			uint32_t level;
			uint32_t offset;
			uint32_t size;
			uint32_t width;
			uint32_t height;
		};
		std::vector<SubResource> subResources;
	};
}