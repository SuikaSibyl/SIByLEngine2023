#pragma once

#include <memory>
#include <vector>
#include <cstdint>
#include <filesystem>
#include <SE.Math.Geometric.hpp>
#include <SE.RHI.hpp>

namespace SIByL::Image {
SE_EXPORT enum struct ColorType {
  COLOR_R8G8B8_UINT,
  COLOR_R8G8B8A8_UINT,
  COLOR_R32G32B32_FLOAT,
};

SE_EXPORT struct COLOR_R8G8B8_UINT : public Math::Vector3<uint8_t> {};
SE_EXPORT struct COLOR_R8G8B8A8_UINT : public Math::Vector4<uint8_t> {};
SE_EXPORT struct COLOR_R32G32B32_FLOAT : public Math::Vector3<float> {};


SE_EXPORT template <class ColorStruct>
struct Image {
  Image(size_t width, size_t height, size_t channel = 4);
  auto operator[](size_t i) -> ColorStruct*;

  size_t width, height, channel;
  ColorType type;
  Core::Buffer data;
};

template <class ColorStruct>
Image<ColorStruct>::Image(size_t width, size_t height, size_t channel)
    : width(width), height(height), channel(channel) {
  data = Core::Buffer(width * height * sizeof(ColorStruct));
  memset(data.data, 0, data.size);
}

template <class ColorStruct>
auto Image<ColorStruct>::operator[](size_t i) -> ColorStruct* {
  return &(reinterpret_cast<ColorStruct*>(data.data)[i * width]);
}

SE_EXPORT struct Texture_Host {
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
        extend,
        mip_levels,
        array_layers,
        1,
        dimension,
        format,
        (uint32_t)RHI::TextureUsage::COPY_DST |
            (uint32_t)RHI::TextureUsage::TEXTURE_BINDING,
        {format}};
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

SE_EXPORT struct JPEG {
  static auto toJPEG(Image<COLOR_R8G8B8_UINT> const& i) noexcept
      -> Core::Buffer;
  static auto toJPEG(Image<COLOR_R8G8B8A8_UINT> const& i) noexcept
      -> Core::Buffer;
  static auto fromJPEG(std::filesystem::path const& path) noexcept
      -> std::unique_ptr<Image<COLOR_R8G8B8A8_UINT>>;
};

SE_EXPORT struct PNG {
  static auto fromPNG(std::filesystem::path const& path) noexcept
      -> std::unique_ptr<Image<COLOR_R8G8B8A8_UINT>>;
  static auto writePNG(std::filesystem::path const& path, uint32_t width,
                       uint32_t height, uint32_t channel, float* data) noexcept
      -> void;
};

SE_EXPORT struct TGA {
  static auto fromTGA(std::filesystem::path const& path) noexcept
      -> std::unique_ptr<Image<COLOR_R8G8B8A8_UINT>>;
  static auto writeTGA(std::filesystem::path const& path, uint32_t width,
                       uint32_t height, uint32_t channel, float* data) noexcept
      -> void;
};

SE_EXPORT struct PPM {
  static auto toPPM(Image<COLOR_R8G8B8_UINT> const& i) noexcept -> Core::Buffer;
  static auto writePPM(char const* path, uint32_t width, uint32_t height,
                       uint32_t channel, float* data) noexcept -> void;
};

SE_EXPORT struct PFM {
  static auto toPFM(Image<COLOR_R32G32B32_FLOAT> const& i) noexcept
      -> Core::Buffer;
};

SE_EXPORT struct HDR {
  // static auto toHDR(Image<COLOR_R8G8B8_UINT> const& i) noexcept ->
  // Core::Buffer;
  static auto writeHDR(std::filesystem::path const& path, uint32_t width,
                       uint32_t height, uint32_t channel, float* data) noexcept
      -> void;
};

SE_EXPORT struct EXR {
  static auto writeEXR(std::filesystem::path const& path, uint32_t width,
                       uint32_t height, uint32_t channel, float* data) noexcept
      -> void;
};

SE_EXPORT struct BMP {
  static auto fromBMP(std::filesystem::path const& path) noexcept
      -> std::unique_ptr<Image<COLOR_R8G8B8A8_UINT>>;

  static auto writeBMP(std::filesystem::path const& path, uint32_t width,
                       uint32_t height, uint32_t channel, float* data) noexcept
      -> void;
};

SE_EXPORT struct DDS {
  static auto fromDDS(std::filesystem::path const& path) noexcept
      -> std::unique_ptr<Texture_Host>;
};
}  // namespace SIByL::Image

namespace SIByL {
SE_EXPORT struct ImageLoader {
  static auto load_rgba8(std::filesystem::path const& path) noexcept
      -> std::unique_ptr<Image::Image<Image::COLOR_R8G8B8A8_UINT>>;
};
}  // namespace SIByL