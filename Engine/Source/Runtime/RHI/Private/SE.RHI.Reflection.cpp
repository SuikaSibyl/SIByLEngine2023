#include "../Public/SE.RHI.Reflection.hpp"

namespace SIByL {
auto to_string(RHI::TextureFormat format) noexcept -> std::string {
  switch (format) {
    case SIByL::RHI::TextureFormat::UNKOWN: return "UNKOWN";
    case SIByL::RHI::TextureFormat::R8_UNORM: return "R8_UNORM";
    case SIByL::RHI::TextureFormat::R8_SNORM: return "R8_SNORM";
    case SIByL::RHI::TextureFormat::R8_UINT: return "R8_UINT";
    case SIByL::RHI::TextureFormat::R8_SINT: return "R8_SINT";
    case SIByL::RHI::TextureFormat::R16_UINT: return "R16_UINT";
    case SIByL::RHI::TextureFormat::R16_SINT: return "R16_SINT";
    case SIByL::RHI::TextureFormat::R16_FLOAT: return "R16_FLOAT";
    case SIByL::RHI::TextureFormat::RG8_UNORM: return "RG8_UNORM";
    case SIByL::RHI::TextureFormat::RG8_SNORM: return "RG8_SNORM";
    case SIByL::RHI::TextureFormat::RG8_UINT: return "RG8_UINT";
    case SIByL::RHI::TextureFormat::RG8_SINT: return "RG8_SINT";
    case SIByL::RHI::TextureFormat::R32_UINT: return "R32_UINT";
    case SIByL::RHI::TextureFormat::R32_SINT: return "R32_SINT";
    case SIByL::RHI::TextureFormat::R32_FLOAT: return "R32_FLOAT";
    case SIByL::RHI::TextureFormat::RG16_UINT: return "RG16_UINT";
    case SIByL::RHI::TextureFormat::RG16_SINT: return "RG16_SINT";
    case SIByL::RHI::TextureFormat::RG16_FLOAT: return "RG16_FLOAT";
    case SIByL::RHI::TextureFormat::RGBA8_UNORM: return "RGBA8_UNORM";
    case SIByL::RHI::TextureFormat::RGBA8_UNORM_SRGB: return "RGBA8_UNORM_SRGB";
    case SIByL::RHI::TextureFormat::RGBA8_SNORM: return "RGBA8_SNORM";
    case SIByL::RHI::TextureFormat::RGBA8_UINT: return "RGBA8_UINT";
    case SIByL::RHI::TextureFormat::RGBA8_SINT: return "RGBA8_SINT";
    case SIByL::RHI::TextureFormat::BGRA8_UNORM: return "BGRA8_UNORM";
    case SIByL::RHI::TextureFormat::BGRA8_UNORM_SRGB: return "BGRA8_UNORM_SRGB";
    case SIByL::RHI::TextureFormat::RGB9E5_UFLOAT: return "RGB9E5_UFLOAT";
    case SIByL::RHI::TextureFormat::RG11B10_UFLOAT: return "RG11B10_UFLOAT";
    case SIByL::RHI::TextureFormat::RG32_UINT: return "RG32_UINT";
    case SIByL::RHI::TextureFormat::RG32_SINT: return "RG32_SINT";
    case SIByL::RHI::TextureFormat::RG32_FLOAT: return "RG32_FLOAT";
    case SIByL::RHI::TextureFormat::RGBA16_UINT: return "RGBA16_UINT";
    case SIByL::RHI::TextureFormat::RGBA16_SINT: return "RGBA16_SINT";
    case SIByL::RHI::TextureFormat::RGBA16_FLOAT: return "RGBA16_FLOAT";
    case SIByL::RHI::TextureFormat::RGBA32_UINT: return "RGBA32_UINT";
    case SIByL::RHI::TextureFormat::RGBA32_SINT: return "RGBA32_SINT";
    case SIByL::RHI::TextureFormat::RGBA32_FLOAT: return "RGBA32_FLOAT";
    case SIByL::RHI::TextureFormat::STENCIL8: return "STENCIL8";
    case SIByL::RHI::TextureFormat::DEPTH16_UNORM: return "DEPTH16_UNORM";
    case SIByL::RHI::TextureFormat::DEPTH24: return "DEPTH24";
    case SIByL::RHI::TextureFormat::DEPTH24STENCIL8: return "DEPTH24STENCIL8";
    case SIByL::RHI::TextureFormat::DEPTH32_FLOAT: return "DEPTH32_FLOAT";
    case SIByL::RHI::TextureFormat::COMPRESSION: return "COMPRESSION";
    case SIByL::RHI::TextureFormat::RGB10A2_UNORM: return "RGB10A2_UNORM";
    case SIByL::RHI::TextureFormat::DEPTH32STENCIL8: return "DEPTH32STENCIL8";
    case SIByL::RHI::TextureFormat::BC1_RGB_UNORM_BLOCK: return "BC1_RGB_UNORM_BLOCK";
    case SIByL::RHI::TextureFormat::BC1_RGB_SRGB_BLOCK: return "BC1_RGB_SRGB_BLOCK";
    case SIByL::RHI::TextureFormat::BC1_RGBA_UNORM_BLOCK: return "BC1_RGBA_UNORM_BLOCK";
    case SIByL::RHI::TextureFormat::BC1_RGBA_SRGB_BLOCK: return "BC1_RGBA_SRGB_BLOCK";
    case SIByL::RHI::TextureFormat::BC2_UNORM_BLOCK: return "BC2_UNORM_BLOCK";
    case SIByL::RHI::TextureFormat::BC2_SRGB_BLOCK: return "BC2_SRGB_BLOCK";
    case SIByL::RHI::TextureFormat::BC3_UNORM_BLOCK: return "BC3_UNORM_BLOCK";
    case SIByL::RHI::TextureFormat::BC3_SRGB_BLOCK: return "BC3_SRGB_BLOCK";
    case SIByL::RHI::TextureFormat::BC4_UNORM_BLOCK: return "BC4_UNORM_BLOCK";
    case SIByL::RHI::TextureFormat::BC4_SNORM_BLOCK: return "BC4_SNORM_BLOCK";
    case SIByL::RHI::TextureFormat::BC5_UNORM_BLOCK: return "BC5_UNORM_BLOCK";
    case SIByL::RHI::TextureFormat::BC5_SNORM_BLOCK: return "BC5_SNORM_BLOCK";
    case SIByL::RHI::TextureFormat::BC6H_UFLOAT_BLOCK: return "BC6H_UFLOAT_BLOCK";
    case SIByL::RHI::TextureFormat::BC6H_SFLOAT_BLOCK: return "BC6H_SFLOAT_BLOCK";
    case SIByL::RHI::TextureFormat::BC7_UNORM_BLOCK: return "BC7_UNORM_BLOCK";
    case SIByL::RHI::TextureFormat::BC7_SRGB_BLOCK: return "BC7_SRGB_BLOCK";
    default: return "ERROR";
  }
}
}