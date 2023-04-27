module;
#include <ddspp.h>
#include <filesystem>
export module SE.Image:DDS;
import :Color;
import :Image;
import SE.Core.Log;
import SE.Core.Memory;
import SE.Core.IO;
import SE.RHI;

namespace SIByL::Image
{
    export struct DDS {
        static auto fromDDS(std::filesystem::path const& path) noexcept -> std::unique_ptr<Texture_Host>;

    };

    inline auto getRHIFormat(ddspp::DXGIFormat format) noexcept -> RHI::TextureFormat {
        switch (format)
        {
        case ddspp::UNKNOWN:                    return RHI::TextureFormat::UNKOWN;
        case ddspp::R32G32B32A32_TYPELESS:      return RHI::TextureFormat::RGBA32_FLOAT;
        case ddspp::R32G32B32A32_FLOAT:         return RHI::TextureFormat::RGBA32_FLOAT;
        case ddspp::R32G32B32A32_UINT:          return RHI::TextureFormat::RGBA32_UINT;
        case ddspp::R32G32B32A32_SINT:          return RHI::TextureFormat::RGBA32_SINT;
        case ddspp::R16G16B16A16_FLOAT:         return RHI::TextureFormat::RGBA16_FLOAT;
        case ddspp::R16G16B16A16_UINT:          return RHI::TextureFormat::RGBA16_UINT;
        case ddspp::R16G16B16A16_SINT:          return RHI::TextureFormat::RGBA16_SINT;
        case ddspp::R32G32_FLOAT:               return RHI::TextureFormat::RG32_FLOAT;
        case ddspp::R32G32_UINT:                return RHI::TextureFormat::RG32_UINT;
        case ddspp::R32G32_SINT:                return RHI::TextureFormat::RG32_SINT;
        case ddspp::D32_FLOAT_S8X24_UINT:       return RHI::TextureFormat::DEPTH32STENCIL8;
        case ddspp::R32_FLOAT_X8X24_TYPELESS:   return RHI::TextureFormat::R32_FLOAT;
        case ddspp::R10G10B10A2_UNORM:          return RHI::TextureFormat::RGB10A2_UNORM;
        case ddspp::R11G11B10_FLOAT:            return RHI::TextureFormat::RG11B10_UFLOAT;
        case ddspp::R8G8B8A8_UNORM:             return RHI::TextureFormat::RGBA8_UNORM;
        case ddspp::R8G8B8A8_UNORM_SRGB:        return RHI::TextureFormat::RGBA8_UNORM_SRGB;
        case ddspp::R8G8B8A8_UINT:              return RHI::TextureFormat::RGBA8_UINT;
        case ddspp::R8G8B8A8_SNORM:             return RHI::TextureFormat::RGBA8_SNORM;
        case ddspp::R8G8B8A8_SINT:              return RHI::TextureFormat::RGBA8_SINT;
        case ddspp::R16G16_FLOAT:               return RHI::TextureFormat::RG16_FLOAT;
        case ddspp::R16G16_UINT:                return RHI::TextureFormat::RG16_UINT;
        case ddspp::R16G16_SINT:                return RHI::TextureFormat::RG16_SINT;
        case ddspp::D32_FLOAT:                  return RHI::TextureFormat::DEPTH32_FLOAT;
        case ddspp::R32_FLOAT:                  return RHI::TextureFormat::R32_FLOAT;
        case ddspp::R32_UINT:                   return RHI::TextureFormat::R32_UINT;
        case ddspp::R32_SINT:                   return RHI::TextureFormat::R32_SINT;
        case ddspp::D24_UNORM_S8_UINT:          return RHI::TextureFormat::DEPTH24STENCIL8;
        case ddspp::R8G8_UNORM:                 return RHI::TextureFormat::RG8_UNORM;
        case ddspp::R8G8_UINT:                  return RHI::TextureFormat::RG8_UINT;
        case ddspp::R8G8_SNORM:                 return RHI::TextureFormat::RG8_SNORM;
        case ddspp::R8G8_SINT:                  return RHI::TextureFormat::RG8_SINT;
        case ddspp::R16_FLOAT:                  return RHI::TextureFormat::R16_FLOAT;
        case ddspp::D16_UNORM:                  return RHI::TextureFormat::DEPTH16_UNORM;
        case ddspp::R16_UINT:                   return RHI::TextureFormat::R16_UINT;
        case ddspp::R16_SINT:                   return RHI::TextureFormat::R16_SINT;
        case ddspp::R8_UNORM:                   return RHI::TextureFormat::R8_UNORM;
        case ddspp::R8_UINT:                    return RHI::TextureFormat::R8_UINT;
        case ddspp::R8_SNORM:                   return RHI::TextureFormat::R8_SNORM;
        case ddspp::R8_SINT:                    return RHI::TextureFormat::R8_SINT;
        case ddspp::BC1_UNORM:                  return RHI::TextureFormat::BC1_RGBA_UNORM_BLOCK;
        case ddspp::BC1_UNORM_SRGB:             return RHI::TextureFormat::BC1_RGBA_SRGB_BLOCK;
        case ddspp::BC2_UNORM:                  return RHI::TextureFormat::BC2_UNORM_BLOCK;
        case ddspp::BC2_UNORM_SRGB:             return RHI::TextureFormat::BC2_SRGB_BLOCK;
        case ddspp::BC3_UNORM:                  return RHI::TextureFormat::BC3_UNORM_BLOCK;
        case ddspp::BC3_UNORM_SRGB:             return RHI::TextureFormat::BC3_SRGB_BLOCK;
        case ddspp::BC4_UNORM:                  return RHI::TextureFormat::BC4_UNORM_BLOCK;
        case ddspp::BC4_SNORM:                  return RHI::TextureFormat::BC4_SNORM_BLOCK;
        case ddspp::BC5_UNORM:                  return RHI::TextureFormat::BC5_UNORM_BLOCK;
        case ddspp::BC5_SNORM:                  return RHI::TextureFormat::BC5_SNORM_BLOCK;
        case ddspp::BC6H_UF16:                  return RHI::TextureFormat::BC6H_UFLOAT_BLOCK;
        case ddspp::BC6H_SF16:                  return RHI::TextureFormat::BC6H_SFLOAT_BLOCK;
        case ddspp::BC7_UNORM:                  return RHI::TextureFormat::BC7_UNORM_BLOCK;
        case ddspp::BC7_UNORM_SRGB:             return RHI::TextureFormat::BC7_SRGB_BLOCK;
        case ddspp::R32G32B32_TYPELESS:
        case ddspp::R32G32B32_FLOAT:
        case ddspp::R32G32B32_UINT:
        case ddspp::R32G32B32_SINT:
        case ddspp::R16G16B16A16_UNORM:
        case ddspp::R16G16B16A16_SNORM:         
        case ddspp::R16G16B16A16_TYPELESS:
        case ddspp::R32G32_TYPELESS:
        case ddspp::R32G8X24_TYPELESS:
        case ddspp::X32_TYPELESS_G8X24_UINT:
        case ddspp::R10G10B10A2_TYPELESS:
        case ddspp::R10G10B10A2_UINT:
        case ddspp::R8G8B8A8_TYPELESS:
        case ddspp::R16G16_TYPELESS:
        case ddspp::R16G16_UNORM:
        case ddspp::R16G16_SNORM:
        case ddspp::R32_TYPELESS:
        case ddspp::R24G8_TYPELESS:
        case ddspp::R24_UNORM_X8_TYPELESS:
        case ddspp::X24_TYPELESS_G8_UINT:
        case ddspp::R8G8_TYPELESS:
        case ddspp::R16_TYPELESS:
        case ddspp::R16_UNORM:
        case ddspp::R16_SNORM:
        case ddspp::R8_TYPELESS:
        case ddspp::A8_UNORM:
        case ddspp::R1_UNORM:
        case ddspp::R9G9B9E5_SHAREDEXP:
        case ddspp::R8G8_B8G8_UNORM:
        case ddspp::G8R8_G8B8_UNORM:
        case ddspp::BC1_TYPELESS:
        case ddspp::BC2_TYPELESS:
        case ddspp::BC3_TYPELESS:
        case ddspp::BC4_TYPELESS:
        case ddspp::BC5_TYPELESS:
        case ddspp::B5G6R5_UNORM:
        case ddspp::B5G5R5A1_UNORM:
        case ddspp::B8G8R8A8_UNORM:
        case ddspp::B8G8R8X8_UNORM:
        case ddspp::R10G10B10_XR_BIAS_A2_UNORM:
        case ddspp::B8G8R8A8_TYPELESS:
        case ddspp::B8G8R8A8_UNORM_SRGB:
        case ddspp::B8G8R8X8_TYPELESS:
        case ddspp::B8G8R8X8_UNORM_SRGB:
        case ddspp::BC6H_TYPELESS:
        case ddspp::BC7_TYPELESS:
        case ddspp::AYUV:
        case ddspp::Y410:
        case ddspp::Y416:
        case ddspp::NV12:
        case ddspp::P010:
        case ddspp::P016:
        case ddspp::OPAQUE_420:
        case ddspp::YUY2:
        case ddspp::Y210:
        case ddspp::Y216:
        case ddspp::NV11:
        case ddspp::AI44:
        case ddspp::IA44:
        case ddspp::P8:
        case ddspp::A8P8:
        case ddspp::B4G4R4A4_UNORM:
        case ddspp::P208:
        case ddspp::V208:
        case ddspp::V408:
        case ddspp::ASTC_4X4_TYPELESS:
        case ddspp::ASTC_4X4_UNORM:
        case ddspp::ASTC_4X4_UNORM_SRGB:
        case ddspp::ASTC_5X4_TYPELESS:
        case ddspp::ASTC_5X4_UNORM:
        case ddspp::ASTC_5X4_UNORM_SRGB:
        case ddspp::ASTC_5X5_TYPELESS:
        case ddspp::ASTC_5X5_UNORM:
        case ddspp::ASTC_5X5_UNORM_SRGB:
        case ddspp::ASTC_6X5_TYPELESS:
        case ddspp::ASTC_6X5_UNORM:
        case ddspp::ASTC_6X5_UNORM_SRGB:
        case ddspp::ASTC_6X6_TYPELESS:
        case ddspp::ASTC_6X6_UNORM:
        case ddspp::ASTC_6X6_UNORM_SRGB:
        case ddspp::ASTC_8X5_TYPELESS:
        case ddspp::ASTC_8X5_UNORM:
        case ddspp::ASTC_8X5_UNORM_SRGB:
        case ddspp::ASTC_8X6_TYPELESS:
        case ddspp::ASTC_8X6_UNORM:
        case ddspp::ASTC_8X6_UNORM_SRGB:
        case ddspp::ASTC_8X8_TYPELESS:
        case ddspp::ASTC_8X8_UNORM:
        case ddspp::ASTC_8X8_UNORM_SRGB:
        case ddspp::ASTC_10X5_TYPELESS:
        case ddspp::ASTC_10X5_UNORM:
        case ddspp::ASTC_10X5_UNORM_SRGB:
        case ddspp::ASTC_10X6_TYPELESS:
        case ddspp::ASTC_10X6_UNORM:
        case ddspp::ASTC_10X6_UNORM_SRGB:
        case ddspp::ASTC_10X8_TYPELESS:
        case ddspp::ASTC_10X8_UNORM:
        case ddspp::ASTC_10X8_UNORM_SRGB:
        case ddspp::ASTC_10X10_TYPELESS:
        case ddspp::ASTC_10X10_UNORM:
        case ddspp::ASTC_10X10_UNORM_SRGB:
        case ddspp::ASTC_12X10_TYPELESS:
        case ddspp::ASTC_12X10_UNORM:
        case ddspp::ASTC_12X10_UNORM_SRGB:
        case ddspp::ASTC_12X12_TYPELESS:
        case ddspp::ASTC_12X12_UNORM:
        case ddspp::ASTC_12X12_UNORM_SRGB:
        case ddspp::FORCE_UINT:
        default: {
            Core::LogManager::Error("DDS Load failed due to unsupported texture format.");
            return RHI::TextureFormat::UNKOWN;
        }
        }
    }

    auto DDS::fromDDS(std::filesystem::path const& path) noexcept -> std::unique_ptr<Texture_Host> {
        // Load image as a stream of bytes
        Core::Buffer dds_file;
        Core::syncReadFile(path.string().c_str(), dds_file);
        unsigned char* ddsData = static_cast<unsigned char*>(dds_file.data);
        // Decode header and get pointer to initial data
        ddspp::Descriptor desc;
        ddspp::Result decodeResult = ddspp::decode_header(ddsData, desc);
        // Feed to the graphics API
        if (decodeResult == ddspp::Success) {
            const unsigned char* initialData = ddsData + desc.headerSize;
            std::unique_ptr<Texture_Host> texture = std::make_unique<Texture_Host>();
            texture->buffer = std::move(dds_file);
            texture->data_offset = desc.headerSize;
            texture->data_size = texture->buffer.size - texture->data_offset;
            texture->format = getRHIFormat(desc.format);
            texture->extend = { desc.width, desc.height, desc.depth };
            texture->mip_levels = desc.numMips;
            //texture->mip_levels = 1;
            texture->array_layers = desc.arraySize;
            if (desc.type == ddspp::Texture2D)
                texture->dimension = RHI::TextureDimension::TEX2D;

            uint32_t width = desc.width;
            uint32_t height = desc.height;
            uint32_t offset = 0;
            for (uint32_t i = 0; i < texture->mip_levels; ++i) {
                uint32_t rowLength = (width + desc.blockWidth - 1) / desc.blockWidth;
                uint32_t imageHeight = (height + desc.blockHeight - 1) / desc.blockHeight;
                uint32_t currentSize = rowLength * imageHeight * desc.bitsPerPixelOrBlock / 8;

                texture->subResources.emplace_back(Texture_Host::SubResource{ i, 0, offset, currentSize, width, height });

                offset += currentSize;
                width = std::max(width >> 1, 1U);
                height = std::max(height >> 1, 1U);
            }

            return texture;
        }
    }
}