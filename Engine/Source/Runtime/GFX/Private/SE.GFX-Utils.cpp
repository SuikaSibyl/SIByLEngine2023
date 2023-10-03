#include "../Public/SE.GFX-Utils.h"

namespace SIByL::GFX {
auto CaptureImage(GFX::Texture* tex, std::string const& path) noexcept -> void {
  RHI::RHILayer* rhiLayer = GFXManager::get()->rhiLayer;
  size_t width = tex->texture->width();
  size_t height = tex->texture->height();

  RHI::TextureFormat format;
  size_t pixelSize;
  if (tex->texture->format() == RHI::TextureFormat::RGBA32_FLOAT) {
    format = RHI::TextureFormat::RGBA32_FLOAT;
    pixelSize = sizeof(Math::vec4);
  } else if (tex->texture->format() == RHI::TextureFormat::RGBA8_UNORM) {
    format = RHI::TextureFormat::RGBA8_UNORM;
    pixelSize = sizeof(uint8_t) * 4;
  } else {
    Core::LogManager::Error(
        "Editor :: ViewportWidget :: captureImage() :: Unsupported format to "
        "capture.");
    return;
  }

  std::unique_ptr<RHI::CommandEncoder> commandEncoder =
      rhiLayer->getDevice()->createCommandEncoder({});

  Core::GUID copyDst = 0;
  if (copyDst == 0) {
    copyDst = Core::ResourceManager::get()->requestRuntimeGUID<GFX::Texture>();
    RHI::TextureDescriptor desc{
        {width, height, 1},
        1,
        1,
        1,
        RHI::TextureDimension::TEX2D,
        format,
        (uint32_t)RHI::TextureUsage::COPY_DST |
            (uint32_t)RHI::TextureUsage::TEXTURE_BINDING,
        {format},
        RHI::TextureFlags::HOSTI_VISIBLE};
    GFX::GFXManager::get()->registerTextureResource(copyDst, desc);

    GFX::Texture* copydst =
        Core::ResourceManager::get()->getResource<GFX::Texture>(copyDst);
    commandEncoder->pipelineBarrier(RHI::BarrierDescriptor{
        (uint32_t)RHI::PipelineStages::ALL_GRAPHICS_BIT,
        (uint32_t)RHI::PipelineStages::TRANSFER_BIT,
        (uint32_t)RHI::DependencyType::NONE,
        {},
        {},
        {RHI::TextureMemoryBarrierDescriptor{
            copydst->texture.get(),
            RHI::ImageSubresourceRange{(uint32_t)RHI::TextureAspect::COLOR_BIT,
                                       0, 1, 0, 1},
            (uint32_t)RHI::AccessFlagBits::NONE,
            (uint32_t)RHI::AccessFlagBits::TRANSFER_WRITE_BIT,
            RHI::TextureLayout::SHADER_READ_ONLY_OPTIMAL,
            RHI::TextureLayout::TRANSFER_DST_OPTIMAL}}});
  }
  rhiLayer->getDevice()->waitIdle();
  commandEncoder->pipelineBarrier(RHI::BarrierDescriptor{
      (uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT,
      (uint32_t)RHI::PipelineStages::TRANSFER_BIT,
      (uint32_t)RHI::DependencyType::NONE,
      {},
      {},
      {RHI::TextureMemoryBarrierDescriptor{
          tex->texture.get(),
          RHI::ImageSubresourceRange{(uint32_t)RHI::TextureAspect::COLOR_BIT, 0,
                                     1, 0, 1},
          (uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT |
              (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT,
          (uint32_t)RHI::AccessFlagBits::TRANSFER_READ_BIT,
          RHI::TextureLayout::SHADER_READ_ONLY_OPTIMAL,
          RHI::TextureLayout::TRANSFER_SRC_OPTIMAL}}});
  commandEncoder->copyTextureToTexture(
      RHI::ImageCopyTexture{tex->texture.get()},
      RHI::ImageCopyTexture{Core::ResourceManager::get()
                                ->getResource<GFX::Texture>(copyDst)
                                ->texture.get()},
      RHI::Extend3D{uint32_t(width), uint32_t(height), 1});
  commandEncoder->pipelineBarrier(RHI::BarrierDescriptor{
      (uint32_t)RHI::PipelineStages::TRANSFER_BIT,
      (uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT,
      (uint32_t)RHI::DependencyType::NONE,
      {},
      {},
      {RHI::TextureMemoryBarrierDescriptor{
          tex->texture.get(),
          RHI::ImageSubresourceRange{(uint32_t)RHI::TextureAspect::COLOR_BIT, 0,
                                     1, 0, 1},
          (uint32_t)RHI::AccessFlagBits::TRANSFER_READ_BIT,
          (uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT |
              (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT,
          RHI::TextureLayout::TRANSFER_SRC_OPTIMAL,
          RHI::TextureLayout::SHADER_READ_ONLY_OPTIMAL}}});
  commandEncoder->pipelineBarrier(RHI::BarrierDescriptor{
      (uint32_t)RHI::PipelineStages::TRANSFER_BIT,
      (uint32_t)RHI::PipelineStages::HOST_BIT,
      (uint32_t)RHI::DependencyType::NONE,
      {},
      {},
      {RHI::TextureMemoryBarrierDescriptor{
          Core::ResourceManager::get()
              ->getResource<GFX::Texture>(copyDst)
              ->texture.get(),
          RHI::ImageSubresourceRange{(uint32_t)RHI::TextureAspect::COLOR_BIT, 0,
                                     1, 0, 1},
          (uint32_t)RHI::AccessFlagBits::TRANSFER_WRITE_BIT,
          (uint32_t)RHI::AccessFlagBits::HOST_READ_BIT,
          RHI::TextureLayout::TRANSFER_DST_OPTIMAL,
          RHI::TextureLayout::TRANSFER_DST_OPTIMAL}}});
  rhiLayer->getDevice()->getGraphicsQueue()->submit(
      {commandEncoder->finish({})});
  rhiLayer->getDevice()->waitIdle();
  std::future<bool> mapped =
      Core::ResourceManager::get()
          ->getResource<GFX::Texture>(copyDst)
          ->texture->mapAsync((uint32_t)RHI::MapMode::READ, 0,
                              width * height * pixelSize);
  if (mapped.get()) {
    void* data = Core::ResourceManager::get()
                     ->getResource<GFX::Texture>(copyDst)
                     ->texture->getMappedRange(0, width * height * pixelSize);
    if (tex->texture->format() == RHI::TextureFormat::RGBA32_FLOAT) {
      std::string filepath = path + ".exr";
      Image::EXR::writeEXR(filepath, width, height, 4,
                           reinterpret_cast<float*>(data));
    } else if (tex->texture->format() == RHI::TextureFormat::RGBA8_UNORM) {
      std::string filepath = path + ".bmp";
      Image::BMP::writeBMP(filepath, width, height, 4,
                           reinterpret_cast<float*>(data));
    }
    Core::ResourceManager::get()
        ->getResource<GFX::Texture>(copyDst)
        ->texture->unmap();
  }
}
}