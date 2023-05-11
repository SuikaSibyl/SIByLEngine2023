#include <SE.RHI-Interface.hpp>

namespace SIByL::RHI {
auto Device::readbackDeviceLocalTexture(Texture* texture, void* data,
                                        uint32_t size) noexcept -> void {
  RHI::TextureDescriptor desc = texture->getDescriptor();
  desc.usage = (uint32_t)RHI::TextureUsage::COPY_DST;
  desc.flags = desc.flags | RHI::TextureFlags::HOSTI_VISIBLE;
  uint32_t pixel_size = 0;
  if (desc.format == TextureFormat::RGBA8_UNORM)
    pixel_size = sizeof(uint8_t) * 4;
  else if (desc.format == TextureFormat::RGBA32_FLOAT)
    pixel_size = sizeof(float) * 4;
  else {
    Core::LogManager::Error(
        "RHI :: Format not supported to be readback for now.");
    return;
  }
  std::unique_ptr<RHI::Texture> cpyDst = createTexture(desc);
  waitIdle();
  std::unique_ptr<RHI::CommandEncoder> commandEncoder =
      createCommandEncoder({});
  commandEncoder->pipelineBarrier(RHI::BarrierDescriptor{
      (uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT,
      (uint32_t)RHI::PipelineStages::TRANSFER_BIT,
      (uint32_t)RHI::DependencyType::NONE,
      {},
      {},
      {RHI::TextureMemoryBarrierDescriptor{
          texture,
          RHI::ImageSubresourceRange{(uint32_t)RHI::TextureAspect::COLOR_BIT, 0,
                                     1, 0, 1},
          (uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT |
              (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT,
          (uint32_t)RHI::AccessFlagBits::TRANSFER_READ_BIT,
          RHI::TextureLayout::SHADER_READ_ONLY_OPTIMAL,
          RHI::TextureLayout::TRANSFER_SRC_OPTIMAL}}});
  commandEncoder->copyTextureToTexture(
      RHI::ImageCopyTexture{texture}, RHI::ImageCopyTexture{cpyDst.get()},
      RHI::Extend3D{desc.size.width, desc.size.height, 1});
  commandEncoder->pipelineBarrier(RHI::BarrierDescriptor{
      (uint32_t)RHI::PipelineStages::TRANSFER_BIT,
      (uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT,
      (uint32_t)RHI::DependencyType::NONE,
      {},
      {},
      {RHI::TextureMemoryBarrierDescriptor{
          texture,
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
          cpyDst.get(),
          RHI::ImageSubresourceRange{(uint32_t)RHI::TextureAspect::COLOR_BIT, 0,
                                     1, 0, 1},
          (uint32_t)RHI::AccessFlagBits::TRANSFER_WRITE_BIT,
          (uint32_t)RHI::AccessFlagBits::HOST_READ_BIT,
          RHI::TextureLayout::TRANSFER_DST_OPTIMAL,
          RHI::TextureLayout::TRANSFER_DST_OPTIMAL}}});
  getGraphicsQueue()->submit({commandEncoder->finish({})});
  waitIdle();
  uint32_t mapped_size = desc.size.width * desc.size.height * pixel_size;
  std::future<bool> mapped =
      cpyDst->mapAsync((uint32_t)RHI::MapMode::READ, 0, mapped_size);
  if (mapped.get()) {
    void* mapped_data = cpyDst->getMappedRange(0, mapped_size);
    memcpy(data, mapped_data, mapped_size);
    cpyDst->unmap();
  }
}

auto Device::writebackDeviceLocalBuffer(Buffer* buffer, void* data,
                                        uint32_t size) noexcept -> void {
  // create staging buffer
  RHI::BufferDescriptor stagingBufferDescriptor;
  stagingBufferDescriptor.size = size;
  stagingBufferDescriptor.usage = (uint32_t)RHI::BufferUsage::COPY_SRC;
  stagingBufferDescriptor.memoryProperties =
      (uint32_t)RHI::MemoryProperty::HOST_VISIBLE_BIT |
      (uint32_t)RHI::MemoryProperty::HOST_COHERENT_BIT;
  stagingBufferDescriptor.mappedAtCreation = true;
  std::unique_ptr<RHI::Buffer> stagingBuffer =
      createBuffer(stagingBufferDescriptor);
  // staging buffer writeback
  std::future<bool> mapped = stagingBuffer->mapAsync(0, 0, buffer->size());
  if (mapped.get()) {
    void* mapdata = stagingBuffer->getMappedRange(0, buffer->size());
    memcpy(mapdata, data, (size_t)buffer->size());
    stagingBuffer->unmap();
  }
  // copy buffer
  std::unique_ptr<RHI::CommandEncoder> commandEncoder =
      createCommandEncoder({nullptr});
  commandEncoder->copyBufferToBuffer(stagingBuffer.get(), 0, buffer, 0,
                                     buffer->size());
  commandEncoder->pipelineBarrier(RHI::BarrierDescriptor{
      (uint32_t)RHI::PipelineStages::TRANSFER_BIT,
      (uint32_t)RHI::PipelineStages::ALL_COMMANDS_BIT,
      (uint32_t)RHI::DependencyType::NONE,
      {},
      {RHI::BufferMemoryBarrierDescriptor{
          buffer,
          (uint32_t)RHI::AccessFlagBits::TRANSFER_WRITE_BIT,
          (uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT |
              (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT,
      }}});
  getGraphicsQueue()->submit({commandEncoder->finish({})});
  getGraphicsQueue()->waitIdle();
}

auto Device::createDeviceLocalBuffer(void* data, uint32_t size,
                                     BufferUsagesFlags usage) noexcept
    -> std::unique_ptr<Buffer> {
  std::unique_ptr<Buffer> buffer = nullptr;
  // create vertex buffer
  RHI::BufferDescriptor descriptor;
  descriptor.size = size;
  descriptor.usage = usage | (uint32_t)RHI::BufferUsage::COPY_DST |
                     (uint32_t)RHI::BufferUsage::COPY_SRC;
  descriptor.memoryProperties = (uint32_t)RHI::MemoryProperty::DEVICE_LOCAL_BIT;
  descriptor.mappedAtCreation = true;
  buffer = createBuffer(descriptor);
  // create staging buffer
  RHI::BufferDescriptor stagingBufferDescriptor;
  stagingBufferDescriptor.size = size;
  stagingBufferDescriptor.usage = (uint32_t)RHI::BufferUsage::COPY_SRC;
  stagingBufferDescriptor.memoryProperties =
      (uint32_t)RHI::MemoryProperty::HOST_VISIBLE_BIT |
      (uint32_t)RHI::MemoryProperty::HOST_COHERENT_BIT;
  stagingBufferDescriptor.mappedAtCreation = true;
  std::unique_ptr<RHI::Buffer> stagingBuffer =
      createBuffer(stagingBufferDescriptor);
  std::future<bool> mapped = stagingBuffer->mapAsync(0, 0, descriptor.size);
  if (mapped.get()) {
    void* mapdata = stagingBuffer->getMappedRange(0, descriptor.size);
    memcpy(mapdata, data, (size_t)descriptor.size);
    stagingBuffer->unmap();
  }
  std::unique_ptr<RHI::CommandEncoder> commandEncoder =
      createCommandEncoder({nullptr});
  commandEncoder->copyBufferToBuffer(stagingBuffer.get(), 0, buffer.get(), 0,
                                     descriptor.size);
  std::unique_ptr<Fence> fence = createFence();
  fence->reset();
  getGraphicsQueue()->submit({commandEncoder->finish({})}, fence.get());
  getGraphicsQueue()->waitIdle();
  fence->wait();
  return buffer;
}

auto Device::readbackDeviceLocalBuffer(Buffer* buffer, void* data,
                                       uint32_t size) noexcept -> void {
  // create staging buffer
  RHI::BufferDescriptor stagingBufferDescriptor;
  stagingBufferDescriptor.size = size;
  stagingBufferDescriptor.usage = (uint32_t)RHI::BufferUsage::COPY_DST;
  stagingBufferDescriptor.memoryProperties =
      (uint32_t)RHI::MemoryProperty::HOST_VISIBLE_BIT |
      (uint32_t)RHI::MemoryProperty::HOST_COHERENT_BIT;
  stagingBufferDescriptor.mappedAtCreation = true;
  std::unique_ptr<RHI::Buffer> stagingBuffer =
      createBuffer(stagingBufferDescriptor);
  // copy buffer
  std::unique_ptr<RHI::CommandEncoder> commandEncoder =
      createCommandEncoder({nullptr});
  commandEncoder->pipelineBarrier(RHI::BarrierDescriptor{
      (uint32_t)RHI::PipelineStages::ALL_COMMANDS_BIT,
      (uint32_t)RHI::PipelineStages::TRANSFER_BIT,
      (uint32_t)RHI::DependencyType::NONE,
      {},
      {RHI::BufferMemoryBarrierDescriptor{
          buffer,
          (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT,
          (uint32_t)RHI::AccessFlagBits::TRANSFER_READ_BIT,
      }}});
  commandEncoder->copyBufferToBuffer(buffer, 0, stagingBuffer.get(), 0,
                                     buffer->size());
  getGraphicsQueue()->submit({commandEncoder->finish({})});
  getGraphicsQueue()->waitIdle();
  // buffer readback
  std::future<bool> mapped = stagingBuffer->mapAsync(0, 0, buffer->size());
  if (mapped.get()) {
    void* mapdata = stagingBuffer->getMappedRange(0, buffer->size());
    memcpy(data, mapdata, (size_t)buffer->size());
    stagingBuffer->unmap();
  }
}
}  // namespace SIByL::RHI