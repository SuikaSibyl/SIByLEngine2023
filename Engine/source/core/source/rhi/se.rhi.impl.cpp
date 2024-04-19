#define DLIB_EXPORT
#include <se.rhi.hpp>
#undef DLIB_EXPORT

namespace se::rhi {
auto getVertexFormatSize(VertexFormat format) noexcept -> size_t {
  switch (format) {
    case VertexFormat::UINT8X2:
      return sizeof(uint8_t) * 2;
    case VertexFormat::UINT8X4:
      return sizeof(uint8_t) * 4;
    case VertexFormat::SINT8X2:
      return sizeof(uint8_t) * 2;
    case VertexFormat::SINT8X4:
      return sizeof(uint8_t) * 4;
    case VertexFormat::UNORM8X2:
      return sizeof(uint8_t) * 2;
    case VertexFormat::UNORM8X4:
      return sizeof(uint8_t) * 4;
    case VertexFormat::SNORM8X2:
      return sizeof(uint8_t) * 2;
    case VertexFormat::SNORM8X4:
      return sizeof(uint8_t) * 4;
    case VertexFormat::UINT16X2:
      return sizeof(uint16_t) * 2;
    case VertexFormat::UINT16X4:
      return sizeof(uint16_t) * 4;
    case VertexFormat::SINT16X2:
      return sizeof(uint16_t) * 2;
    case VertexFormat::SINT16X4:
      return sizeof(uint16_t) * 4;
    case VertexFormat::UNORM16X2:
      return sizeof(uint16_t) * 2;
    case VertexFormat::UNORM16X4:
      return sizeof(uint16_t) * 4;
    case VertexFormat::SNORM16X2:
      return sizeof(uint16_t) * 2;
    case VertexFormat::SNORM16X4:
      return sizeof(uint16_t) * 4;
    case VertexFormat::FLOAT16X2:
      return sizeof(uint16_t) * 2;
    case VertexFormat::FLOAT16X4:
      return sizeof(uint16_t) * 4;
    case VertexFormat::FLOAT32:
      return sizeof(float) * 1;
    case VertexFormat::FLOAT32X2:
      return sizeof(float) * 2;
    case VertexFormat::FLOAT32X3:
      return sizeof(float) * 3;
    case VertexFormat::FLOAT32X4:
      return sizeof(float) * 4;
    case VertexFormat::UINT32:
      return sizeof(uint32_t) * 1;
    case VertexFormat::UINT32X2:
      return sizeof(uint32_t) * 2;
    case VertexFormat::UINT32X3:
      return sizeof(uint32_t) * 3;
    case VertexFormat::UINT32X4:
      return sizeof(uint32_t) * 4;
    case VertexFormat::SINT32:
      return sizeof(int32_t) * 1;
    case VertexFormat::SINT32X2:
      return sizeof(int32_t) * 2;
    case VertexFormat::SINT32X3:
      return sizeof(int32_t) * 3;
    case VertexFormat::SINT32X4:
      return sizeof(int32_t) * 4;
    default:
      return 0;
  }
}

auto hasDepthBit(TextureFormat format) noexcept -> bool {
  return format == TextureFormat::DEPTH16_UNORM ||
         format == TextureFormat::DEPTH24 ||
         format == TextureFormat::DEPTH24STENCIL8 ||
         format == TextureFormat::DEPTH32_FLOAT ||
         format == TextureFormat::DEPTH32STENCIL8; }

auto hasStencilBit(TextureFormat format) noexcept -> bool {
  return format == TextureFormat::STENCIL8 ||
         format == TextureFormat::DEPTH24STENCIL8 ||
         format == TextureFormat::DEPTH32STENCIL8; }

auto getTextureAspect(TextureFormat format) noexcept -> TextureAspects {
  switch (format) {
  case TextureFormat::STENCIL8:
    return (TextureAspects)TextureAspectBit::STENCIL_BIT;
  case TextureFormat::DEPTH16_UNORM:
  case TextureFormat::DEPTH24:
  case TextureFormat::DEPTH32_FLOAT:
    return (TextureAspects)TextureAspectBit::DEPTH_BIT;
  case TextureFormat::DEPTH24STENCIL8:
  case TextureFormat::DEPTH32STENCIL8:
    return (TextureAspects)TextureAspectBit::DEPTH_BIT |
           (TextureAspects)TextureAspectBit::STENCIL_BIT;
  default:
    return (TextureAspects)TextureAspectBit::COLOR_BIT;
  }
}

auto Device::readbackDeviceLocalTexture(Texture* texture, void* data,
                                        uint32_t size) noexcept -> void {
  TextureDescriptor desc = texture->getDescriptor();
  desc.usage = (uint32_t)TextureUsageBit::COPY_DST;
  desc.flags = desc.flags | (uint32_t)TextureFlagBit::HOSTI_VISIBLE;
  uint32_t pixel_size = 0;
  if (desc.format == TextureFormat::RGBA8_UNORM)
    pixel_size = sizeof(uint8_t) * 4;
  else if (desc.format == TextureFormat::RGBA32_FLOAT)
    pixel_size = sizeof(float) * 4;
  else {
    root::print::error("RHI :: Format not supported to be readback for now.");
    return;
  }
  std::unique_ptr<Texture> cpyDst = createTexture(desc);
  waitIdle();
  std::unique_ptr<CommandEncoder> commandEncoder = createCommandEncoder({});
  commandEncoder->pipelineBarrier(BarrierDescriptor{
      (uint32_t)PipelineStageBit::FRAGMENT_SHADER_BIT,
      (uint32_t)PipelineStageBit::TRANSFER_BIT,
      (uint32_t)DependencyType::NONE,
      {},
      {},
      {TextureMemoryBarrierDescriptor{
          texture,
          ImageSubresourceRange{(uint32_t)TextureAspectBit::COLOR_BIT, 0, 1, 0, 1},
          (uint32_t)AccessFlagBits::SHADER_READ_BIT |
          (uint32_t)AccessFlagBits::SHADER_WRITE_BIT,
          (uint32_t)AccessFlagBits::TRANSFER_READ_BIT,
          TextureLayout::SHADER_READ_ONLY_OPTIMAL,
          TextureLayout::TRANSFER_SRC_OPTIMAL}}});
  commandEncoder->copyTextureToTexture(
      ImageCopyTexture{texture}, ImageCopyTexture{cpyDst.get()},
      Extend3D{desc.size.width, desc.size.height, 1});
  commandEncoder->pipelineBarrier(BarrierDescriptor{
      (uint32_t)PipelineStageBit::TRANSFER_BIT,
      (uint32_t)PipelineStageBit::FRAGMENT_SHADER_BIT,
      (uint32_t)DependencyType::NONE,
      {},
      {},
      {TextureMemoryBarrierDescriptor{
          texture,
          ImageSubresourceRange{(uint32_t)TextureAspectBit::COLOR_BIT, 0, 1, 0, 1},
          (uint32_t)AccessFlagBits::TRANSFER_READ_BIT,
          (uint32_t)AccessFlagBits::SHADER_READ_BIT |
          (uint32_t)AccessFlagBits::SHADER_WRITE_BIT,
          TextureLayout::TRANSFER_SRC_OPTIMAL,
          TextureLayout::SHADER_READ_ONLY_OPTIMAL}}});
  commandEncoder->pipelineBarrier(BarrierDescriptor{
      (uint32_t)PipelineStageBit::TRANSFER_BIT,
      (uint32_t)PipelineStageBit::HOST_BIT,
      (uint32_t)DependencyType::NONE,
      {},
      {},
      {TextureMemoryBarrierDescriptor{
          cpyDst.get(),
          ImageSubresourceRange{(uint32_t)TextureAspectBit::COLOR_BIT, 0, 1, 0, 1},
          (uint32_t)AccessFlagBits::TRANSFER_WRITE_BIT,
          (uint32_t)AccessFlagBits::HOST_READ_BIT,
          TextureLayout::TRANSFER_DST_OPTIMAL,
          TextureLayout::TRANSFER_DST_OPTIMAL}}});
  getGraphicsQueue()->submit({commandEncoder->finish()});
  waitIdle();
  uint32_t mapped_size = desc.size.width * desc.size.height * pixel_size;
  std::future<bool> mapped =
      cpyDst->mapAsync((uint32_t)MapMode::READ, 0, mapped_size);
  if (mapped.get()) {
    void* mapped_data = cpyDst->getMappedRange(0, mapped_size);
    memcpy(data, mapped_data, mapped_size);
    cpyDst->unmap();
  }
}

auto Device::writebackDeviceLocalBuffer(Buffer* buffer, void* data,
                                        uint32_t size) noexcept -> void {
  // create staging buffer
  BufferDescriptor stagingBufferDescriptor;
  stagingBufferDescriptor.size = size;
  stagingBufferDescriptor.usage = (uint32_t)BufferUsageBit::COPY_SRC;
  stagingBufferDescriptor.memoryProperties =
      (uint32_t)MemoryPropertyBit::HOST_VISIBLE_BIT |
      (uint32_t)MemoryPropertyBit::HOST_COHERENT_BIT;
  stagingBufferDescriptor.mappedAtCreation = true;
  std::unique_ptr<Buffer> stagingBuffer = createBuffer(stagingBufferDescriptor);
  // staging buffer writeback
  std::future<bool> mapped = stagingBuffer->mapAsync(0, 0, buffer->size());
  if (mapped.get()) {
    void* mapdata = stagingBuffer->getMappedRange(0, buffer->size());
    memcpy(mapdata, data, (size_t)buffer->size());
    stagingBuffer->unmap();
  }
  // copy buffer
  std::unique_ptr<CommandEncoder> commandEncoder = createCommandEncoder({nullptr});
  commandEncoder->copyBufferToBuffer(stagingBuffer.get(), 0, buffer, 0, buffer->size());
  commandEncoder->pipelineBarrier(BarrierDescriptor{
      (uint32_t)PipelineStageBit::TRANSFER_BIT,
      (uint32_t)PipelineStageBit::ALL_COMMANDS_BIT,
      (uint32_t)DependencyType::NONE,
      {},
      {BufferMemoryBarrierDescriptor{
        buffer,
        (uint32_t)AccessFlagBits::TRANSFER_WRITE_BIT,
        (uint32_t)AccessFlagBits::SHADER_READ_BIT |
        (uint32_t)AccessFlagBits::SHADER_WRITE_BIT,
      }}});
  getGraphicsQueue()->submit({commandEncoder->finish()});
  getGraphicsQueue()->waitIdle();
}

auto Device::createDeviceLocalBuffer(void const* data, uint32_t size, BufferUsages usage) noexcept
    -> std::unique_ptr<Buffer> {
  std::unique_ptr<Buffer> buffer = nullptr;
  // create vertex buffer
  BufferDescriptor descriptor;
  descriptor.size = size;
  descriptor.usage = usage | (uint32_t)BufferUsageBit::COPY_DST |
                             (uint32_t)BufferUsageBit::COPY_SRC;
  descriptor.memoryProperties = (uint32_t)MemoryPropertyBit::DEVICE_LOCAL_BIT;
  descriptor.mappedAtCreation = true;
  buffer = createBuffer(descriptor);
  // create staging buffer
  BufferDescriptor stagingBufferDescriptor;
  stagingBufferDescriptor.size = size;
  stagingBufferDescriptor.usage = (uint32_t)BufferUsageBit::COPY_SRC;
  stagingBufferDescriptor.memoryProperties =
    (uint32_t)MemoryPropertyBit::HOST_VISIBLE_BIT |
    (uint32_t)MemoryPropertyBit::HOST_COHERENT_BIT;
  stagingBufferDescriptor.mappedAtCreation = true;
  std::unique_ptr<Buffer> stagingBuffer =
      createBuffer(stagingBufferDescriptor);
  std::future<bool> mapped = stagingBuffer->mapAsync(0, 0, descriptor.size);
  if (mapped.get()) {
    void* mapdata = stagingBuffer->getMappedRange(0, descriptor.size);
    memcpy(mapdata, data, (size_t)descriptor.size);
    stagingBuffer->unmap();
  }
  std::unique_ptr<CommandEncoder> commandEncoder =
      createCommandEncoder({nullptr});
  commandEncoder->pipelineBarrier(BarrierDescriptor{
      (uint32_t)PipelineStageBit::HOST_BIT,
      (uint32_t)PipelineStageBit::TRANSFER_BIT,
      0,
      // Optional (Memory Barriers)
      {},
      {BufferMemoryBarrierDescriptor{
          stagingBuffer.get(),
          (uint32_t)AccessFlagBits::HOST_WRITE_BIT,
          (uint32_t)AccessFlagBits::TRANSFER_READ_BIT,
      }},
      {}});
  commandEncoder->copyBufferToBuffer(stagingBuffer.get(), 0, buffer.get(), 0,
                                     descriptor.size);
  std::unique_ptr<Fence> fence = createFence();
  fence->reset();
  getGraphicsQueue()->submit({commandEncoder->finish()}, fence.get());
  getGraphicsQueue()->waitIdle();
  fence->wait();
  return buffer;
}

auto Device::readbackDeviceLocalBuffer(Buffer* buffer, void* data,
                                       uint32_t size) noexcept -> void {
  // create staging buffer
  BufferDescriptor stagingBufferDescriptor;
  stagingBufferDescriptor.size = size;
  stagingBufferDescriptor.usage = (uint32_t)BufferUsageBit::COPY_DST;
  stagingBufferDescriptor.memoryProperties =
      (uint32_t)MemoryPropertyBit::HOST_VISIBLE_BIT |
      (uint32_t)MemoryPropertyBit::HOST_COHERENT_BIT;
  stagingBufferDescriptor.mappedAtCreation = true;
  std::unique_ptr<Buffer> stagingBuffer = createBuffer(stagingBufferDescriptor);
  // copy buffer
  std::unique_ptr<CommandEncoder> commandEncoder = createCommandEncoder({nullptr});
  commandEncoder->pipelineBarrier(BarrierDescriptor{
      (uint32_t)PipelineStageBit::ALL_COMMANDS_BIT,
      (uint32_t)PipelineStageBit::TRANSFER_BIT,
      (uint32_t)DependencyType::NONE,
      {},
      {BufferMemoryBarrierDescriptor{
          buffer,
          (uint32_t)AccessFlagBits::SHADER_WRITE_BIT,
          (uint32_t)AccessFlagBits::TRANSFER_READ_BIT,
      }}});
  commandEncoder->copyBufferToBuffer(buffer, 0, stagingBuffer.get(), 0,
                                     buffer->size());
  getGraphicsQueue()->submit({commandEncoder->finish()});
  getGraphicsQueue()->waitIdle();
  // buffer readback
  std::future<bool> mapped = stagingBuffer->mapAsync(0, 0, buffer->size());
  if (mapped.get()) {
    void* mapdata = stagingBuffer->getMappedRange(0, buffer->size());
    memcpy(data, mapdata, (size_t)buffer->size());
    stagingBuffer->unmap();
  }
}

auto Device::trainsitionTextureLayout(Texture* texture, TextureLayout oldLayout,
    TextureLayout newLayout) noexcept -> void {
  std::unique_ptr<CommandEncoder> commandEncoder = createCommandEncoder({ nullptr });
  commandEncoder->pipelineBarrier(se::rhi::BarrierDescriptor{
  (uint32_t)se::rhi::PipelineStageBit::TOP_OF_PIPE_BIT,
  (uint32_t)se::rhi::PipelineStageBit::ALL_COMMANDS_BIT, 0,
  {}, {},
  std::vector<se::rhi::TextureMemoryBarrierDescriptor>{
    se::rhi::TextureMemoryBarrierDescriptor{
      texture,
      se::rhi::ImageSubresourceRange {
        (uint32_t)se::rhi::TextureAspectBit::COLOR_BIT, 0, 1, 0, 1},
        // memory barrier mask
        (uint32_t)se::rhi::AccessFlagBits::NONE,
        (uint32_t)se::rhi::AccessFlagBits::NONE,
        // only if layout transition is need
        oldLayout, newLayout
      }
  }
  });
  getGraphicsQueue()->submit({ commandEncoder->finish() });
  waitIdle();
}

}