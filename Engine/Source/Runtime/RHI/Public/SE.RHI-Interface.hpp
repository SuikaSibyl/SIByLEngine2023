﻿#pragma once
#include <future>
#include <memory>
#include <optional>
#include <variant>
#include <vector>
#include <Print/SE.Core.Log.hpp>
#include <Memory/SE.Core.Memory.hpp>
#include <SE.Math.Geometric.hpp>
import SE.Platform.Window;

namespace SIByL::RHI {
// *************************|****************************************
// Initialization			|   Initialization
// |
struct Context;
struct Adapter;
struct RequestAdapterOptions;
struct Device;
struct DeviceDescriptor;
// *************************|****************************************
// Buffers					|	Buffers
// |
SE_EXPORT using BufferUsagesFlags = uint32_t;
struct Buffer;
struct BufferDescriptor;
// *************************|****************************************
// Textures & Views			|   Textures & Views
// |
struct Texture;
struct TextureDescriptor;
struct TextureView;
struct TextureViewDescriptor;
struct ExternalTexture;
struct ExternalTextureDescriptor;
// *************************|****************************************
// Samplers					|   Samplers
// |
struct Sampler;
struct SamplerDescriptor;
// *************************|****************************************
// Swapchain				|   Swapchain
// |
struct SwapChain;
struct SwapChainDescriptor;
// *************************|****************************************
// Pipeline					|   Pipeline
// |
struct ShaderModule;
struct ShaderModuleDescriptor;
struct ComputePipeline;
struct ComputePipelineDescriptor;
struct RenderPipeline;
struct RenderPipelineDescriptor;
struct PipelineLayout;
struct PipelineLayoutDescriptor;
// *************************|****************************************
// Resource Binding			|   Resource Binding
// |
struct BindGroup;
struct BindGroupDescriptor;
struct BindGroupLayout;
struct BindGroupLayoutDescriptor;
// *************************|****************************************
// Command Encoder			|   Command Encoder
// |
struct CommandBuffer;
struct MultiFrameFlights;
struct MultiFrameFlightsDescriptor;
struct CommandEncoder;
struct CommandEncoderDescriptor;
struct RenderBundleEncoder;
struct RenderBundleEncoderDescriptor;
// *************************|****************************************
// Queue					|   Queue
// |
struct Queue;
struct QueueDescriptor;
// *************************|****************************************
// Queries					|   Queries
// |
struct QuerySet;
struct QuerySetDescriptor;
// *************************|****************************************
// Synchronize				|   Synchronize
// |
struct Fence;
struct Barrier;
struct BarrierDescriptor;
struct MemoryBarrier;
struct MemoryBarrierDescriptor;
struct Semaphore;
/** pipeline stage enums */
SE_EXPORT using PipelineStageFlags = uint32_t;
/** pipeline stage enums */
SE_EXPORT enum class PipelineStages : uint32_t {
  TOP_OF_PIPE_BIT = 0x00000001,
  DRAW_INDIRECT_BIT = 0x00000002,
  VERTEX_INPUT_BIT = 0x00000004,
  VERTEX_SHADER_BIT = 0x00000008,
  TESSELLATION_CONTROL_SHADER_BIT = 0x00000010,
  TESSELLATION_EVALUATION_SHADER_BIT = 0x00000020,
  GEOMETRY_SHADER_BIT = 0x00000040,
  FRAGMENT_SHADER_BIT = 0x00000080,
  EARLY_FRAGMENT_TESTS_BIT = 0x00000100,
  LATE_FRAGMENT_TESTS_BIT = 0x00000200,
  COLOR_ATTACHMENT_OUTPUT_BIT = 0x00000400,
  COMPUTE_SHADER_BIT = 0x00000800,
  TRANSFER_BIT = 0x00001000,
  BOTTOM_OF_PIPE_BIT = 0x00002000,
  HOST_BIT = 0x00004000,
  ALL_GRAPHICS_BIT = 0x00008000,
  ALL_COMMANDS_BIT = 0x00010000,
  TRANSFORM_FEEDBACK_BIT_EXT = 0x01000000,
  CONDITIONAL_RENDERING_BIT_EXT = 0x00040000,
  ACCELERATION_STRUCTURE_BUILD_BIT_KHR = 0x02000000,
  RAY_TRACING_SHADER_BIT_KHR = 0x00200000,
  TASK_SHADER_BIT_NV = 0x00080000,
  MESH_SHADER_BIT_NV = 0x00100000,
  FRAGMENT_DENSITY_PROCESS_BIT = 0x00800000,
  FRAGMENT_SHADING_RATE_ATTACHMENT_BIT = 0x00400000,
  COMMAND_PREPROCESS_BIT = 0x00020000,
};
// *************************|****************************************
// Ray Tracing				|   Ray Tracing
// |
struct BLAS;
struct BLASDescriptor;
struct TLAS;
struct TLASDescriptor;
struct RayTracingPipeline;
struct RayTracingPipelineDescriptor;
struct RayTracingPassEncoder;
struct RayTracingPassDescriptor;
// *************************|****************************************
// Extensions				|   Extensions
// |
struct RayTracingExtension;
// *************************|****************************************

//
// ===========================================================================
// Initialization Interface

struct AdapterInfo;
struct RequestAdapterOptions;

////////////////////////////////////
//
// Context
//

/** Context Extensions for extending API capability */
SE_EXPORT using ContextExtensionsFlags = uint32_t;
/** Context Extensions for extending API capability */
SE_EXPORT enum struct ContextExtension {
  NONE = 0 << 0,
  DEBUG_UTILS = 1 << 0,
  MESH_SHADER = 1 << 1,
  FRAGMENT_BARYCENTRIC = 1 << 2,
  SAMPLER_FILTER_MIN_MAX = 1 << 3,
  RAY_TRACING = 1 << 4,
  SHADER_NON_SEMANTIC_INFO = 1 << 5,
  BINDLESS_INDEXING = 1 << 6,
  ATOMIC_FLOAT = 1 << 7,
  CONSERVATIVE_RASTERIZATION = 1 << 8,
  COOPERATIVE_MATRIX = 1 << 9,
};

SE_EXPORT enum struct PowerPreference {
  LOW_POWER,
  HIGH_PERFORMANCE,
};

SE_EXPORT struct RequestAdapterOptions {
  PowerPreference powerPerference = PowerPreference::HIGH_PERFORMANCE;
  bool forceFallbackAdapter = false;
};

/** Context Interface for multiple-Graphics-API */
SE_EXPORT struct Context {
  /** virtual destructor */
  virtual ~Context() = default;
  /** Initialize the context */
  virtual auto init(Platform::Window* window = nullptr,
                    ContextExtensionsFlags ext = 0) noexcept -> bool = 0;
  /** Request an adapter */
  virtual auto requestAdapter(
      RequestAdapterOptions const& options = RequestAdapterOptions{}) noexcept
      -> std::unique_ptr<Adapter> = 0;
  /** Get the binded window */
  virtual auto getBindedWindow() const noexcept -> Platform::Window* = 0;
  /** clean up context resources */
  virtual auto destroy() noexcept -> void = 0;
};

////////////////////////////////////
//
// Adapter
//

/** Describes the physical properties of a given GPU. */
SE_EXPORT struct Adapter {
  /** virtual destructor */
  virtual ~Adapter() = default;
  /** Requests a device from the adapter. */
  virtual auto requestDevice() noexcept -> std::unique_ptr<Device> = 0;
  /** Requests the AdapterInfo for this Adapter. */
  virtual auto requestAdapterInfo() const noexcept -> AdapterInfo = 0;
};

/** information of adapter */
SE_EXPORT struct AdapterInfo {
  std::string vendor;
  std::string architecture;
  std::string device;
  std::string description;
  float timestampPeriod;
};

////////////////////////////////////
//
// Device
//

/**
 * Device is a logical instantiation of an adapter, through which internal
 * objects are created. Is the exclusive owner of all internal objects created
 * from it.
 */
SE_EXPORT struct Device {
  /** virtual destructor */
  virtual ~Device() = default;
  /** destroy the device */
  virtual auto destroy() noexcept -> void = 0;
  /** wait until idle */
  virtual auto waitIdle() noexcept -> void = 0;
  // Read-only fields
  // ---------------------------
  /** the graphics queue for this device */
  virtual auto getGraphicsQueue() noexcept -> Queue* = 0;
  /** the compute queue for this device */
  virtual auto getComputeQueue() noexcept -> Queue* = 0;
  /** the present queue for this device */
  virtual auto getPresentQueue() noexcept -> Queue* = 0;
  // Create resources on device
  // ---------------------------
  /** create a buffer on the device */
  virtual auto createBuffer(BufferDescriptor const& desc) noexcept
      -> std::unique_ptr<Buffer> = 0;
  /** create a texture on the device */
  virtual auto createTexture(TextureDescriptor const& desc) noexcept
      -> std::unique_ptr<Texture> = 0;
  /** create a sampler on the device */
  virtual auto createSampler(SamplerDescriptor const& desc) noexcept
      -> std::unique_ptr<Sampler> = 0;
  /** create a external texture on the device */
  virtual auto importExternalTexture(
      ExternalTextureDescriptor const& desc) noexcept
      -> std::unique_ptr<ExternalTexture> = 0;
  /* create a swapchain on the device */
  virtual auto createSwapChain(SwapChainDescriptor const& desc) noexcept
      -> std::unique_ptr<SwapChain> = 0;
  // Create resources binding objects
  // ---------------------------
  /** create a bind group layout on the device */
  virtual auto createBindGroupLayout(
      BindGroupLayoutDescriptor const& desc) noexcept
      -> std::unique_ptr<BindGroupLayout> = 0;
  /** create a pipeline layout on the device */
  virtual auto createPipelineLayout(
      PipelineLayoutDescriptor const& desc) noexcept
      -> std::unique_ptr<PipelineLayout> = 0;
  /** create a bind group on the device */
  virtual auto createBindGroup(BindGroupDescriptor const& desc) noexcept
      -> std::unique_ptr<BindGroup> = 0;
  // Create pipeline objects
  // ---------------------------
  /** create a shader module on the device */
  virtual auto createShaderModule(ShaderModuleDescriptor const& desc) noexcept
      -> std::unique_ptr<ShaderModule> = 0;
  /** create a compute pipeline on the device */
  virtual auto createComputePipeline(
      ComputePipelineDescriptor const& desc) noexcept
      -> std::unique_ptr<ComputePipeline> = 0;
  /** create a render pipeline on the device */
  virtual auto createRenderPipeline(
      RenderPipelineDescriptor const& desc) noexcept
      -> std::unique_ptr<RenderPipeline> = 0;
  /** create a compute pipeline on the device in async way */
  virtual auto createComputePipelineAsync(
      ComputePipelineDescriptor const& desc) noexcept
      -> std::future<std::unique_ptr<ComputePipeline>> = 0;
  /** create a render pipeline on the device in async way */
  virtual auto createRenderPipelineAsync(
      RenderPipelineDescriptor const& desc) noexcept
      -> std::future<std::unique_ptr<RenderPipeline>> = 0;
  // Create command encoders
  // ---------------------------
  /** create a multi frame flights */
  virtual auto createMultiFrameFlights(
      MultiFrameFlightsDescriptor const& desc) noexcept
      -> std::unique_ptr<MultiFrameFlights> = 0;
  /** create a command encoder */
  virtual auto createCommandEncoder(
      CommandEncoderDescriptor const& desc) noexcept
      -> std::unique_ptr<CommandEncoder> = 0;
  /** create a render bundle encoder */
  virtual auto createRenderBundleEncoder(
      CommandEncoderDescriptor const& desc) noexcept
      -> std::unique_ptr<RenderBundleEncoder> = 0;
  // Create query sets
  // ---------------------------
  virtual auto createQuerySet(QuerySetDescriptor const& desc) noexcept
      -> std::unique_ptr<QuerySet> = 0;
  // Create ray tracing objects
  // ---------------------------
  /** create a BLAS */
  virtual auto createBLAS(BLASDescriptor const& desc) noexcept
      -> std::unique_ptr<BLAS> = 0;
  /** create a TLAS */
  virtual auto createTLAS(TLASDescriptor const& desc) noexcept
      -> std::unique_ptr<TLAS> = 0;
  /** create a ray tracing pipeline on the device */
  virtual auto createRayTracingPipeline(
      RayTracingPipelineDescriptor const& desc) noexcept
      -> std::unique_ptr<RayTracingPipeline> = 0;
  // Create memory barrier objects
  // ---------------------------
  virtual auto createFence() noexcept -> std::unique_ptr<Fence> = 0;
  // Get extensions
  // ---------------------------
  /** fetch a ray tracing extension is available */
  virtual auto getRayTracingExtension() noexcept -> RayTracingExtension* = 0;
  // Create utilities
  // ---------------------------
  /** create a device local buffer with initialzie value */
  auto createDeviceLocalBuffer(void* data, uint32_t size,
                               BufferUsagesFlags usage) noexcept
      -> std::unique_ptr<Buffer>;
  /** read back device local buffer */
  auto readbackDeviceLocalBuffer(Buffer* buffer, void* data,
                                 uint32_t size) noexcept -> void;
  /** write back device local buffer */
  auto writebackDeviceLocalBuffer(Buffer* buffer, void* data,
                                  uint32_t size) noexcept -> void;
  /** read back device local texture */
  auto readbackDeviceLocalTexture(Texture* texture, void* data,
                                  uint32_t size) noexcept -> void;
};

struct DeviceDescriptor {};

SE_EXPORT struct MultiFrameFlights {
  /** virtual destructor */
  virtual ~MultiFrameFlights() = default;
  /** start frame */
  virtual auto frameStart() noexcept -> void = 0;
  /** end frame */
  virtual auto frameEnd() noexcept -> void = 0;
  /** get current flight id */
  virtual auto getFlightIndex() noexcept -> uint32_t = 0;
  /** get current swapchain id */
  virtual auto getSwapchainIndex() noexcept -> uint32_t = 0;
  /** get current command buffer */
  virtual auto getCommandBuffer() noexcept -> CommandBuffer* = 0;
  /** get current Image Available Semaphore */
  virtual auto getImageAvailableSeamaphore() noexcept -> Semaphore* = 0;
  /** get current Render Finished Semaphore */
  virtual auto getRenderFinishedSeamaphore() noexcept -> Semaphore* = 0;
  /** get current fence */
  virtual auto getFence() noexcept -> Fence* = 0;
};

SE_EXPORT struct MultiFrameFlightsDescriptor {
  int maxFlightNum = 1;
  SwapChain* swapchain = nullptr;
};

// Initialization Interface
// ===========================================================================
// Buffers Interface

/** An object that holds a pointer (which can be null) to a buffer of a fixed
 * number of bytes */
SE_EXPORT using ArrayBuffer = void*;

/** Determine how a GPUBuffer may be used after its creation. */
SE_EXPORT enum struct BufferUsage {
  MAP_READ = 1 << 0,
  MAP_WRITE = 1 << 1,
  COPY_SRC = 1 << 2,
  COPY_DST = 1 << 3,
  INDEX = 1 << 4,
  VERTEX = 1 << 5,
  UNIFORM = 1 << 6,
  STORAGE = 1 << 7,
  INDIRECT = 1 << 8,
  QUERY_RESOLVE = 1 << 9,
  SHADER_DEVICE_ADDRESS = 1 << 10,
  ACCELERATION_STRUCTURE_STORAGE = 1 << 11,
  ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY = 1 << 12,
  SHADER_BINDING_TABLE = 1 << 13,
};

/** Determine the memory properties. */
SE_EXPORT using MemoryPropertiesFlags = uint32_t;
/** Determine the memory properties. */
SE_EXPORT enum class MemoryProperty {
  DEVICE_LOCAL_BIT = 1 << 0,
  HOST_VISIBLE_BIT = 1 << 1,
  HOST_COHERENT_BIT = 1 << 2,
  HOST_CACHED_BIT = 1 << 3,
  LAZILY_ALLOCATED_BIT = 1 << 4,
  PROTECTED_BIT = 1 << 5,
  FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
};

/** Current map state of the buffer */
SE_EXPORT enum struct BufferMapState { UNMAPPED, PENDING, MAPPED };

/** Shader mode of the buffer */
SE_EXPORT enum struct BufferShareMode {
  CONCURRENT,
  EXCLUSIVE,
};

/** Determine how a Buffer is mapped when calling mapAsync(). */
SE_EXPORT using MapModeFlags = uint32_t;
/** Determine how a Buffer is mapped when calling mapAsync(). */
SE_EXPORT enum struct MapMode {
  READ = 1,
  WRITE = 2,
};

/**
 * A block of memory that can be used in GPU operations.
 * Data is stored in linear layout, and some can be mapped.
 */
SE_EXPORT struct Buffer {
  /** virtual destructor */
  virtual ~Buffer() = default;
  // Readonly Attributes
  // ---------------------------
  /** readonly get buffer size on GPU */
  virtual auto size() const noexcept -> size_t = 0;
  /** readonly get buffer usage flags on GPU */
  virtual auto bufferUsageFlags() const noexcept -> BufferUsagesFlags = 0;
  /** readonly get map state on GPU */
  virtual auto bufferMapState() const noexcept -> BufferMapState = 0;
  /** readonly get device */
  virtual auto getDevice() const noexcept -> Device* = 0;
  // Map methods
  // ---------------------------
  /** Maps the given range of the GPUBuffer */
  virtual auto mapAsync(MapModeFlags mode, size_t offset = 0,
                        size_t size = 0) noexcept -> std::future<bool> = 0;
  /** Returns an ArrayBuffer with the contents of the GPUBuffer in the given
   * mapped range */
  virtual auto getMappedRange(size_t offset = 0, size_t size = 0) noexcept
      -> ArrayBuffer = 0;
  /** Unmaps the mapped range of the GPUBuffer and makes it’s contents available
   * for use by the GPU again. */
  virtual auto unmap() noexcept -> void = 0;
  // Lifecycle methods
  // ---------------------------
  /** destroy the buffer */
  virtual auto destroy() noexcept -> void = 0;
  /** set debug name */
  virtual auto setName(std::string const& name) -> void = 0;
  /** get debug name */
  virtual auto getName() const noexcept -> std::string const& = 0;
};

SE_EXPORT struct BufferDescriptor {
  /** The size of the buffer in bytes. */
  size_t size;
  /** The allowed usages for the buffer. */
  BufferUsagesFlags usage;
  /** The queue access share mode of the buffer. */
  BufferShareMode shareMode = BufferShareMode::EXCLUSIVE;
  /** The memory properties of the buffer. */
  MemoryPropertiesFlags memoryProperties = 0;
  /** If true creates the buffer in an already mapped state. */
  bool mappedAtCreation = false;
  /** buffer alignment (-1 means dont care) */
  int minimumAlignment = -1;
};

// Buffers Interface
// ===========================================================================
// Textures/TextureViews Interface

SE_EXPORT enum struct TextureDimension { TEX1D, TEX2D, TEX3D };

SE_EXPORT enum struct TextureFormat {
  // Unknown
  UNKOWN,
  // 8-bit formats
  R8_UNORM,
  R8_SNORM,
  R8_UINT,
  R8_SINT,
  // 16-bit formats
  R16_UINT,
  R16_SINT,
  R16_FLOAT,
  RG8_UNORM,
  RG8_SNORM,
  RG8_UINT,
  RG8_SINT,
  // 32-bit formats
  R32_UINT,
  R32_SINT,
  R32_FLOAT,
  RG16_UINT,
  RG16_SINT,
  RG16_FLOAT,
  RGBA8_UNORM,
  RGBA8_UNORM_SRGB,
  RGBA8_SNORM,
  RGBA8_UINT,
  RGBA8_SINT,
  BGRA8_UNORM,
  BGRA8_UNORM_SRGB,
  // Packed 32-bit formats
  RGB9E5_UFLOAT,
  RG11B10_UFLOAT,
  // 64-bit formats
  RG32_UINT,
  RG32_SINT,
  RG32_FLOAT,
  RGBA16_UINT,
  RGBA16_SINT,
  RGBA16_FLOAT,
  // 128-bit formats
  RGBA32_UINT,
  RGBA32_SINT,
  RGBA32_FLOAT,
  // Depth/stencil formats
  STENCIL8,
  DEPTH16_UNORM,
  DEPTH24,
  DEPTH24STENCIL8,
  DEPTH32_FLOAT,
  //
  COMPRESSION,
  //
  RGB10A2_UNORM,
  // "depth32float-stencil8" feature
  DEPTH32STENCIL8,
  // Compressed
  BC1_RGB_UNORM_BLOCK,
  BC1_RGB_SRGB_BLOCK,
  BC1_RGBA_UNORM_BLOCK,
  BC1_RGBA_SRGB_BLOCK,
  BC2_UNORM_BLOCK,
  BC2_SRGB_BLOCK,
  BC3_UNORM_BLOCK,
  BC3_SRGB_BLOCK,
  BC4_UNORM_BLOCK,
  BC4_SNORM_BLOCK,
  BC5_UNORM_BLOCK,
  BC5_SNORM_BLOCK,
  BC6H_UFLOAT_BLOCK,
  BC6H_SFLOAT_BLOCK,
  BC7_UNORM_BLOCK,
  BC7_SRGB_BLOCK,
};

SE_EXPORT inline auto hasDepthBit(TextureFormat format) noexcept -> bool {
  return format == TextureFormat::DEPTH16_UNORM ||
         format == TextureFormat::DEPTH24 ||
         format == TextureFormat::DEPTH24STENCIL8 ||
         format == TextureFormat::DEPTH32_FLOAT ||
         format == TextureFormat::DEPTH32STENCIL8;
}

SE_EXPORT inline auto hasStencilBit(TextureFormat format) noexcept -> bool {
  return format == TextureFormat::STENCIL8 ||
         format == TextureFormat::DEPTH24STENCIL8 ||
         format == TextureFormat::DEPTH32STENCIL8;
}

/** Determine how a Texture may be used after its creation. */
SE_EXPORT using TextureUsagesFlags = uint32_t;
/** Determine how a Texture may be used after its creation. */
SE_EXPORT enum struct TextureUsage {
  COPY_SRC = 1 << 0,
  COPY_DST = 1 << 1,
  TEXTURE_BINDING = 1 << 2,
  STORAGE_BINDING = 1 << 3,
  COLOR_ATTACHMENT = 1 << 4,
  DEPTH_ATTACHMENT = 1 << 5,
  TRANSIENT_ATTACHMENT = 1 << 6,
  INPUT_ATTACHMENT = 1 << 7,
};

SE_EXPORT struct Extend3D {
  uint32_t width;
  uint32_t height;
  uint32_t depthOrArrayLayers;
};

SE_EXPORT enum struct TextureLayout : uint32_t {
  UNDEFINED,
  GENERAL,
  COLOR_ATTACHMENT_OPTIMAL,
  DEPTH_STENCIL_ATTACHMENT_OPTIMA,
  DEPTH_STENCIL_READ_ONLY_OPTIMAL,
  SHADER_READ_ONLY_OPTIMAL,
  TRANSFER_SRC_OPTIMAL,
  TRANSFER_DST_OPTIMAL,
  PREINITIALIZED,
  DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL,
  DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL,
  DEPTH_ATTACHMENT_OPTIMAL,
  DEPTH_READ_ONLY_OPTIMAL,
  STENCIL_ATTACHMENT_OPTIMAL,
  STENCIL_READ_ONLY_OPTIMAL,
  PRESENT_SRC,
  SHARED_PRESENT,
  FRAGMENT_DENSITY_MAP_OPTIMAL,
  FRAGMENT_SHADING_RATE_ATTACHMENT_OPTIMAL,
  READ_ONLY_OPTIMAL,
  ATTACHMENT_OPTIMAL,
};

SE_EXPORT struct Texture {
  // Texture Behaviors
  // ---------------------------
  /** virtual descructor */
  virtual ~Texture() = default;
  /** create texture view of this texture */
  virtual auto createView(TextureViewDescriptor const& desc) noexcept
      -> std::unique_ptr<TextureView> = 0;
  /** destroy this texture */
  virtual auto destroy() noexcept -> void = 0;
  // Readonly Attributes
  // ---------------------------
  /** readonly width of the texture */
  virtual auto width() const noexcept -> uint32_t = 0;
  /** readonly height of the texture */
  virtual auto height() const noexcept -> uint32_t = 0;
  /** readonly depth or arrayLayers of the texture */
  virtual auto depthOrArrayLayers() const noexcept -> uint32_t = 0;
  /** readonly mip level count of the texture */
  virtual auto mipLevelCount() const noexcept -> uint32_t = 0;
  /** readonly sample count of the texture */
  virtual auto sampleCount() const noexcept -> uint32_t = 0;
  /** the dimension of the set of texel for each of this GPUTexture's
   * subresources. */
  virtual auto dimension() const noexcept -> TextureDimension = 0;
  /** readonly format of the texture */
  virtual auto format() const noexcept -> TextureFormat = 0;
  /** set debug name */
  virtual auto setName(std::string const& name) -> void = 0;
  /** get name */
  virtual auto getName() -> std::string const& = 0;
  /** get texture descriptor */
  virtual auto getDescriptor() -> TextureDescriptor = 0;
  // Map methods
  // ---------------------------
  /** Maps the given range of the GPUBuffer */
  virtual auto mapAsync(MapModeFlags mode, size_t offset = 0,
                        size_t size = 0) noexcept -> std::future<bool> = 0;
  /** Returns an ArrayBuffer with the contents of the GPUBuffer in the given
   * mapped range */
  virtual auto getMappedRange(size_t offset = 0, size_t size = 0) noexcept
      -> ArrayBuffer = 0;
  /** Unmaps the mapped range of the GPUBuffer and makes it’s contents available
   * for use by the GPU again. */
  virtual auto unmap() noexcept -> void = 0;
};

SE_EXPORT enum struct TextureFlags {
  NONE = 0,
  HOSTI_VISIBLE = 1 << 0,
  CUBE_COMPATIBLE = 1 << 1,
};

SE_EXPORT struct TextureDescriptor {
  Extend3D size;
  uint32_t mipLevelCount = 1;
  uint32_t arrayLayerCount = 1;
  uint32_t sampleCount = 1;
  TextureDimension dimension = TextureDimension::TEX2D;
  TextureFormat format;
  /** The allowed usages for the texture. */
  TextureUsagesFlags usage;
  /**
   * Specifies what view format values will be allowed when calling createView()
   * on this texture (in addition to the texture’s actual format).
   */
  std::vector<TextureFormat> viewFormats;
  TextureFlags flags = TextureFlags::NONE;
};

SE_EXPORT struct Color {
  double r, g, b, a;
};

SE_EXPORT enum struct TextureViewDimension {
  TEX1D,
  TEX1D_ARRAY,
  TEX2D,
  TEX2D_ARRAY,
  CUBE,
  CUBE_ARRAY,
  TEX3D,
  TEX3D_ARRAY,
};

/** Determine how a texture is used in command. */
SE_EXPORT using TextureAspectFlags = uint32_t;
/** Determine how a texture is used in command. */
SE_EXPORT enum struct TextureAspect {
  COLOR_BIT = 1 << 0,
  STENCIL_BIT = 1 << 1,
  DEPTH_BIT = 1 << 2,
};

SE_EXPORT struct ImageSubresourceRange {
  TextureAspectFlags aspectMask;
  uint32_t baseMipLevel;
  uint32_t levelCount;
  uint32_t baseArrayLayer;
  uint32_t layerCount;
};

SE_EXPORT struct TextureClearDescriptor {
  std::vector<ImageSubresourceRange> subresources;
  Color clearColor;
};

/* get aspect from texture format */
SE_EXPORT inline auto getTextureAspect(TextureFormat format) noexcept
    -> TextureAspectFlags {
  switch (format) {
    case SIByL::RHI::TextureFormat::STENCIL8:
      return (TextureAspectFlags)TextureAspect::STENCIL_BIT;
    case SIByL::RHI::TextureFormat::DEPTH16_UNORM:
    case SIByL::RHI::TextureFormat::DEPTH24:
    case SIByL::RHI::TextureFormat::DEPTH32_FLOAT:
      return (TextureAspectFlags)TextureAspect::DEPTH_BIT;
    case SIByL::RHI::TextureFormat::DEPTH24STENCIL8:
    case SIByL::RHI::TextureFormat::DEPTH32STENCIL8:
      return (TextureAspectFlags)TextureAspect::DEPTH_BIT |
             (TextureAspectFlags)TextureAspect::STENCIL_BIT;
    default:
      return (TextureAspectFlags)TextureAspect::COLOR_BIT;
  }
}

SE_EXPORT struct TextureView {
  /** virtual destructor */
  virtual ~TextureView() = default;
  /** get binded texture */
  virtual auto getTexture() noexcept -> Texture* = 0;
  /** set debug name */
  virtual auto setName(std::string const& name) -> void = 0;
  /** get width */
  virtual auto getWidth() noexcept -> uint32_t = 0;
  /** get height */
  virtual auto getHeight() noexcept -> uint32_t = 0;
};

SE_EXPORT enum struct TextureViewType : uint32_t {
  SRV,
  UAV,
  RTV,
  DSV,
};

SE_EXPORT struct TextureViewDescriptor {
  TextureFormat format;
  TextureViewDimension dimension = TextureViewDimension::TEX2D;
  TextureAspectFlags aspect = (uint32_t)TextureAspect::COLOR_BIT;
  uint32_t baseMipLevel = 0;
  uint32_t mipLevelCount = 1;
  uint32_t baseArrayLayer = 0;
  uint32_t arrayLayerCount = 1;
};

/**
 * A sampleable texture wrapping an external video object.
 * The contents of a GPUExternalTexture object are a snapshot and may not
 * change.
 */
SE_EXPORT struct ExternalTexture {
  /** virtual destructor */
  virtual ~ExternalTexture();
  /** indicates whether the texture has expired or not */
  virtual auto expired() const noexcept -> bool = 0;
};

SE_EXPORT struct ExternalTextureDescriptor {};

// Textures/TextureViews Interface
// ===========================================================================
// Samplers Interface

SE_EXPORT struct Sampler {
  /** virtual destructor */
  virtual ~Sampler() = default;
  /** set debug name */
  virtual auto setName(std::string const& name) -> void = 0;
  /** get debug name */
  virtual auto getName() const noexcept -> std::string const& = 0;
};

SE_EXPORT enum struct AddressMode {
  CLAMP_TO_EDGE,
  REPEAT,
  MIRROR_REPEAT,
};

SE_EXPORT enum struct FilterMode {
  NEAREST,
  LINEAR,
};

SE_EXPORT enum struct MipmapFilterMode {
  NEAREST,
  LINEAR,
};

SE_EXPORT enum struct CompareFunction {
  NEVER,
  LESS,
  EQUAL,
  LESS_EQUAL,
  GREATER,
  NOT_EQUAL,
  GREATER_EQUAL,
  ALWAYS,
};

SE_EXPORT struct SamplerDescriptor {
  AddressMode addressModeU = AddressMode::MIRROR_REPEAT;
  AddressMode addressModeV = AddressMode::MIRROR_REPEAT;
  AddressMode addressModeW = AddressMode::CLAMP_TO_EDGE;
  FilterMode magFilter = FilterMode::LINEAR;
  FilterMode minFilter = FilterMode::LINEAR;
  MipmapFilterMode mipmapFilter = MipmapFilterMode::LINEAR;
  float lodMinClamp = 0.f;
  float lodMapClamp = 32.f;
  CompareFunction compare = CompareFunction::ALWAYS;
  uint16_t maxAnisotropy = 1;
  float maxLod = 32.f;
};

// Samplers Interface
// ===========================================================================
// SwapChain Interface

SE_EXPORT struct SwapChain {
  /** virtual destructor */
  virtual ~SwapChain() = default;
  /** get texture view */
  virtual auto getTextureView(int i) noexcept -> TextureView* = 0;
  /** invalid swapchain */
  virtual auto recreate() noexcept -> void = 0;
};

SE_EXPORT struct SwapChainDescriptor {};

// SwapChain Interface
// ===========================================================================
// Resource Binding Interface

/**
 * Defines the interface between a set of resources bound
 * in a BindGroup and their accessibility in shader stages.
 */
SE_EXPORT struct BindGroupLayout {
  /** virtual destructor */
  virtual ~BindGroupLayout() = default;
  /** get BindGroup Layout Descriptor */
  virtual auto getBindGroupLayoutDescriptor() const noexcept
      -> BindGroupLayoutDescriptor const& = 0;
};

/** Determine how a Texture may be used after its creation. */
SE_EXPORT using ShaderStagesFlags = uint32_t;
/** Determine how a Texture may be used after its creation. */
SE_EXPORT enum struct ShaderStages {
  VERTEX = 1 << 0,
  FRAGMENT = 1 << 1,
  COMPUTE = 1 << 2,
  GEOMETRY = 1 << 3,
  RAYGEN = 1 << 4,
  MISS = 1 << 5,
  CLOSEST_HIT = 1 << 6,
  INTERSECTION = 1 << 7,
  ANY_HIT = 1 << 8,
  CALLABLE = 1 << 9,
  TASK = 1 << 10,
  MESH = 1 << 11,
};

SE_EXPORT enum struct BindingResourceType {
  SAMPLER,
  TEXTURE_VIEW,
  BUFFER_BINDING,
  EXTERNAL_TEXTURE,
  BINDLESS_TEXTURE,
  TLAS,
};

SE_EXPORT enum struct BufferBindingType { UNIFORM, STORAGE, READ_ONLY_STORAGE };

SE_EXPORT struct BufferBindingLayout {
  BufferBindingType type = BufferBindingType::UNIFORM;
  bool hasDynamicOffset = false;
  size_t minBindingSize = 0;
};

SE_EXPORT enum struct SamplerBindingType {
  FILTERING,
  NON_FILTERING,
  COMPARISON,
};

SE_EXPORT struct SamplerBindingLayout {
  SamplerBindingType type = SamplerBindingType::FILTERING;
};

SE_EXPORT enum struct TextureSampleType {
  FLOAT,
  UNFILTERABLE_FLOAT,
  DEPTH,
  SINT,
  UINT,
};

SE_EXPORT struct TextureBindingLayout {
  TextureSampleType sampleType = TextureSampleType::FLOAT;
  TextureViewDimension viewDimension = TextureViewDimension::TEX2D;
  bool multisampled = false;
};

SE_EXPORT enum struct StorageTextureAccess { WIRTE_ONLY };

SE_EXPORT struct StorageTextureBindingLayout {
  StorageTextureAccess access = StorageTextureAccess::WIRTE_ONLY;
  TextureFormat format;
  TextureViewDimension viewDimension = TextureViewDimension::TEX2D;
};

SE_EXPORT struct ExternalTextureBindingLayout {};

SE_EXPORT struct BindlessTexturesBindingLayout {};

SE_EXPORT struct AccelerationStructureBindingLayout {};

/**
 * Describes a single shader resource binding
 * to be included in a GPUBindGroupLayout.
 */
SE_EXPORT struct BindGroupLayoutEntry {
  /** initialization */
  BindGroupLayoutEntry(uint32_t binding, ShaderStagesFlags visibility,
                       BufferBindingLayout const& buffer)
      : binding(binding), visibility(visibility), buffer(buffer) {}
  BindGroupLayoutEntry(uint32_t binding, ShaderStagesFlags visibility,
                       SamplerBindingLayout const& sampler)
      : binding(binding), visibility(visibility), sampler(sampler) {}
  BindGroupLayoutEntry(uint32_t binding, ShaderStagesFlags visibility,
                       TextureBindingLayout const& texture)
      : binding(binding), visibility(visibility), texture(texture) {}
  BindGroupLayoutEntry(uint32_t binding, ShaderStagesFlags visibility,
                       TextureBindingLayout const& texture,
                       SamplerBindingLayout const& sampler)
      : binding(binding),
        visibility(visibility),
        texture(texture),
        sampler(sampler) {}
  BindGroupLayoutEntry(uint32_t binding, ShaderStagesFlags visibility,
                       StorageTextureBindingLayout const& storageTexture)
      : binding(binding),
        visibility(visibility),
        storageTexture(storageTexture) {}
  BindGroupLayoutEntry(uint32_t binding, ShaderStagesFlags visibility,
                       ExternalTextureBindingLayout const& externalTexture)
      : binding(binding),
        visibility(visibility),
        externalTexture(externalTexture) {}
  BindGroupLayoutEntry(
      uint32_t binding, ShaderStagesFlags visibility,
      AccelerationStructureBindingLayout const& accelerationStructure)
      : binding(binding),
        visibility(visibility),
        accelerationStructure(accelerationStructure) {}
  BindGroupLayoutEntry(uint32_t binding, ShaderStagesFlags visibility,
                       BindlessTexturesBindingLayout const& bindlessTextures)
      : binding(binding),
        visibility(visibility),
        bindlessTextures(bindlessTextures) {}

  /* A unique identifier for a resource binding within the BindGroupLayout */
  uint32_t binding;
  /* Array of elements binded on the binding position */
  uint32_t array_size = 1;
  /** indicates whether it will be accessible from the associated shader stage
   */
  ShaderStagesFlags visibility;
  // possible resources layout types
  // ---------------------------
  std::optional<BufferBindingLayout> buffer;
  std::optional<SamplerBindingLayout> sampler;
  std::optional<TextureBindingLayout> texture;
  std::optional<StorageTextureBindingLayout> storageTexture;
  std::optional<ExternalTextureBindingLayout> externalTexture;
  std::optional<AccelerationStructureBindingLayout> accelerationStructure;
  std::optional<BindlessTexturesBindingLayout> bindlessTextures;
};

SE_EXPORT struct BindGroupLayoutDescriptor {
  std::vector<BindGroupLayoutEntry> entries;
};

/**
 * Defines a set of resources to be bound together in a group
 * and how the resources are used in shader stages.
 */
struct BindGroupEntry;
SE_EXPORT struct BindGroup {
  virtual ~BindGroup() = default;
  /** update binding */
  virtual auto updateBinding(
      std::vector<BindGroupEntry> const& entries) noexcept -> void = 0;
};

SE_EXPORT struct BufferBinding {
  Buffer* buffer;
  size_t offset;
  size_t size;
};

SE_EXPORT struct BindingResource {
  BindingResource() = default;
  BindingResource(TextureView* view, Sampler* sampler)
      : type(BindingResourceType::SAMPLER),
        textureView(view),
        sampler(sampler) {}
  BindingResource(Sampler* sampler)
      : type(BindingResourceType::SAMPLER), sampler(sampler){}
  BindingResource(TextureView* view)
      : type(BindingResourceType::TEXTURE_VIEW), textureView(view) {}
  BindingResource(BufferBinding const& buffer)
      : type(BindingResourceType::BUFFER_BINDING), bufferBinding(buffer) {}
  // Binding bindless texture-sampler-pair array,
  // where every texture share the same sampler
  BindingResource(std::vector<TextureView*> const& bindlessTextures, Sampler* sampler)
      : type(BindingResourceType::BINDLESS_TEXTURE),
        bindlessTextures(bindlessTextures),
        sampler(sampler) {}
  // Binding bindless texture-sampler-pair array,
  // where every texture bind its own sampler
  BindingResource(std::vector<TextureView*> const& bindlessTextures,
                  std::vector<Sampler*> const& samplers)
      : type(BindingResourceType::BINDLESS_TEXTURE),
        bindlessTextures(bindlessTextures),
        samplers(samplers) {}
  // Binding bindless storage texture array
  BindingResource(std::vector<TextureView*> const& storageTextures)
      : type(BindingResourceType::TEXTURE_VIEW),
        storageArray(storageTextures) {}
  // Binding a tlas resource
  BindingResource(TLAS* tlas) : type(BindingResourceType::TLAS), tlas(tlas) {}
  BindingResourceType type;
  Sampler* sampler = nullptr;
  TextureView* textureView = nullptr;
  std::vector<Sampler*> samplers = {};
  std::vector<TextureView*> bindlessTextures = {};
  std::vector<TextureView*> storageArray = {};
  ExternalTexture* externalTexture = nullptr;
  std::optional<BufferBinding> bufferBinding;
  TLAS* tlas = nullptr;
};

SE_EXPORT struct BindGroupEntry {
  uint32_t binding;
  BindingResource resource;
};

SE_EXPORT struct BindGroupDescriptor {
  BindGroupLayout* layout;
  std::vector<BindGroupEntry> entries;
};

/**
 * Defines the mapping between resources of all BindGroup objects set up
 * during command encoding in setBindGroup(), and the shaders of the pipeline
 * set by RenderCommandsMixin.setPipeline or ComputePassEncoder.setPipeline.
 */
SE_EXPORT struct PipelineLayout {
  /** virtual destructor */
  virtual ~PipelineLayout() = default;
};

SE_EXPORT struct PushConstantEntry {
  ShaderStagesFlags shaderStages;
  uint32_t offset;
  uint32_t size;
};

SE_EXPORT struct PipelineLayoutDescriptor {
  std::vector<PushConstantEntry> pushConstants;
  std::vector<BindGroupLayout*> bindGroupLayouts;
};

// Resource Binding Interface
// ===========================================================================
// Shader Modules Interface

/** An internal shader module object */
SE_EXPORT struct ShaderModule {
  /** virtual destrucor */
  virtual ~ShaderModule() = default;
  /** set debug name */
  virtual auto setName(std::string const& name) -> void = 0;
  /** get debug name */
  virtual auto getName() -> std::string const& = 0;
};

SE_EXPORT struct ShaderModuleDescriptor {
  /** The shader source code for the shader module. */
  Core::Buffer* code;
  /** stage */
  ShaderStages stage;
  /** name of entry point */
  std::string name = "main";
};

SE_EXPORT struct ShaderModuleCompilationHint {
  /**
   * if the layout is not given, reflection will
   * be used to automatically generate layout
   */
  std::optional<PipelineLayout> layout;
};

// Shader Modules Interface
// ===========================================================================
// Pipelines Interface

SE_EXPORT struct PipelineBase {
  /** Gets a BindGroupLayout that is compatible with the PipelineBase's
   * BindGroupLayout at index.*/
  auto getBindGroupLayout(size_t index) noexcept -> BindGroupLayout&;
};

SE_EXPORT using PipelineConstantValue = double;

/**
 * Describes the entry point in the user-provided ShaderModule that
 * controls one of the programmable stages of a pipeline.
 */
SE_EXPORT struct ProgrammableStage {
  ShaderModule* module;
  std::string entryPoint;
};

SE_EXPORT struct PipelineDescriptorBase {
  /** The definition of the layout of resources which can be used with this */
  PipelineLayout* layout = nullptr;
};

SE_EXPORT struct ComputePipeline {
  /** virtual destructor */
  virtual ~ComputePipeline() = default;
  /** set debug name */
  virtual auto setName(std::string const& name) -> void = 0;
};

SE_EXPORT struct ComputePipelineDescriptor : public PipelineDescriptorBase {
  /* the compute shader */
  ProgrammableStage compute;
};

SE_EXPORT enum struct PrimitiveTopology {
  POINT_LIST,
  LINE_LIST,
  LINE_STRIP,
  TRIANGLE_LIST,
  TRIANGLE_STRIP
};

/**
 * Determine polygons with vertices whose framebuffer coordinates
 * are given in which order are considered front-facing.
 */
SE_EXPORT enum struct FrontFace {
  CCW,  // counter-clockwise
  CW,   // clockwise
};

SE_EXPORT enum struct CullMode {
  NONE,
  FRONT,
  BACK,
  BOTH,
};

SE_EXPORT using SampleMask = uint32_t;

SE_EXPORT struct MultisampleState {
  uint32_t count = 1;
  SampleMask mask = 0xFFFFFFFF;
  bool alphaToCoverageEnabled = false;
};

SE_EXPORT enum struct BlendFactor {
  ZERO,
  ONE,
  SRC,
  ONE_MINUS_SRC,
  SRC_ALPHA,
  ONE_MINUS_SRC_ALPHA,
  DST,
  ONE_MINUS_DST,
  DST_ALPHA,
  ONE_MINUS_DST_ALPHA,
  SRC_ALPHA_SATURATED,
  CONSTANT,
  ONE_MINUS_CONSTANT,
};

SE_EXPORT enum struct BlendOperation {
  ADD,
  SUBTRACT,
  REVERSE_SUBTRACT,
  MIN,
  MAX,
};

SE_EXPORT struct BlendComponent {
  BlendOperation operation = BlendOperation::ADD;
  BlendFactor srcFactor = BlendFactor::ONE;
  BlendFactor dstFactor = BlendFactor::ZERO;
};

SE_EXPORT struct BlendState {
  BlendComponent color;
  BlendComponent alpha;
  /** check whether the blend should be enabled */
  inline auto blendEnable() const noexcept -> bool {
    return !((color.operation == BlendOperation::ADD) &&
             (color.srcFactor == BlendFactor::ONE) &&
             (color.dstFactor == BlendFactor::ZERO) &&
             (alpha.operation == BlendOperation::ADD) &&
             (alpha.srcFactor == BlendFactor::ONE) &&
             (alpha.dstFactor == BlendFactor::ZERO));
  }
};

SE_EXPORT using ColorWriteFlags = uint32_t;

SE_EXPORT struct ColorTargetState {
  TextureFormat format;
  BlendState blend;
  ColorWriteFlags writeMask = 0xF;
};

SE_EXPORT struct FragmentState : public ProgrammableStage {
  std::vector<ColorTargetState> targets;
};

SE_EXPORT using StencilValue = uint32_t;
SE_EXPORT using DepthBias = int32_t;

/** Could specify features in rasterization stage.
* Like: Conservative rasterization. */
SE_EXPORT struct RasterizeState {
  // The mode of conservative rasterization.
  // Read the article from NV for more information and understanding:
  // @url: https://developer.nvidia.com/content/dont-be-conservative-conservative-rasterization
  enum struct ConservativeMode {
    DISABLED,
    OVERESTIMATE,
    UNDERESTIMATE,
  } mode = ConservativeMode::DISABLED;
  // extraPrimitiveOverestimationSize is the extra size in pixels to increase
  // the generating primitive during conservative rasterization at each of its
  // edges in X and Y equally in screen space beyond the base overestimation
  // specified in
  // VkPhysicalDeviceConservativeRasterizationPropertiesEXT::primitiveOverestimationSize.
  // If conservativeRasterizationMode is not
  // VK_CONSERVATIVE_RASTERIZATION_MODE_OVERESTIMATE_EXT, this value is ignored.
  float extraPrimitiveOverestimationSize = 0.f;
};

SE_EXPORT enum struct StencilOperation {
  KEEP,
  ZERO,
  REPLACE,
  INVERT,
  INCREMENT_CLAMP,
  DECREMENT_CLAMP,
  INCREMENT_WARP,
  DECREMENT_WARP,
};

SE_EXPORT enum struct DataFormat {
  UINT8X2,
  UINT8X4,
  SINT8X2,
  SINT8X4,
  UNORM8X2,
  UNORM8X4,
  SNORM8X2,
  SNORM8X4,
  UINT16X2,
  UINT16X4,
  SINT16X2,
  SINT16X4,
  UNORM16X2,
  UNORM16X4,
  SNORM16X2,
  SNORM16X4,
  FLOAT16X2,
  FLOAT16X4,
  FLOAT32,
  FLOAT32X2,
  FLOAT32X3,
  FLOAT32X4,
  UINT32,
  UINT32X2,
  UINT32X3,
  UINT32X4,
  SINT32,
  SINT32X2,
  SINT32X3,
  SINT32X4
};

SE_EXPORT using VertexFormat = DataFormat;

SE_EXPORT inline auto getVertexFormatSize(VertexFormat format) noexcept -> size_t {
  switch (format) {
    case SIByL::RHI::VertexFormat::UINT8X2:
      return sizeof(uint8_t) * 2;
    case SIByL::RHI::VertexFormat::UINT8X4:
      return sizeof(uint8_t) * 4;
    case SIByL::RHI::VertexFormat::SINT8X2:
      return sizeof(uint8_t) * 2;
    case SIByL::RHI::VertexFormat::SINT8X4:
      return sizeof(uint8_t) * 4;
    case SIByL::RHI::VertexFormat::UNORM8X2:
      return sizeof(uint8_t) * 2;
    case SIByL::RHI::VertexFormat::UNORM8X4:
      return sizeof(uint8_t) * 4;
    case SIByL::RHI::VertexFormat::SNORM8X2:
      return sizeof(uint8_t) * 2;
    case SIByL::RHI::VertexFormat::SNORM8X4:
      return sizeof(uint8_t) * 4;
    case SIByL::RHI::VertexFormat::UINT16X2:
      return sizeof(uint16_t) * 2;
    case SIByL::RHI::VertexFormat::UINT16X4:
      return sizeof(uint16_t) * 4;
    case SIByL::RHI::VertexFormat::SINT16X2:
      return sizeof(uint16_t) * 2;
    case SIByL::RHI::VertexFormat::SINT16X4:
      return sizeof(uint16_t) * 4;
    case SIByL::RHI::VertexFormat::UNORM16X2:
      return sizeof(uint16_t) * 2;
    case SIByL::RHI::VertexFormat::UNORM16X4:
      return sizeof(uint16_t) * 4;
    case SIByL::RHI::VertexFormat::SNORM16X2:
      return sizeof(uint16_t) * 2;
    case SIByL::RHI::VertexFormat::SNORM16X4:
      return sizeof(uint16_t) * 4;
    case SIByL::RHI::VertexFormat::FLOAT16X2:
      return sizeof(uint16_t) * 2;
    case SIByL::RHI::VertexFormat::FLOAT16X4:
      return sizeof(uint16_t) * 4;
    case SIByL::RHI::VertexFormat::FLOAT32:
      return sizeof(float) * 1;
    case SIByL::RHI::VertexFormat::FLOAT32X2:
      return sizeof(float) * 2;
    case SIByL::RHI::VertexFormat::FLOAT32X3:
      return sizeof(float) * 3;
    case SIByL::RHI::VertexFormat::FLOAT32X4:
      return sizeof(float) * 4;
    case SIByL::RHI::VertexFormat::UINT32:
      return sizeof(uint32_t) * 1;
    case SIByL::RHI::VertexFormat::UINT32X2:
      return sizeof(uint32_t) * 2;
    case SIByL::RHI::VertexFormat::UINT32X3:
      return sizeof(uint32_t) * 3;
    case SIByL::RHI::VertexFormat::UINT32X4:
      return sizeof(uint32_t) * 4;
    case SIByL::RHI::VertexFormat::SINT32:
      return sizeof(int32_t) * 1;
    case SIByL::RHI::VertexFormat::SINT32X2:
      return sizeof(int32_t) * 2;
    case SIByL::RHI::VertexFormat::SINT32X3:
      return sizeof(int32_t) * 3;
    case SIByL::RHI::VertexFormat::SINT32X4:
      return sizeof(int32_t) * 4;
    default:
      return 0;
  }
}

SE_EXPORT enum struct VertexStepMode {
  VERTEX,
  INSTANCE,
};

SE_EXPORT enum struct IndexFormat { UINT16_t, UINT32_T };

SE_EXPORT struct PrimitiveState {
  PrimitiveTopology topology = PrimitiveTopology::TRIANGLE_LIST;
  IndexFormat stripIndexFormat;
  FrontFace frontFace = FrontFace::CCW;
  CullMode cullMode = CullMode::NONE;
  bool unclippedDepth = false;
};

SE_EXPORT struct StencilFaceState {
  CompareFunction compare = CompareFunction::ALWAYS;
  StencilOperation failOp = StencilOperation::KEEP;
  StencilOperation depthFailOp = StencilOperation::KEEP;
  StencilOperation passOp = StencilOperation::KEEP;
};

SE_EXPORT struct DepthStencilState {
  TextureFormat format;
  bool depthWriteEnabled = false;
  CompareFunction depthCompare = CompareFunction::ALWAYS;
  StencilFaceState stencilFront = {};
  StencilFaceState stencilBack = {};
  StencilValue stencilReadMask = 0xFFFFFFFF;
  StencilValue stencilWriteMask = 0xFFFFFFFF;
  DepthBias depthBias = 0;
  float depthBiasSlopeScale = 0;
  float depthBiasClamp = 0;
};

SE_EXPORT struct VertexAttribute {
  VertexFormat format;
  size_t offset;
  uint32_t shaderLocation;
};

SE_EXPORT struct VertexBufferLayout {
  size_t arrayStride;
  VertexStepMode stepMode = VertexStepMode::VERTEX;
  std::vector<VertexAttribute> attributes;
};

SE_EXPORT struct VertexState : public ProgrammableStage {
  std::vector<VertexBufferLayout> buffers;
};

SE_EXPORT struct RenderPipeline {
  /** virtual destructor */
  virtual ~RenderPipeline() = default;
  /** virtual constructor */
  RenderPipeline() = default;
  RenderPipeline(RenderPipeline&&) = default;
  RenderPipeline(RenderPipeline const&) = delete;
  auto operator=(RenderPipeline&&) -> RenderPipeline& = default;
  auto operator=(RenderPipeline const&) -> RenderPipeline& = delete;
  /** set debug name */
  virtual auto setName(std::string const& name) -> void = 0;
};

/** Describes a render pipeline by configuring each of the render stages. */
SE_EXPORT struct RenderPipelineDescriptor : public PipelineDescriptorBase {
  // required structures
  // --------------------------------------------------------------
  VertexState vertex;                   // vertex shader stage
  PrimitiveState primitive = {};        // vertex primitive config
  DepthStencilState depthStencil;       // OM pipeline config
  MultisampleState multisample = {};    // multisample config
  FragmentState fragment;               // fragment shader stage
  // optional structures
  // --------------------------------------------------------------
  ProgrammableStage geometry;           // geometry shader stage
  ProgrammableStage task;               // task shader stage
  ProgrammableStage mesh;               // mesh shader stage
  RasterizeState rasterize;             // rasterize pipeline stage
};

// Pipelines Interface
// ===========================================================================
// Command Buffers Interface

/**
 * Pre-recorded lists of GPU commands that can be submitted to a GPUQueue
 * for execution. Each GPU command represents a task to be performed on the
 * GPU, such as setting state, drawing, copying resources, etc.
 *
 * A CommandBuffer can only be submitted once, at which point it becomes
 * invalid. To reuse rendering commands across multiple submissions, use
 * RenderBundle.
 */
SE_EXPORT struct CommandBuffer {};

SE_EXPORT struct CommandBufferDescriptor {};

// Command Buffers Interface
// ===========================================================================
// Command Encoding Interface

struct CommandsMixin;
struct RenderPassEncoder;
struct RenderPassDescriptor;
struct ComputePassEncoder;
struct ComputePassDescriptor;
struct ImageCopyBuffer;
struct ImageCopyTexture;
struct RenderBundle;

/**
 * CommandsMixin defines state common to all interfaces
 * which encode commands. It has no methods.
 */
SE_EXPORT struct CommandsMixin {};

SE_EXPORT struct DebugUtilLabelDescriptor {
  std::string name;
  Math::vec4 color;
};

SE_EXPORT struct CommandEncoder {
  /** virtual descructor */
  virtual ~CommandEncoder() = default;
  /** Begins encoding a render pass described by descriptor. */
  virtual auto beginRenderPass(RenderPassDescriptor const& desc) noexcept
      -> std::unique_ptr<RenderPassEncoder> = 0;
  /** Begins encoding a compute pass described by descriptor. */
  virtual auto beginComputePass(ComputePassDescriptor const& desc) noexcept
      -> std::unique_ptr<ComputePassEncoder> = 0;
  /** Begins encoding a ray tracing pass described by descriptor. */
  virtual auto beginRayTracingPass(
      RayTracingPassDescriptor const& desc) noexcept
      -> std::unique_ptr<RayTracingPassEncoder> = 0;
  /** Insert a barrier. */
  virtual auto pipelineBarrier(BarrierDescriptor const& desc) noexcept
      -> void = 0;
  /** Encode a command into the CommandEncoder that copies data from
   * a sub-region of a GPUBuffer to a sub-region of another Buffer. */
  virtual auto copyBufferToBuffer(Buffer* source, size_t sourceOffset,
                                  Buffer* destination, size_t destinationOffset,
                                  size_t size) noexcept -> void = 0;
  /** Encode a command into the CommandEncoder that fills a sub-region of a
   * Buffer with zeros. */
  virtual auto clearBuffer(Buffer* buffer, size_t offset, size_t size) noexcept
      -> void = 0;
  /** Encode a command into the CommandEncoder that fills a sub-region of a
   * texture with any color. */
  virtual auto clearTexture(Texture* texture,
                            TextureClearDescriptor const& desc) noexcept
      -> void = 0;
  /** Encode a command into the CommandEncoder that fills a sub-region of a
   * Buffer with a value. */
  virtual auto fillBuffer(Buffer* buffer, size_t offset, size_t size,
                          float fillValue) noexcept -> void = 0;
  /** Encode a command into the CommandEncoder that copies data from a
   * sub-region of a Buffer to a sub-region of one or multiple continuous
   * texture subresources. */
  virtual auto copyBufferToTexture(ImageCopyBuffer const& source,
                                   ImageCopyTexture const& destination,
                                   Extend3D const& copySize) noexcept
      -> void = 0;
  /** Encode a command into the CommandEncoder that copies data from a
   * sub-region of one or multiple continuous texture subresourcesto a
   * sub-region of a Buffer. */
  virtual auto copyTextureToBuffer(ImageCopyTexture const& source,
                                   ImageCopyBuffer const& destination,
                                   Extend3D const& copySize) noexcept
      -> void = 0;
  /** Encode a command into the CommandEncoder that copies data from
   * a sub-region of one or multiple contiguous texture subresources to
   * another sub-region of one or multiple continuous texture subresources. */
  virtual auto copyTextureToTexture(ImageCopyTexture const& source,
                                    ImageCopyTexture const& destination,
                                    Extend3D const& copySize) noexcept
      -> void = 0;
  /** Reset the queryset. */
  virtual auto resetQuerySet(QuerySet* querySet, uint32_t firstQuery,
                             uint32_t queryCount) noexcept -> void = 0;
  /** Writes a timestamp value into a querySet when all
   * previous commands have completed executing. */
  virtual auto writeTimestamp(QuerySet* querySet, PipelineStages stageMask,
                              uint32_t queryIndex) noexcept -> void = 0;
  /** Resolves query results from a QuerySet out into a range of a Buffer. */
  virtual auto resolveQuerySet(QuerySet* querySet, uint32_t firstQuery,
                               uint32_t queryCount, Buffer& destination,
                               uint64_t destinationOffset) noexcept -> void = 0;
  /** copy accceleration structure to a new instance */
  virtual auto cloneBLAS(BLAS* src) noexcept -> std::unique_ptr<BLAS> = 0;
  /** update blas by refitting, only deformation is allowed */
  virtual auto updateBLAS(BLAS* src, Buffer* vertexBuffer,
                          Buffer* indexBuffer) noexcept -> void = 0;
  /** Completes recording of the commands sequence and returns a corresponding
   * GPUCommandBuffer. */
  virtual auto finish(
      std::optional<CommandBufferDescriptor> const& descriptor = {}) noexcept
      -> CommandBuffer* = 0;
  /** begin Debug marker region */
  virtual auto beginDebugUtilsLabelEXT(
      DebugUtilLabelDescriptor const& desc) noexcept -> void = 0;
  /** end Debug marker region */
  virtual auto endDebugUtilsLabelEXT() noexcept -> void = 0;
};

SE_EXPORT struct CommandEncoderDescriptor {
  CommandBuffer* externalCommandBuffer = nullptr;
};

SE_EXPORT struct ImageDataLayout {
  uint64_t offset = 0;
  uint32_t bytesPerRow;
  uint32_t rowsPerImage;
};

SE_EXPORT struct ImageCopyBuffer : ImageDataLayout {
  Buffer* buffer;
};

SE_EXPORT using IntegerCoordinate = uint32_t;

SE_EXPORT struct Origin2DDict {
  IntegerCoordinate x = 0;
  IntegerCoordinate y = 0;
};
SE_EXPORT using Origin2D =
    std::variant<Origin2DDict, std::vector<IntegerCoordinate>>;

SE_EXPORT struct Origin3DDict {
  IntegerCoordinate x = 0;
  IntegerCoordinate y = 0;
  IntegerCoordinate z = 0;
};

SE_EXPORT using Origin3D = Origin3DDict;

SE_EXPORT struct ImageCopyTexture {
  Texture* texutre;
  uint32_t mipLevel = 0;
  Origin3D origin = {};
  TextureAspectFlags aspect = (TextureAspectFlags)TextureAspect::COLOR_BIT;
};

SE_EXPORT struct ImageCopyTextureTagged : public ImageCopyTexture {
  bool premultipliedAlpha = false;
};

SE_EXPORT struct ImageCopyExternalImage {
  Origin2D origin = {};
  bool flipY = false;
};

// Command Encoding Interface
// ===========================================================================
// Programmable Passes Interface

SE_EXPORT using BufferDynamicOffset = uint32_t;

SE_EXPORT struct BindingCommandMixin {
  /** virtual destructor */
  virtual ~BindingCommandMixin() = default;
  /** Sets the current GPUBindGroup for the given index. */
  virtual auto setBindGroup(
      uint32_t index, BindGroup* bindgroup,
      std::vector<BufferDynamicOffset> const& dynamicOffsets = {}) noexcept
      -> void = 0;
  /** Sets the current GPUBindGroup for the given index. */
  virtual auto setBindGroup(uint32_t index, BindGroup* bindgroup,
                            uint64_t dynamicOffsetDataStart,
                            uint32_t dynamicOffsetDataLength) noexcept
      -> void = 0;
  /** Push constants */
  virtual auto pushConstants(void* data, ShaderStagesFlags stages,
                             uint32_t offset, uint32_t size) noexcept
      -> void = 0;
};

// Programmable Passes Interface
// ===========================================================================
// Debug Marks Interface

SE_EXPORT struct DebugCommandMixin {
  auto pushDebugGroup(std::string const& groupLabel) noexcept -> void;
  auto popDebugGroup() noexcept -> void;
  auto insertDebugMarker(std::string const& markerLabel) noexcept -> void;
};

// Debug Marks Interface
// ===========================================================================
// Compute Passes Interface

SE_EXPORT struct ComputePassEncoder : public BindingCommandMixin {
  /** virtual descructor */
  virtual ~ComputePassEncoder() = default;
  /** Sets the current GPUComputePipeline. */
  virtual auto setPipeline(ComputePipeline* pipeline) noexcept -> void = 0;
  /** Dispatch work to be performed with the current GPUComputePipeline.*/
  virtual auto dispatchWorkgroups(uint32_t workgroupCountX,
                                  uint32_t workgroupCountY = 1,
                                  uint32_t workgroupCountZ = 1) noexcept
      -> void = 0;
  /** Dispatch work to be performed with the current GPUComputePipeline using
   * parameters read from a GPUBuffer. */
  virtual auto dispatchWorkgroupsIndirect(Buffer* indirectBuffer,
                                          uint64_t indirectOffset) noexcept
      -> void = 0;
  /** Completes recording of the compute pass commands sequence. */
  virtual auto end() noexcept -> void = 0;
};

SE_EXPORT enum struct ComputePassTimestampLocation {
  BEGINNING,
  END,
};

SE_EXPORT struct ComputePassTimestampWrite {
  std::unique_ptr<QuerySet> querySet = nullptr;
  uint32_t queryIndex;
  ComputePassTimestampLocation location;
};

SE_EXPORT using ComputePassTimestampWrites =
    std::vector<ComputePassTimestampWrite>;

SE_EXPORT struct ComputePassDescriptor {
  ComputePassTimestampWrites timestampWrites = {};
};

// Compute Passes Interface
// ===========================================================================
// Render Passes Interface

struct RenderPassColorAttachment;
struct RenderPassDepthStencilAttachment;

/**
 * RenderCommandsMixin defines rendering commands common to
 * RenderPassEncoder and RenderBundleEncoder.
 */
SE_EXPORT struct RenderCommandsMixin {
  /** virtual descructor */
  virtual ~RenderCommandsMixin() = default;
  /** Sets the current GPURenderPipeline. */
  virtual auto setPipeline(RenderPipeline* pipeline) noexcept -> void = 0;
  /** Sets the current index buffer. */
  virtual auto setIndexBuffer(Buffer* buffer, IndexFormat indexFormat,
                              uint64_t offset = 0, uint64_t size = 0) noexcept
      -> void = 0;
  /** Sets the current vertex buffer for the given slot. */
  virtual auto setVertexBuffer(uint32_t slot, Buffer* buffer,
                               uint64_t offset = 0, uint64_t size = 0) noexcept
      -> void = 0;
  /** Draws primitives. */
  virtual auto draw(uint32_t vertexCount, uint32_t instanceCount = 1,
                    uint32_t firstVertex = 0,
                    uint32_t firstInstance = 0) noexcept -> void = 0;
  /** Draws indexed primitives. */
  virtual auto drawIndexed(uint32_t indexCount, uint32_t instanceCount = 1,
                           uint32_t firstIndex = 0, int32_t baseVertex = 0,
                           uint32_t firstInstance = 0) noexcept -> void = 0;
  /** Draws primitives using parameters read from a GPUBuffer. */
  virtual auto drawIndirect(Buffer* indirectBuffer, uint64_t indirectOffset,
                            uint32_t drawCount, uint32_t stride) noexcept
      -> void = 0;
  /** Draws indexed primitives using parameters read from a GPUBuffer. */
  virtual auto drawIndexedIndirect(Buffer* indirectBuffer, uint64_t offset,
                                   uint32_t drawCount, uint32_t stride) noexcept
      -> void = 0;
};

SE_EXPORT struct RenderPassEncoder : public RenderCommandsMixin,
                                  public BindingCommandMixin {
  /** virtual descructor */
  virtual ~RenderPassEncoder() = default;
  /** Sets the viewport used during the rasterization stage to linearly map
   * from normalized device coordinates to viewport coordinates. */
  virtual auto setViewport(float x, float y, float width, float height,
                           float minDepth, float maxDepth) noexcept -> void = 0;
  /** Sets the scissor rectangle used during the rasterization stage.
   * After transformation into viewport coordinates any fragments
   * which fall outside the scissor rectangle will be discarded. */
  virtual auto setScissorRect(IntegerCoordinate x, IntegerCoordinate y,
                              IntegerCoordinate width,
                              IntegerCoordinate height) noexcept -> void = 0;
  /** Sets the constant blend color and alpha values used with
   * "constant" and "one-minus-constant" GPUBlendFactors. */
  virtual auto setBlendConstant(Color color) noexcept -> void = 0;
  /** Sets the [[stencil_reference]] value used during
   * stencil tests with the "replace" GPUStencilOperation. */
  virtual auto setStencilReference(StencilValue reference) noexcept -> void = 0;
  /** begin occlusion query */
  virtual auto beginOcclusionQuery(uint32_t queryIndex) noexcept -> void = 0;
  /** end occlusion query */
  virtual auto endOcclusionQuery() noexcept -> void = 0;
  /** Executes the commands previously recorded into the given GPURenderBundles
   * as part of this render pass. */
  virtual auto executeBundles(std::vector<RenderBundle> const& bundles) noexcept
      -> void = 0;
  /** Completes recording of the render pass commands sequence. */
  virtual auto end() noexcept -> void = 0;
};

SE_EXPORT enum struct RenderPassTimestampLocation {
  BEGINNING,
  END,
};

SE_EXPORT struct RenderPassTimestampWrite {
  QuerySet* querySet;
  uint32_t queryIndex;
  RenderPassTimestampLocation location;
};

SE_EXPORT using RenderPassTimestampWrites = std::vector<RenderPassTimestampWrite>;

SE_EXPORT enum struct LoadOp {
  DONT_CARE,
  LOAD,
  CLEAR,
};

SE_EXPORT enum struct StoreOp { DONT_CARE, STORE, DISCARD };

SE_EXPORT struct RenderPassColorAttachment {
  TextureView* view;
  TextureView* resolveTarget = nullptr;
  Color clearValue;
  LoadOp loadOp;
  StoreOp storeOp;
};

SE_EXPORT struct RenderPassDepthStencilAttachment {
  TextureView* view = nullptr;
  float depthClearValue = 0;
  LoadOp depthLoadOp;
  StoreOp depthStoreOp;
  bool depthReadOnly = false;
  StencilValue stencilClearValue = 0;
  LoadOp stencilLoadOp;
  StoreOp stencilStoreOp;
  bool stencilReadOnly = false;
};

SE_EXPORT struct RenderPassDescriptor {
  std::vector<RenderPassColorAttachment> colorAttachments;
  RenderPassDepthStencilAttachment depthStencilAttachment;
  // std::unique_ptr<QuerySet> occlusionQuerySet = nullptr;
  RenderPassTimestampWrites timestampWrites = {};
  uint64_t maxDrawCount = 50000000;
};

SE_EXPORT struct RenderPassLayout {
  std::vector<TextureFormat> colorFormats;
  TextureFormat depthStencilFormat;
  uint32_t sampleCount = 1;
};

// Render Passes Interface
// ===========================================================================
// Bundles Interface

SE_EXPORT struct RenderBundle {};

SE_EXPORT struct RenderBundleDescriptor {};

SE_EXPORT struct RenderBundleEncoder {
  auto finish(
      std::optional<RenderBundleDescriptor> const& descriptor = {}) noexcept
      -> RenderBundle;
};

SE_EXPORT struct RenderBundleEncoderDescriptor : public RenderPassLayout {
  bool depthReadOnly = false;
  bool stencilReadOnly = false;
};

// Bundles Interface
// ===========================================================================
// Ray Tracing Interface

SE_EXPORT struct RayTracingPassEncoder : public BindingCommandMixin {
  /** virtual destructor */
  virtual ~RayTracingPassEncoder() = default;
  /** set a ray tracing pipeline as the current pipeline */
  virtual auto setPipeline(RayTracingPipeline* pipeline) noexcept -> void = 0;
  /** trace rays using current ray tracing pipeline */
  virtual auto traceRays(uint32_t width, uint32_t height,
                         uint32_t depth) noexcept -> void = 0;
  /** trace rays using current ray tracing pipeline by an indirect buffer */
  virtual auto traceRaysIndirect(Buffer* indirectBuffer,
                                 uint64_t indirectOffset) noexcept -> void = 0;
  /**  end the ray tracing pass */
  virtual auto end() noexcept -> void = 0;
};

SE_EXPORT struct RayTracingPassDescriptor {};

// Ray Tracing Interface
// ===========================================================================
// Queue Interface

SE_EXPORT struct Queue {
  /** virtual destructor */
  virtual ~Queue() = default;
  /** Schedules the execution of the command buffers by the GPU on this queue.
   */
  virtual auto submit(
      std::vector<CommandBuffer*> const& commandBuffers) noexcept -> void = 0;
  /** Schedules the execution of the command buffers by the GPU on this queue.
   * With sync objects */
  virtual auto submit(std::vector<CommandBuffer*> const& commandBuffers, Fence* fence) noexcept
      -> void = 0;
  /** Schedules the execution of the command buffers by the GPU on this queue.
   * With sync objects */
  virtual auto submit(std::vector<CommandBuffer*> const& commandBuffers,
                      Semaphore* wait, Semaphore* signal, Fence* fence) noexcept
      -> void = 0;
  /** Returns a Promise that resolves once this queue finishes
   * processing all the work submitted up to this moment. */
  virtual auto onSubmittedWorkDone() noexcept -> std::future<bool> = 0;
  /** Issues a write operation of the provided data into a Buffer. */
  virtual auto writeBuffer(Buffer* buffer, uint64_t bufferOffset,
                           ArrayBuffer* data, uint64_t dataOffset,
                           Extend3D const& size) noexcept -> void = 0;
  /** Issues a write operation of the provided data into a Texture. */
  virtual auto writeTexture(ImageCopyTexture const& destination,
                            ArrayBuffer* data, ImageDataLayout const& layout,
                            Extend3D const& size) noexcept -> void = 0;
  /** Issues a copy operation of the contents of a platform
   * image/canvas into the destination texture. */
  virtual auto copyExternalImageToTexture(
      ImageCopyExternalImage const& source,
      ImageCopyExternalImage const& destination,
      Extend3D const& copySize) noexcept -> void = 0;
  /** Present swap chain. */
  virtual auto presentSwapChain(SwapChain* swapchain, uint32_t imageIndex,
                                Semaphore* semaphore) noexcept -> void = 0;
  /** wait until idle */
  virtual auto waitIdle() noexcept -> void = 0;
  /** set debug name */
  virtual auto setName(std::string const& name) -> void = 0;
};

/** Describes a queue request */
SE_EXPORT struct QueueDescriptor {};

// Queue Interface
// ===========================================================================
// Queries Interface

SE_EXPORT enum struct QueryType { 
    OCCLUSION, 
    PIPELINE_STATISTICS,
    TIMESTAMP,
};

SE_EXPORT enum struct QueryResultBits {
  RESULT_64 = 0x1,
  RESULT_WAIT = 0x1 << 1,
  RESULT_WITH_AVAILABILITY = 0x1 << 2,
  RESULT_PARTIAL = 0x1 << 3,
};
SE_EXPORT using QueryResultFlag = uint32_t;

SE_EXPORT struct QuerySet {
  virtual ~QuerySet() = default;
  /** fetch the query pooll result */
  virtual auto resolveQueryResult(uint32_t firstQuery, uint32_t queryCount,
                                  size_t dataSize, void* pData, uint64_t stride,
                                  QueryResultFlag flag) noexcept -> void = 0;
  QueryType type;
  uint32_t count;
};

SE_EXPORT struct QuerySetDescriptor {
  QueryType type;
  uint32_t count;
};

// Queries Interface
// ===========================================================================
// Synchronization Interface

/**
 * ╔═════════════════╗
 * ║      Fence      ║
 * ╚═════════════════╝
 * Fences are objects used to synchronize the CPU and GPU.
 * Both the CPU and GPU can be instructed to wait at a fence so that the other
 * can catch up. This can be used to manage resource allocation and
 * deallocation, making it easier to manage overall graphics memory usage.
 *
 * To signal a fence, all previously submitted commands to the queue must
 * complete. We will also get a full memory barrier that all pending writes are
 * made available
 *
 * However, fence would not make memory available to th CPU
 * Therefore if we do a CPU read, an extra barrier with ACCESS_HOST_READ_BIT
 * flag should be used. In mental model, we can think this as flushing GPU L2
 * cache out to GPU main memory
 *
 * ╭──────────────┬──────────────────╮
 * │  Vulkan		 │   vk::Fence      │
 * │  DirectX 12  │   ID3D12Fence    │
 * │  OpenGL      │   glFenceSync    │
 * ╰──────────────┴──────────────────╯
 */
SE_EXPORT struct Fence {
  /** virtual desctructor */
  virtual ~Fence() = default;
  /* wait the fence */
  virtual auto wait() noexcept -> void = 0;
  /* reset the fence */
  virtual auto reset() noexcept -> void = 0;
};

/**
 * ╔════════════════════╗
 * ║      Barriers      ║
 * ╚════════════════════╝
 * Barrier is a more granular form of synchronization, inside command buffers.
 * It is not a "resource" but a "command",
 * it divide all commands on queue into two parts: "before" and "after",
 * and indicate dependency of {a subset of "after"} on {another subset of
 * "before"}
 *
 * As a command, input parameters are:
 *  ► PipelineStageFlags    - srcStageMask
 *  ► PipelineStageFlags    - dstStageMask
 *  ► DependencyFlags       - dependencyFlags
 *  ► [MemoryBarrier]       - pBufferMemoryBarriers
 *  ► [BufferMemoryBarrier] - pBufferMemoryBarriers
 *  ► [ImageMemoryBarrier]  - pImageMemoryBarriers
 *
 * In cmdPipelineBarrier, we are specifying 4 things to happen in order:
 * 1. Wait for srcStageMask to complete
 * 2. Make all writes performed in possible combinations of srcStageMasks +
 * srcAccessMask available
 * 3. Make available memory visible to possible combination of dstStageMask +
 * dstAccessMask
 * 4. Unblock work in dstStageMask
 *
 * ╭──────────────┬───────────────────────────╮
 * │  Vulkan		 │   vkCmdPipelineBarrier    │
 * │  DirectX 12  │   D3D12_RESOURCE_BARRIER  │
 * │  OpenGL      │   glMemoryBarrier         │
 * ╰──────────────┴───────────────────────────╯
 *
 *  ╭╱─────────────────────────────────────────╲╮
 *  ╳   Execution Barrier - Source Stage Mask   ╳
 *  ╰╲─────────────────────────────────────────╱╯
 * This present what we are waiting for.
 * Essentially, it is every commands before this one.
 * The mask can restrict the scrope of what we are waiting for.
 *
 *  ╭╱─────────────────────────────────────────╲╮
 *  ╳    Execution Barrier - Dst Stage Mask     ╳
 *  ╰╲─────────────────────────────────────────╱╯
 * Any work submitted after this barrier will need to wait for
 * the work represented by srcStageMask before it can execute.
 * For example, FRAGMENT_SHADER_BIT, then vertex shading could be executed
 * ealier.
 *
 *  ╭╱────────────────────╲╮
 *  ╳    Memory Barrier    ╳
 *  ╰╲────────────────────╱╯
 * Execution order and memory order are two different things
 * because of multiple & incoherent caches,
 * synchronizing execution alone is not enough to ensure the different units
 * on GPU can transfer data between themselves.
 *
 * GPU memory write is fistly "available", and only "visible" after cache
 * flushing That is where we should use a MemoryBarrier
 */
SE_EXPORT struct Barrier {
  /** virtual desctructor */
  virtual ~Barrier() = default;
};

// ╔══════════════════════════╗
// ║      Memory Barrier      ║
// ╚══════════════════════════╝
// Memory barrier is a structure specifying a global memory barrier
// A global memory barrier deals with access to any resource,
// and it’s the simplest form of a memory barrier.
//
// Description includes:
//  ► AccessFlags - srcAccessMask
//  ► AccessFlags - dstAccessMask

SE_EXPORT struct MemoryBarrier {
  /** virtual desctructor */
  virtual ~MemoryBarrier() = default;
};

// ╔════════════════════════════════╗
// ║     Buffer Memory Barrier      ║
// ╚════════════════════════════════╝
// It is quite similar to Memory Barrier
// Memory availability and visibility are restricted to a specific buffer.

SE_EXPORT struct BufferMemoryBarrier {
  /** virtual desctructor */
  virtual ~BufferMemoryBarrier() = default;
};

// ╔═══════════════════════════════╗
// ║     Image Memory Barrier      ║
// ╚═══════════════════════════════╝
// Beyond the memory barrier,
// Image Memory Barrier also take cares about layout change,
// The layout transition happens in-between the make available and make visible
// stages The layout transition itself is considered a read/write operation, the
// memory for image must be available before transition takes place, After a
// layout transition, the memory is automatically made available.
//
// Could think of the layout transition
// as some kind of in-place data munging which happens in L2 cache
//
// It can also be used to transfer queue family ownership when
// SHARING_MODE_EXCLUSIVE is used

SE_EXPORT struct ImageMemoryBarrier {
  /** virtual desctructor */
  virtual ~ImageMemoryBarrier() = default;
};

/**
 * < Access Flags >
 * Access Flags describe the access need for barrier.
 * In memory barrier, Access Flags are combined with Stage Flags.
 *
 * Warning: do not use AccessMask!=0 with TOP_OF_PIPE/BOTTOM_OF_PIPE
 * Because these stages do not perform memory accesses,
 * they are purely used for execution barriers
 */
SE_EXPORT using AccessFlags = uint32_t;
SE_EXPORT enum class AccessFlagBits : uint32_t {
  INDIRECT_COMMAND_READ_BIT = 0x00000001,
  INDEX_READ_BIT = 0x00000002,
  VERTEX_ATTRIBUTE_READ_BIT = 0x00000004,
  UNIFORM_READ_BIT = 0x00000008,
  INPUT_ATTACHMENT_READ_BIT = 0x00000010,
  SHADER_READ_BIT = 0x00000020,
  SHADER_WRITE_BIT = 0x00000040,
  COLOR_ATTACHMENT_READ_BIT = 0x00000080,
  COLOR_ATTACHMENT_WRITE_BIT = 0x00000100,
  DEPTH_STENCIL_ATTACHMENT_READ_BIT = 0x00000200,
  DEPTH_STENCIL_ATTACHMENT_WRITE_BIT = 0x00000400,
  TRANSFER_READ_BIT = 0x00000800,
  TRANSFER_WRITE_BIT = 0x00001000,
  HOST_READ_BIT = 0x00002000,
  HOST_WRITE_BIT = 0x00004000,
  MEMORY_READ_BIT = 0x00008000,
  MEMORY_WRITE_BIT = 0x00010000,
  TRANSFORM_FEEDBACK_WRITE_BIT = 0x02000000,
  TRANSFORM_FEEDBACK_COUNTER_READ_BIT = 0x04000000,
  TRANSFORM_FEEDBACK_COUNTER_WRITE_BIT = 0x08000000,
  CONDITIONAL_RENDERING_READ_BIT = 0x00100000,
  COLOR_ATTACHMENT_READ_NONCOHERENT_BIT = 0x00080000,
  ACCELERATION_STRUCTURE_READ_BIT = 0x00200000,
  ACCELERATION_STRUCTURE_WRITE_BIT = 0x00400000,
  FRAGMENT_DENSITY_MAP_READ_BIT = 0x01000000,
  FRAGMENT_SHADING_RATE_ATTACHMENT_READ_BIT = 0x00800000,
  COMMAND_PREPROCESS_READ_BIT = 0x00020000,
  COMMAND_PREPROCESS_WRITE_BIT = 0x00040000,
  NONE = 0,
};

SE_EXPORT struct MemoryBarrierDesc {
  // memory barrier mask
  AccessFlags srcAccessMask;
  AccessFlags dstAccessMask;
};

SE_EXPORT struct BufferMemoryBarrierDescriptor {
  // buffer memory barrier mask
  Buffer* buffer;
  AccessFlags srcAccessMask;
  AccessFlags dstAccessMask;
  uint64_t offset = 0;
  uint64_t size = uint64_t(-1); // default: the whole buffer
  // only if queue transition is need
  Queue* srcQueue = nullptr;
  Queue* dstQueue = nullptr;
};

SE_EXPORT struct TextureMemoryBarrierDescriptor {
  // specify image object
  Texture* texture;
  ImageSubresourceRange subresourceRange;
  // memory barrier mask
  AccessFlags srcAccessMask;
  AccessFlags dstAccessMask;
  // only if layout transition is need
  TextureLayout oldLayout;
  TextureLayout newLayout;
  // only if queue transition is need
  Queue* srcQueue = nullptr;
  Queue* dstQueue = nullptr;
};

/**
 * ╔═════════════════════════════╗
 * ║      BarrierDescriptor      ║
 * ╚═════════════════════════════╝
 *
 * < Pipeline Stages >
 * Pipeline Stage, is a sub-stage of a command.
 * They are used in the barrier command,
 * In a barrier it wait the "before" parts to complete and then execute the
 * "after" part However, using pipeline stages, we only need to wait certain
 * stages of the "before" part
 *
 * < Common Stages >
 * ┌───────────────────────────────┐ ┌─────────────────────────────┐
 * │      COMPUTE / TRANSFER       │ │      RENDER - Fragment      │
 * ├───────────────────────────────┤ ├─────────────────────────────┤
 * │  TOP_OF_PIPE				  │ │  EARLY_FRAGMENT_TESTS │ │  DRAW_INDIRECT				  │ │  FRAGMENT_SHADER		      │
 * │  COMPUTE / TRANSFER           │ │  LATE_FRAGMENT_TESTS        │
 * │  BOTTOM_OF_PIPE				  │ │  COLOR_ATTACHMENT_OUTPUT │
 * └───────────────────────────────┘ └─────────────────────────────┘
 * ┌───────────────────────────────────────────────────────────────┐
 * │                    RENDER - Geometry		     	          │
 * ├───────────────────────────────────────────────────────────────┤
 * │  DRAW_INDIRECT - Parses indirect buffers				      │
 * │  VERTEX_INPUT - Consumes fixed function VBOs and IBOs	      │
 * │  VERTEX_SHADER - Actual vertex shader │ │  TESSELLATION_CONTROL_SHADER
 * │ │  TESSELLATION_EVALUATION_SHADER
 * │ │  GEOMETRY_SHADER
 * │ └───────────────────────────────────────────────────────────────┘ <
 * Dependency Flags > Basically, we could use the NONE flag.
 */

/** dependency of barriers */
SE_EXPORT using DependencyTypeFlags = uint32_t;
/** dependency of barriers */
SE_EXPORT enum class DependencyType : uint32_t {
  NONE = 0 << 0,
  BY_REGION_BIT = 1 << 0,
  VIEW_LOCAL_BIT = 1 << 1,
  DEVICE_GROUP_BIT = 1 << 2,
};

SE_EXPORT struct BarrierDescriptor {
  // Necessary (Execution Barrier)
  PipelineStageFlags srcStageMask;
  PipelineStageFlags dstStageMask;
  DependencyTypeFlags dependencyType;
  // Optional (Memory Barriers)
  std::vector<MemoryBarrier*> memoryBarriers;
  std::vector<BufferMemoryBarrierDescriptor> bufferMemoryBarriers;
  std::vector<TextureMemoryBarrierDescriptor> textureMemoryBarriers;
};

/**
 * ╔═════════════════════╗
 * ║      Semaphore      ║
 * ╚═════════════════════╝
 * Semaphores are objects used introduce dependencies between operations,
 * it actually facilitate GPU <-> GPU synchronization,
 * such as waiting before acquiring the next image in the swapchain
 * before submitting command buffers to your device queue.
 *
 * To signal a semaphore, all previously submitted commands to the queue must
 * complete. We will also get a full memory barrier that all pending writes are
 * made available
 *
 * While signaling a semaphore makes all memory abailable
 * waiting for a semaphore makes memory visible.
 * Therefore, no extra barrier is need if we use a semaphore.
 *
 * Vulkan is unique in that semaphores are a part of the API,
 * with DirectX and Metal delegating that to OS calls.
 *
 * ╭──────────────┬───────────────────╮
 * │  Vulkan	  │   vk::Semaphore   │
 * │  DirectX 12  │   HANDLE          │
 * │  OpenGL      │   Varies by OS    │
 * ╰──────────────┴───────────────────╯
 */

SE_EXPORT struct Semaphore {};

// Synchronization Interface
// ===========================================================================
// Ray Tracing Interface

SE_EXPORT struct BLAS {
  /** virtual destructor */
  virtual ~BLAS() = default;
  /** get descriptor */
  virtual auto getDescriptor() noexcept -> BLASDescriptor = 0;
};

SE_EXPORT enum struct BLASGeometryFlagBits : uint32_t {
  NONE = 0 << 0,
  OPAQUE_GEOMETRY = 1 << 0,
  NO_DUPLICATE_ANY_HIT_INVOCATION = 1 << 1,
};
SE_EXPORT using BLASGeometryFlags = uint32_t;

/** Affline transform matrix */
SE_EXPORT struct AffineTransformMatrix {
  AffineTransformMatrix() = default;
  AffineTransformMatrix(Math::mat4 const& mat) {
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 4; ++j) matrix[i][j] = mat.data[i][j];
  }
  operator Math::mat4() const {
    Math::mat4 mat;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 4; ++j) mat.data[i][j] = matrix[i][j];
    mat.data[3][0] = 0;
    mat.data[3][1] = 0;
    mat.data[3][2] = 0;
    mat.data[3][3] = 1;
    return mat;
  }
  float matrix[3][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}};
};

SE_EXPORT struct BLASTriangleGeometry {
  RHI::Buffer* positionBuffer = nullptr;
  RHI::Buffer* indexBuffer = nullptr;
  RHI::Buffer* vertexBuffer = nullptr;
  IndexFormat indexFormat = IndexFormat::UINT16_t;
  uint32_t maxVertex = 0;
  uint32_t firstVertex = 0;
  uint32_t primitiveCount = 0;
  uint32_t primitiveOffset = 0;
  AffineTransformMatrix transform;
  BLASGeometryFlags geometryFlags = 0;
  uint32_t materialID = 0;
};

SE_EXPORT struct BLASCustomGeometry {
  AffineTransformMatrix transform;
  std::vector<Math::bounds3> aabbs;
  BLASGeometryFlags geometryFlags = 0;
};

SE_EXPORT struct BLASDescriptor {
  std::vector<BLASTriangleGeometry> triangleGeometries;
  bool allowRefitting = false;
  bool allowCompaction = false;
  std::vector<BLASCustomGeometry> customGeometries;
};

SE_EXPORT struct TLAS {
  /** virtual destructor */
  virtual ~TLAS() = default;
};

SE_EXPORT struct BLASInstance {
  BLAS* blas = nullptr;
  Math::mat4 transform = {};
  uint32_t instanceCustomIndex = 0;  // is used by system now
  uint32_t instanceShaderBindingTableRecordOffset = 0;
  uint32_t mask = 0xFF;
};

SE_EXPORT struct TLASDescriptor {
  std::vector<BLASInstance> instances;
  bool allowRefitting = false;
  bool allowCompaction = false;
};

SE_EXPORT struct RayTracingPipeline {
  /** virtual destructor */
  virtual ~RayTracingPipeline() = default;
  /** set debug name */
  virtual auto setName(std::string const& name) -> void = 0;
};

SE_EXPORT struct RayGenerationShaderBindingTableDescriptor {
  ShaderModule* rayGenShader = nullptr;
};
SE_EXPORT struct RayMissShaderBindingTableDescriptor {
  ShaderModule* rayMissShader = nullptr;
};
SE_EXPORT struct RayHitGroupShaderBindingTableDescriptor {
  struct HitGroupDescriptor {
    ShaderModule* closetHitShader = nullptr;
    ShaderModule* anyHitShader = nullptr;
    ShaderModule* intersectionShader = nullptr;
  };
};
SE_EXPORT struct CallableShaderBindingTableDescriptor {
  ShaderModule* callableShader = nullptr;
};

/**
 * Describe a set of SBTs that are stored consecutively.
 * When looking up a shader, different factors will tell the traversal engine
 * the index of the shader to call. These factors are:
 * - 1. missIndex from the GLSL traceRayEXT call;
 * - 2. The instance's instanceShaderBindingTableRecordOffset from TLAS
 * creation;
 * - 3. sbtRecordOffset from the GLSL traceRayEXT call;
 * - 4. sbtRecordStride from the GLSL traceRayEXT call;
 * - 5. The geometry index geometryIndex of each geometry inside a BLAS.
 */
SE_EXPORT struct SBTsDescriptor {
  /** @indexing: By default, traceRayEXT always uses the ray generation shader
   * at index 0. Therefore we currently support single record slot for a ray
   * generation SBT. */
  struct RayGenerationSBT {
    /** A ray generation record only has a ray generation shader. */
    struct RayGenerationRecord {
      ShaderModule* rayGenShader = nullptr;
    };
    /** As defaultly 0 is chosen, we only provide one record slot*/
    RayGenerationRecord rgenRecord = {};
  } rgenSBT;
  /** @indexing: When a ray didn't intersect anything, traversal calls
   * the index missIndex miss shader, specified in traceRayEXT call. */
  struct MissSBT {
    /** A ray miss record only has a miss shader. */
    struct MissRecord {
      ShaderModule* missShader = nullptr;
    };
    /** There could be multiple miss shader to be selected from */
    std::vector<MissRecord> rmissRecords = {};
  } missSBT;
  /** @indexing: Traversal calls the corresponding shader from the hit record
   * with index. instanceShaderBindingTableRecordOffset (from TLAS) +
   * sbtRecordOffset (from traceRayEXT call)
   * + sbtRecordStride (from traceRayEXT call)* geometryIndex (from BLAS) */
  struct HitGroupSBT {
    /** A hit group record includes a closest hit shader, an optional any hit
     * shader, and an optional intersection shader (only for procedural hit
     * groups). */
    struct HitGroupRecord {
      ShaderModule* closetHitShader = nullptr;
      ShaderModule* anyHitShader = nullptr;
      ShaderModule* intersectionShader = nullptr;
    };
    /** There could be hit group shader to be selected from */
    std::vector<HitGroupRecord> hitGroupRecords = {};
  } hitGroupSBT;
  struct CallableSBT {
    /** A callable record includes only a callable shader. */
    struct CallableRecord {
      ShaderModule* callableShader = nullptr;
    };
    /** There could be hit group shader to be selected from */
    std::vector<CallableRecord> callableRecords = {};
  } callableSBT;
};

SE_EXPORT struct RayTracingPipelineDescriptor : public PipelineDescriptorBase {
  uint32_t maxPipelineRayRecursionDepth = 1;
  SBTsDescriptor sbtsDescriptor = {};
};

SE_EXPORT struct RayTracingExtension {
  virtual ~RayTracingExtension() = default;
};
}  // namespace SIByL::RHI