module;
#include <memory>
#include <future>
#include <vector>
#include <optional>
#include <variant>
export module RHI:Interface;
import Core.Memory;
import Platform.Window;

namespace SIByL::RHI
{
	// *************************|****************************************
	// Initialization			|   Initialization						|
	struct Context;
	struct Adapter;					struct RequestAdapterOptions;
	struct Device; 					struct DeviceDescriptor;
	// *************************|****************************************
	// Buffers					|	Buffers								|
	struct Buffer;					struct BufferDescriptor;
	// *************************|****************************************
	// Textures & Views			|   Textures & Views					|
	struct Texture;					struct TextureDescriptor;
	struct TextureView;				struct TextureViewDescriptor;
	struct ExternalTexture;			struct ExternalTextureDescriptor;
	// *************************|****************************************
	// Samplers					|   Samplers							|
	struct Sampler;					struct SamplerDescriptor;
	// *************************|****************************************
	// Pipeline					|   Pipeline							|
	struct ShaderModule;			struct ShaderModuleDescriptor;
	struct ComputePipeline;			struct ComputePipelineDescriptor;
	struct RenderPipeline;			struct RenderPipelineDescriptor;
	struct PipelineLayout;			struct PipelineLayoutDescriptor;
	// *************************|****************************************
	// Resource Binding			|   Resource Binding					|
	struct BindGroup;				struct BindGroupDescriptor;
	struct BindGroupLayout;			struct BindGroupLayoutDescriptor;
	// *************************|****************************************
	// Command Encoder			|   Command Encoder						|
	struct CommandEncoder;			struct CommandEncoderDescriptor;
	struct RenderBundleEncoder;		struct RenderBundleEncoderDescriptor;
	// *************************|****************************************
	// Queue					|   Queue								|
	struct Queue;				struct QueueDescriptor;
	// *************************|****************************************
	// Queries					|   Queries								|
	struct QuerySet;			struct QuerySetDescriptor;
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
	export using ContextExtensionsFlags = uint32_t;
	/** Context Extensions for extending API capability */
	export enum struct ContextExtension {
		NONE					= 0 << 0,
		DEBUG_UTILS				= 1 << 0,
		MESH_SHADER				= 1 << 1,
		FRAGMENT_BARYCENTRIC	= 1 << 2,
		SAMPLER_FILTER_MIN_MAX	= 1 << 3,
	};

	/** Context Interface for multiple-Graphics-API */
	export struct Context {
		/** Initialize the context */
		virtual auto init(Platform::Window* window = nullptr, ContextExtensionsFlags ext = 0) noexcept -> bool = 0;
		/** Request an adapter */
		virtual auto requestAdapter(RequestAdapterOptions const& options) noexcept -> std::unique_ptr<Adapter> = 0;
		/** Get the binded window */
		virtual auto getBindedWindow() const noexcept -> Platform::Window* = 0;
	};

	////////////////////////////////////
	//
	// Adapter
	//

	/** Describes the physical properties of a given GPU. */
	export struct Adapter {
		/** virtual destructor */
		virtual ~Adapter() = default;
		/** Requests a device from the adapter. */
		virtual auto requestDevice() noexcept -> std::unique_ptr<Device> = 0;
		/** Requests the AdapterInfo for this Adapter. */
		virtual auto requestAdapterInfo() const noexcept -> AdapterInfo = 0;
	};

	/** information of adapter */
	export struct AdapterInfo {
		std::string vendor;
		std::string architecture;
		std::string device;
		std::string description;
	};

	export enum struct PowerPreference {
		LOW_POWER,
		HIGH_PERFORMANCE,
	};

	export struct RequestAdapterOptions {
		PowerPreference powerPerference;
		bool forceFallbackAdapter = false;
	};

	////////////////////////////////////
	//
	// Device
	//

	/**
	* Device is a logical instantiation of an adapter, through which internal objects are created.
	* Is the exclusive owner of all internal objects created from it.
	*/
	export struct Device {
		/** virtual destructor */
		virtual ~Device() = default;
		/** destroy the device */
		virtual auto destroy() noexcept -> void = 0;
		// Create resources on device
		// ---------------------------
		/** create a buffer on the device */
		virtual auto createBuffer(BufferDescriptor const& desc) noexcept -> std::unique_ptr<Buffer> = 0;
		/** create a texture on the device */
		virtual auto createTexture(TextureDescriptor const& desc) noexcept -> std::unique_ptr<Texture> = 0;
		/** create a sampler on the device */
		virtual auto createSampler(SamplerDescriptor const& desc) noexcept -> std::unique_ptr<Sampler> = 0;
		/** create a external texture on the device */
		virtual auto importExternalTexture(ExternalTextureDescriptor const& desc) noexcept -> std::unique_ptr<ExternalTexture> = 0;
		// Create resources binding objects
		// ---------------------------
		/** create a bind group layout on the device */
		virtual auto createBindGroupLayout(BindGroupLayoutDescriptor const& desc) noexcept -> std::unique_ptr<BindGroupLayout> = 0;
		/** create a pipeline layout on the device */
		virtual auto createPipelineLayout(PipelineLayoutDescriptor const& desc) noexcept -> std::unique_ptr<PipelineLayout> = 0;
		/** create a bind group on the device */
		virtual auto createBindGroup(BindGroupDescriptor const& desc) noexcept -> std::unique_ptr<BindGroup> = 0;
		// Create pipeline objects
		// ---------------------------
		/** create a shader module on the device */
		virtual auto createShaderModule(ShaderModuleDescriptor const& desc) noexcept -> std::unique_ptr<ShaderModule> = 0;
		/** create a compute pipeline on the device */
		virtual auto createComputePipeline(ComputePipelineDescriptor const& desc) noexcept -> std::unique_ptr<ComputePipeline> = 0;
		/** create a render pipeline on the device */
		virtual auto createRenderPipeline(RenderPipelineDescriptor const& desc) noexcept -> std::unique_ptr<RenderPipeline> = 0;
		/** create a compute pipeline on the device in async way */
		virtual auto createComputePipelineAsync(ComputePipelineDescriptor const& desc) noexcept
			-> std::promise<std::unique_ptr<ComputePipeline>> = 0;
		/** create a render pipeline on the device in async way */
		virtual auto createRenderPipelineAsync(RenderPipelineDescriptor const& desc) noexcept
			-> std::promise<std::unique_ptr<RenderPipeline>> = 0;
		// Create command encoders
		// ---------------------------
		/** create a command encoder */
		virtual auto createCommandEncoder(CommandEncoderDescriptor const& desc) noexcept 
			-> std::unique_ptr<CommandEncoder> = 0;
		/** create a render bundle encoder */
		virtual auto createRenderBundleEncoder(CommandEncoderDescriptor const& desc) noexcept 
			-> std::unique_ptr<RenderBundleEncoder> = 0;
		// Create query sets
		// ---------------------------
		virtual auto createQuerySet(QuerySetDescriptor const& desc) noexcept -> std::unique_ptr<QuerySet> = 0;
	};

	struct DeviceDescriptor {

	};

	// Initialization Interface
	// ===========================================================================
	// Buffers Interface

	/** An object that holds a pointer (which can be null) to a buffer of a fixed number of bytes */
	export using ArrayBuffer = Core::Buffer;

	/** Determine how a GPUBuffer may be used after its creation. */
	export using BufferUsagesFlags = uint32_t;
	/** Determine how a GPUBuffer may be used after its creation. */
	export enum struct BufferUsage {
		MAP_READ		= 1 < 0,
		MAP_WRITE		= 1 < 1,
		COPY_SRC		= 1 < 2,
		COPY_DST		= 1 < 3,
		INDEX			= 1 < 4,
		VERTEX			= 1 < 5,
		UNIFORM			= 1 < 6,
		STORAGE			= 1 < 7,
		INDIRECT		= 1 < 8,
		QUERY_RESOLVE	= 1 < 9,
	};

	/** Determine the memory properties. */
	export using MemoryPropertiesFlags = uint32_t;
	/** Determine the memory properties. */
	export enum class MemoryProperty {
		DEVICE_LOCAL_BIT		= 1 < 0,
		HOST_VISIBLE_BIT		= 1 < 1,
		HOST_COHERENT_BIT		= 1 < 2,
		HOST_CACHED_BIT			= 1 < 3,
		LAZILY_ALLOCATED_BIT	= 1 < 4,
		PROTECTED_BIT			= 1 < 5,
		FLAG_BITS_MAX_ENUM		= 0x7FFFFFFF
	};

	/** Current map state of the buffer */
	export enum struct BufferMapState {
		UNMAPPED,
		PENDING,
		MAPPED
	};

	/** Shader mode of the buffer */
	export enum struct BufferShareMode {
		CONCURRENT,
		EXCLUSIVE,
	};

	/** Determine how a Buffer is mapped when calling mapAsync(). */
	export using MapModeFlags = uint32_t;
	/** Determine how a Buffer is mapped when calling mapAsync(). */
	export enum struct MapMode {
		READ = 1,
		WRITE = 2,
	};

	/**
	* A block of memory that can be used in GPU operations.
	* Data is stored in linear layout, and some can be mapped.
	*/
	export struct Buffer {
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
		// Map methods
		// ---------------------------
		/** Maps the given range of the GPUBuffer */
		virtual auto mapAsync(MapModeFlags mode, size_t offset = 0, size_t size = 0) noexcept -> std::promise<bool> = 0;
		/** Returns an ArrayBuffer with the contents of the GPUBuffer in the given mapped range */
		virtual auto getMappedRange(size_t offset = 0, size_t size = 0) noexcept -> ArrayBuffer = 0;
		/** Unmaps the mapped range of the GPUBuffer and makes it’s contents available for use by the GPU again. */
		virtual auto unmap() noexcept -> void = 0;
		// Lifecycle methods
		// ---------------------------
		/** destroy the buffer */
		virtual auto destroy() const noexcept -> void = 0;
	};

	export struct BufferDescriptor {
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
	};

	// Buffers Interface
	// ===========================================================================
	// Textures/TextureViews Interface

	export enum struct TextureDimension {
		TEX1D,
		TEX2D,
		TEX3D
	};

	export enum struct TextureFormat {
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
		RGB10A2_UNORM,
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
		// "depth32float-stencil8" feature
		DEPTH32STENCIL8,
	};

	/** Determine how a Texture may be used after its creation. */
	export using TextureUsagesFlags = uint32_t;
	/** Determine how a Texture may be used after its creation. */
	export enum struct TextureUsage {
		COPY_SRC = 1 < 0,
		COPY_DST = 1 < 1,
		TEXTURE_BINDING = 1 < 2,
		STORAGE_BINDING = 1 < 3,
		RENDER_ATTACHMENT = 1 < 4,
	};

	export struct Extend3D {
		uint32_t width;
		uint32_t height;
		uint32_t depthOrArrayLayers;
	};

	export struct Texture {
		// Texture Behaviors
		// ---------------------------
		/** virtual descructor */
		virtual ~Texture() = default;
		/** create texture view of this texture */
		virtual auto createView(TextureViewDescriptor const& desc) noexcept -> std::unique_ptr<TextureView> = 0;
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
		/** the dimension of the set of texel for each of this GPUTexture's subresources. */
		virtual auto dimension() const noexcept -> TextureDimension = 0;
		/** readonly format of the texture */
		virtual auto format() const noexcept -> TextureFormat = 0;
	};

	export struct TextureDescriptor {
		Extend3D size;
		uint32_t mipLevelCount;
		uint32_t sampleCount;
		TextureDimension dimension;
		TextureFormat format;
		/** The allowed usages for the texture. */
		TextureUsagesFlags usage;
		/**
		* Specifies what view format values will be allowed when calling createView()
		* on this texture (in addition to the texture’s actual format).
		*/
		std::vector<TextureFormat> viewFormats;
	};

	export enum struct TextureViewDimension {
		TEX1D,
		TEX2D,
		TEX2D_ARRAY,
		CUBE,
		CUBE_ARRAY,
		TEX3D,
	};

	export enum struct TextureAspect {
		ALL,
		STENCIL_ONLY,
		DEPTH_ONLY,
	};

	export struct TextureView {};

	export struct TextureViewDescriptor {
		TextureFormat format;
		TextureViewDimension dimension;
		TextureAspect aspect = TextureAspect::ALL;
		uint32_t baseMipLevel = 0;
		uint32_t mipLevelCount;
		uint32_t baseArrayLayer = 0;
		uint32_t arrayLayerCount;
	};

	/**
	* A sampleable texture wrapping an external video object.
	* The contents of a GPUExternalTexture object are a snapshot and may not change.
	*/
	export struct ExternalTexture {
		/** virtual destructor */
		virtual ~ExternalTexture();
		/** indicates whether the texture has expired or not */
		virtual auto expired() const noexcept -> bool = 0;
	};

	export struct ExternalTextureDescriptor {
	};

	// Textures/TextureViews Interface
	// ===========================================================================
	// Samplers Interface

	export struct Sampler {};

	export enum struct AddressMode {
		CLAMP_TO_EDGE,
		REPEAT,
		MIRROR_REPEAT,
	};

	export enum struct FilterMode {
		NEAREST,
		LINEAR,
	};

	export enum struct MipmapFilterMode {
		NEAREST,
		LINEAR,
	};

	export enum struct CompareFunction {
		NEVER,
		LESS,
		EQUAL,
		LESS_EQUAL,
		GREATER,
		NOT_EQUAL,
		GREATER_EQUAL,
		ALWAYS,
	};

	export struct SamplerDescriptor {
		AddressMode addressModeU = AddressMode::CLAMP_TO_EDGE;
		AddressMode addressModeV = AddressMode::CLAMP_TO_EDGE;
		AddressMode addressModeW = AddressMode::CLAMP_TO_EDGE;
		FilterMode magFilter = FilterMode::NEAREST;
		FilterMode minFilter = FilterMode::NEAREST;
		MipmapFilterMode mipmapFilter = MipmapFilterMode::NEAREST;
		float lodMinClamp = 0.f;
		float lodMapClamp = 32.f;
		CompareFunction comapre;
		uint16_t maxAnisotropy = 1;
	};

	// Samplers Interface
	// ===========================================================================
	// Resource Binding Interface

	/**
	* Defines the interface between a set of resources bound
	* in a BindGroup and their accessibility in shader stages.
	*/
	export struct BindGroupLayout {};

	/** Determine how a Texture may be used after its creation. */
	export using ShaderStagesFlags = uint32_t;
	/** Determine how a Texture may be used after its creation. */
	export enum struct ShaderStages {
		VERTEX = 1 < 0,
		FRAGMENT = 1 < 1,
		COMPUTE = 1 < 2,
	};

	export enum struct BindingResourceType {
		SAMPLER,
		TEXTURE_VIEW,
		BUFFER_BINDING,
		EXTERNAL_TEXTURE
	};

	export enum struct BufferBindingType {
		UNIFORM,
		STORAGE,
		READ_ONLY_STORAGE
	};

	export struct BufferBindingLayout {
		BufferBindingType type = BufferBindingType::UNIFORM;
		bool hasDynamicOffset = false;
		size_t minBindingSize = 0;
	};

	export enum struct SamplerBindingType {
		FILTERING,
		NON_FILTERING,
		COMPARISON,
	};

	export struct SamplerBindingLayout {
		SamplerBindingType type = SamplerBindingType::FILTERING;
	};

	export enum struct TextureSampleType {
		FLOAT,
		UNFILTERABLE_FLOAT,
		DEPTH,
		SINT,
		UINT,
	};

	export struct TextureBindingLayout {
		TextureSampleType sampleType = TextureSampleType::FLOAT;
		TextureViewDimension viewDimension = TextureViewDimension::TEX2D;
		bool multisampled = false;
	};

	export enum struct StorageTextureAccess {
		WIRTE_ONLY
	};

	export struct StorageTextureBindingLayout {
		StorageTextureAccess access = StorageTextureAccess::WIRTE_ONLY;
		TextureFormat format;
		TextureViewDimension viewDimension = TextureViewDimension::TEX2D;
	};

	export struct ExternalTextureBindingLayout {

	};

	/**
	* Describes a single shader resource binding 
	* to be included in a GPUBindGroupLayout.
	*/
	export struct BindGroupLayoutEntry {
		/* A unique identifier for a resource binding within the BindGroupLayout */
		uint32_t binding;
		/** indicates whether it will be accessible from the associated shader stage */
		ShaderStagesFlags visibility;
		// possible resources layout types
		// ---------------------------
		std::optional<BufferBindingLayout>			buffer;
		std::optional<SamplerBindingLayout>			sampler;
		std::optional<TextureBindingLayout>			texture;
		std::optional<StorageTextureBindingLayout>	storageTexture;
		std::optional<ExternalTextureBindingLayout> externalTexture;
	};

	export struct BindGroupLayoutDescriptor {
		std::vector<BindGroupLayoutEntry> entries;
	};
	

	/**
	* Defines a set of resources to be bound together in a group 
	* and how the resources are used in shader stages.
	*/
	export struct BindGroup {};
	
	export struct BufferBinding {
		Buffer* buffer;
		size_t  offset;
		size_t  size;
	};

	export struct BindingResource {
		BindingResourceType type;
		Sampler* sampler = nullptr;
		TextureView* textureView = nullptr;
		ExternalTexture* externalTexture = nullptr;
		std::optional<BufferBinding> bufferBinding;
	};

	export struct BindGroupEntry {
		uint32_t binding;
		BindingResource resource;
	};

	export struct BindGroupDescriptor {
		BindGroupLayout layout;
		std::vector<BindGroupEntry> entries;
	};

	/**
	* Defines the mapping between resources of all BindGroup objects set up
	* during command encoding in setBindGroup(), and the shaders of the pipeline
	* set by RenderCommandsMixin.setPipeline or ComputePassEncoder.setPipeline.
	*/
	export struct PipelineLayout {};

	export struct PipelineLayoutDescriptor {
		std::vector<BindGroupLayout> bindGroupLayouts;
	};

	// Resource Binding Interface
	// ===========================================================================
	// Shader Modules Interface

	/** An internal shader module object */
	export struct ShaderModule {};

	export struct ShaderModuleDescriptor {
		/** The shader source code for the shader module. */
		char const* code;
	};

	export struct ShaderModuleCompilationHint {
		/** 
		* if the layout is not given, reflection will 
		* be used to automatically generate layout 
		*/
		std::optional<PipelineLayout> layout;
	};

	// Shader Modules Interface
	// ===========================================================================
	// Pipelines Interface

	export struct PipelineBase {
		/** Gets a BindGroupLayout that is compatible with the PipelineBase's BindGroupLayout at index.*/
		auto getBindGroupLayout(size_t index) noexcept -> BindGroupLayout&;
	};
	
	export using PipelineConstantValue = double;

	/**
	* Describes the entry point in the user-provided ShaderModule that 
	* controls one of the programmable stages of a pipeline.
	*/
	export struct ProgrammableStage {
		ShaderModule module;
		std::string entryPoint;
	};

	export struct PipelineDescriptorBase {
		/** The definition of the layout of resources which can be used with this */
		std::optional<PipelineLayout> layout;
	};
	
	export struct ComputePipeline {};

	export struct ComputePipelineDescriptor {
		ProgrammableStage compute;
	};

	export enum struct PrimitiveTopology {
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
	export enum struct FrontFace {
		CCW, // counter-clockwise
		CW,	// clockwise
	};

	export enum struct CullMode {
		NONE,
		FRONT,
		BACK,
	};

	export using SampleMask = uint32_t;

	export struct MultisampleState {
		uint32_t count = 1;
		SampleMask mask = 0xFFFFFFFF;
		bool alphaToCoverageEnabled = false;
	};

	export enum struct BlendFactor {
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
	
	export enum struct BlendOperation {
		ADD,
		SUBTRACT,
		REVERSE_SUBTRACT,
		MIN,
		MAX,
	};

	export struct BlendComponent {
		BlendOperation operation = BlendOperation::ADD;
		BlendFactor	   srcFactor = BlendFactor::ONE;
		BlendFactor    dstFactor = BlendFactor::ZERO;
	};

	export struct BlendState {
		BlendComponent color;
		BlendComponent alpha;
	};

	export using ColorWriteFlags = uint32_t;

	export struct ColorTargetState {
		TextureFormat format;
		BlendState blend;
		ColorWriteFlags writeMask = 0xF;
	};

	export struct FragmentState {
		std::vector<ColorTargetState> targets;
	};

	export using StencilValue = uint32_t;
	export using DepthBias	  = int32_t;

	export enum struct StencilOperation {
		KEEP,
		ZERO,
		REPLACE,
		INVERT,
		INCREMENT_CLAMP,
		DECREMENT_CLAMP,
		INCREMENT_WARP,
		DECREMENT_WARP,
	};

	export enum struct VertexFormat {
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

	export enum struct VertexStepMode {
		VERTEX,
		INSTANCE,
	};

	export enum struct IndexFormat {
		UINT16_t,
		UINT32_T
	};

	export struct PrimitiveState {
		PrimitiveTopology topology = PrimitiveTopology::TRIANGLE_LIST;
		IndexFormat stripIndexFormat;
		FrontFace frontFace = FrontFace::CCW;
		CullMode cullMode = CullMode::NONE;
		bool unclippedDepth = false;
	};

	export struct StencilFaceState {
		CompareFunction  compare = CompareFunction::ALWAYS;
		StencilOperation failOp = StencilOperation::KEEP;
		StencilOperation depthFailOp = StencilOperation::KEEP;
		StencilOperation passOp = StencilOperation::KEEP;
	};

	export struct DepthStencilState {
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

	export struct VertexAttribute {
		VertexFormat format;
		size_t offset;
		uint32_t shaderLocation;
	};

	export struct VertexBufferLayout {
		size_t arrayStride;
		VertexStepMode stepMode = VertexStepMode::VERTEX;
		std::vector<VertexAttribute> attributes;
	};

	export struct VertexState {
		std::vector<VertexBufferLayout> buffers;
	};

	export struct RenderPipeline {};

	/** Describes a render pipeline by configuring each of the render stages. */
	export struct RenderPipelineDescriptor :public PipelineDescriptorBase {
		VertexState vertex;
		PrimitiveState primitive = {};
		DepthStencilState depthStencil;
		MultisampleState multisample = {};
		FragmentState fragment;
	};

	// Pipelines Interface
	// ===========================================================================
	// Command Buffers Interface

	/**
	* Pre-recorded lists of GPU commands that can be submitted to a GPUQueue 
	* for execution. Each GPU command represents a task to be performed on the 
	* GPU, such as setting state, drawing, copying resources, etc.
	* 
	* A CommandBuffer can only be submitted once, at which point it becomes invalid. 
	* To reuse rendering commands across multiple submissions, use RenderBundle.
	*/
	export struct CommandBuffer {};

	export struct CommandBufferDescriptor {};

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
	export struct CommandsMixin {};

	export struct CommandEncoder {

		auto beginRenderPass(RenderPassDescriptor const& desc) noexcept -> RenderPassEncoder;

		auto beginComputePass(ComputePassDescriptor const& desc) noexcept -> ComputePassEncoder;

		auto copyBufferToBuffer(
			Buffer* source,
			size_t	sourceOffset,
			Buffer* destination,
			size_t	destinationOffset,
			size_t	size) noexcept -> void;

		auto copyBufferToTexture(
			ImageCopyBuffer  const& source,
			ImageCopyTexture const& destination,
			Extend3D		 const& copySize) noexcept -> void;

		auto copyTextureToBuffer(
			ImageCopyTexture const& source,
			ImageCopyBuffer  const& destination,
			Extend3D		 const& copySize) noexcept -> void;

		auto copyTextureToTexture(
			ImageCopyTexture const& source,
			ImageCopyTexture const& destination,
			Extend3D		 const& copySize) noexcept -> void;

		auto clearBuffer(
			Buffer& buffer,
			size_t offset = 0,
			size_t size = 0) noexcept -> void;

		auto writeTimestamp(
			QuerySet querySet, 
			uint32_t queryIndex) noexcept -> void;

		auto resolveQuerySet(
			QuerySet querySet,
			uint32_t firstQuery,
			uint32_t queryCount,
			Buffer&  destination,
			uint64_t destinationOffset) noexcept -> void;

		auto finish(std::optional<CommandBufferDescriptor> const& descriptor = {}) noexcept -> CommandBuffer;
	};

	export struct CommandEncoderDescriptor {

	};

	export struct ImageDataLayout {
		uint64_t offset = 0;
		uint32_t bytesPerRow;
		uint32_t rowsPerImage;
	};

	export struct ImageCopyBuffer: ImageDataLayout {};
	
	export using IntegerCoordinate = uint32_t;

	export struct Origin2DDict {
		IntegerCoordinate x = 0;
		IntegerCoordinate y = 0;
	};
	export using Origin2D = std::variant<Origin2DDict, std::vector<IntegerCoordinate>>;

	export struct Origin3DDict {
		IntegerCoordinate x = 0;
		IntegerCoordinate y = 0;
		IntegerCoordinate z = 0;
	};

	export using Origin3D = std::variant<Origin3DDict, std::vector<IntegerCoordinate>>;
	
	export struct ImageCopyTexture {
		Texture* texutre;
		uint32_t mipLevel = 0;
		Origin3D origin = {};
		TextureAspect aspect = TextureAspect::ALL;
	};

	export struct ImageCopyTextureTagged :public ImageCopyTexture {

		bool premultipliedAlpha = false;
	};

	export struct ImageCopyExternalImage {
		Origin2D origin = {};
		bool flipY = false;
	};

	export struct Color {

	};

	export struct RenderPassEncoder {
		auto setViewport(
			float x, float y,
			float width, float height,
			float minDepth, float maxDepth) noexcept -> void;

		auto setScissorRect(
			IntegerCoordinate x, IntegerCoordinate y,
			IntegerCoordinate width, IntegerCoordinate height) noexcept -> void;

		auto setBlendConstant(Color color) noexcept -> void;

		auto setStencilReference(StencilValue reference) noexcept -> void;

		auto beginOcclusionQuery(uint32_t queryIndex) noexcept -> void;
		auto endOcclusionQuery() noexcept -> void;

		auto executeBundles(std::vector<RenderBundle> const& bundles) noexcept -> void;

		auto end() noexcept -> void;
	};

	export struct RenderPassDescriptor {

	};
	
	export struct ComputePassEncoder {

	};

	export struct ComputePassDescriptor {

	};

	// Command Encoding Interface
	// ===========================================================================
	// Programmable Passes Interface

	// Programmable Passes Interface
	// ===========================================================================
	// Debug Marks Interface
	

	// Debug Marks Interface
	// ===========================================================================
	// Compute Passes Interface
	

	// Compute Passes Interface
	// ===========================================================================
	// Render Passes Interface
	
	export struct RenderPassLayout {
		std::vector<TextureFormat> colorFormats;
		TextureFormat depthStencilFormat;
		uint32_t sampleCount = 1;
	};

	export struct RenderCommandsMimin {
		auto setPipeline(RenderPipeline const& pipeline) noexcept -> void;

		auto setIndexBuffer(Buffer* buffer, IndexFormat indexFormat, 
			uint64_t offset = 0, uint64_t size = 0) noexcept -> void;

		auto setVertexBuffer(uint32_t slot, Buffer* buffer, 
			uint64_t offset = 0, uint64_t size = 0) noexcept -> void;

		auto draw(uint32_t vertexCount, uint32_t instanceCount = 1,
			uint32_t firstVertex = 0, uint32_t firstInstance = 0) noexcept -> void;

		auto drawIndexed(uint32_t indexCount, uint32_t instanceCount = 1,
			uint32_t firstIndex = 0,
			int32_t  baseVertex = 0,
			uint32_t firstInstance = 0) noexcept -> void;

		auto drawIndirect(Buffer* indirectBuffer, uint64_t indirectOffset) noexcept -> void;
		auto drawIndexedIndirect(Buffer* indirectBuffer, uint64_t indirectOffset) noexcept -> void;
	};

	// Render Passes Interface
	// ===========================================================================
	// Bundles Interface

	export struct RenderBundle {};

	export struct RenderBundleDescriptor {};

	export struct RenderBundleEncoder {
		auto finish(std::optional<RenderBundleDescriptor> const& descriptor = {}) noexcept -> RenderBundle;
	};

	export struct RenderBundleEncoderDescriptor :public RenderPassLayout {
		bool depthReadOnly	 = false;
		bool stencilReadOnly = false;
	};

	// Bundles Interface
	// ===========================================================================
	// Queue Interface

	export struct Queue {
		auto submit(std::vector<CommandBuffer&> const& commandBuffers) noexcept -> void;

		auto onSubmittedWorkDone() noexcept -> std::promise<bool>;

		auto writeBuffer(
			Buffer* buffer,
			uint64_t bufferOffset,
			ArrayBuffer* data,
			uint64_t dataOffset,
			Extend3D const& size) noexcept -> void;

		auto writeTexture(
			ImageCopyTexture const& destination,
			ArrayBuffer* data,
			ImageDataLayout const& layout,
			Extend3D const& size) noexcept -> void;

		auto copyExternalImageToTexture(
			ImageCopyExternalImage const& source,
			ImageCopyExternalImage const& destination,
			Extend3D const& copySize) noexcept -> void;
	};

	/** Describes a queue request */
	export struct QueueDescriptor {};

	// Queue Interface
	// ===========================================================================
	// Queries Interface

	export enum struct QueryType {
		OCCLUSION,
		TIMESTAMP
	};

	export struct QuerySet {
		virtual auto destroy() noexcept -> void = 0;
		QueryType type;
		uint32_t count;
	};

	export struct QuerySetDescriptor {
		QueryType type;
		uint32_t  count;
	};
}