module;
#include <string>
#include <memory>
#include <functional>
#include "../../../Application/Public/SE.Application.Config.h"
export module SE.GFX.RDG;
import SE.Utility;
import SE.Core.Resource;
import SE.RHI;
import SE.GFX.Core;

namespace SIByL::GFX 
{
	export enum struct RDGPassFlag {
		NONE			= 0 << 0,
		RASTER			= 1 << 0,
		COMPUTE			= 1 << 1,
		ASYNC_COMPUTE	= 1 << 2,
		RAY_TRACING		= 1 << 3,
	};

	/** Provide resources for pass execute phase */
	export struct RDGRegistry {


	};

	/** a custom pass consume */
	export using CustomPassConsumeFn = std::function<void()>;
	/** a custom pass execution */
	export using CustomPassExecuteFn = std::function<void(RDGRegistry const&, RHI::CommandEncoder*)>;
	/** a custom pass setup, should return a custom pass execution */
	export using CustomPassSetupFn = std::function<CustomPassExecuteFn()>;

	export struct RDGPassNode {
		/** pass name */
		char const* name;
		/** pass flags */
		RDGPassFlag flags;

		CustomPassConsumeFn	customPassCosume = nullptr;
		CustomPassSetupFn	customPassSetup = nullptr;
		CustomPassExecuteFn	customPassExec  = nullptr;
	};

	export enum struct ConsumeType {
		READ,
		WRITE,
		READ_WRITE,
		RENDER_TARGET,
		SAMPLE_SOURCE,
		SAMPLER,
	};

	export struct RDGResourceNode {
		RDGResourceNode() = default;
		RDGResourceNode(RDGResourceNode const&) = delete;
		virtual ~RDGResourceNode() = default;
		/** devirtualize resource */
		virtual auto devirtualize() noexcept -> void = 0;
		/** release resource */
		virtual auto release() noexcept -> void = 0;
		/** resource name */
		std::string name;
	};

	export struct RDGTexture :public RDGResourceNode {
		/** descriptor */
		struct Desc {
			RHI::Extend3D size;
			uint32_t mipLevelCount = 1;
			uint32_t sampleCount = 1;
			RHI::TextureDimension dimension = RHI::TextureDimension::TEX2D;
			RHI::TextureFormat format;
		};
		/** texture desc */
		RHI::TextureDescriptor descriptor = {};
		/** guid */
		Core::GUID guid = 0;
		/** devirtualized/external resource */
		GFX::Texture* texture = nullptr;
		/** is external */
		bool isExternal = false;
		/** devirtualize resource */
		virtual auto devirtualize() noexcept -> void override;
		/** release resource */
		virtual auto release() noexcept -> void override;
		/** consume resource */
		auto consume(ConsumeType type, RHI::TextureLayout layout, RHI::TextureUsage usage) noexcept -> void;
	};

	export struct RDGBuffer :public RDGResourceNode {
		/* buffer descriptor */
		RHI::BufferDescriptor descriptor = {};
		/** guid */
		Core::GUID guid = 0;
		/** devirtualized/external resource */
		GFX::Buffer* buffer = nullptr;
		/** is external */
		bool isExternal = false;
		/** devirtualize resource */
		virtual auto devirtualize() noexcept -> void override;
		/** release resource */
		virtual auto release() noexcept -> void override;
	};

	export struct RDGUniformBuffer :public RDGBuffer {
	};

	export template <class T>
	struct RDGStructuredUniformBuffer :public RDGUniformBuffer {
		/** update use structure */
		auto setStructure(T const& x, uint32_t idx) noexcept -> void;
		/** get buffer binding */
		auto getBufferBinding(uint32_t idx) noexcept -> RHI::BufferBinding;
	};

	export template <class T>
	struct RDGStructuredArrayUniformBuffer :public RDGUniformBuffer {
		/** update use structure */
		auto setStructure(T* x, uint32_t idx) noexcept -> void;
		/** get buffer binding */
		auto getBufferBinding(uint32_t idx) noexcept -> RHI::BufferBinding;
		/** array size */
		uint32_t size;
	};

	export struct RDGStorageBuffer :public RDGBuffer {
	};

	export struct RDGMultiStorageBuffer :public RDGStorageBuffer {
	};

	export template <class T>
	struct RDGStructuredArrayMultiStorageBuffer :public RDGMultiStorageBuffer {
		/** update use structure */
		auto setStructure(T* x, uint32_t idx) noexcept -> void;
		/** get buffer binding */
		auto getBufferBinding(uint32_t idx) noexcept -> RHI::BufferBinding;
		/** array size */
		uint32_t size;
	};

	export struct RDGraph {
		////////////////////////////////////
		//  Setup Phase
		// -------------------------------
		/** register external resource - texture */
		auto registerExternalTexture(char const* name, Core::GUID guid) noexcept -> RDGTexture*;
		/** create resource - texture */
		auto createTexture(char const* name, RDGTexture::Desc const& desc) noexcept -> RDGTexture*;
		/** create resource - structured uniform buffer */
		template <class T>
		auto createStructuredUniformBuffer(char const* name) noexcept -> RDGStructuredUniformBuffer<T>*;
		/** create resource - structured array uniform buffer */
		template <class T>
		auto createStructuredArrayUniformBuffer(char const* name, uint32_t size) noexcept -> RDGStructuredArrayUniformBuffer<T>*;
		/** create resource - structured array multi storage buffer */
		template <class T>
		auto createStructuredArrayMultiStorageBuffer(char const* name, uint32_t size) noexcept -> RDGStructuredArrayMultiStorageBuffer<T>*;
		/** get resource - texture */
		auto getTexture(char const* name) noexcept -> RDGTexture*;
		/** get resource - structured uniform buffer */
		template <class T>
		auto getStructuredUniformBuffer(char const* name) noexcept -> RDGStructuredUniformBuffer<T>*;
		/** get resource - structured array multi storage buffer */
		template <class T>
		auto getStructuredArrayMultiStorageBuffer(char const* name) noexcept -> RDGStructuredArrayMultiStorageBuffer<T>*;

		/** add a pass to the RDG */
		auto addPass(
			char const* pass_name, 
			RDGPassFlag flag, 
			CustomPassConsumeFn const& custom_consume, 
			CustomPassSetupFn const& custom_setup) noexcept -> RDGPassNode*;

		/** behavior to RDG compiling */
		using Behavior = std::function<void()>;
		enum struct BehaviorPhase {
			AfterDevirtualize_BeforePassSetup,
		};
		auto addBehavior(Behavior const& behavior, BehaviorPhase phase) noexcept -> void;
		////////////////////////////////////
		//  Compile Phase
		// -------------------------------
		/** compile */
		auto compile() noexcept -> void;
		auto sub_compile_devirtualize() noexcept -> void;
		auto sub_compile_pass_setup() noexcept -> void;
		////////////////////////////////////
		//  Excute Phase
		// -------------------------------
		/** execute */
		auto execute(RHI::CommandEncoder* cmdEncoder) noexcept -> void;
		////////////////////////////////////
		//  Others
		// -------------------------------
		/** clear */
		auto clear() noexcept -> void;
	private:
		/** passes map */
		std::unordered_map<std::string, std::unique_ptr<RDGPassNode>> passes;
		/** textures map */
		std::unordered_map<std::string, std::unique_ptr<RDGTexture>> textures;
		/** uniform buffer map */
		std::unordered_map<std::string, std::unique_ptr<RDGUniformBuffer>> uniformBuffers;
		/** storage buffer map */
		std::unordered_map<std::string, std::unique_ptr<RDGStorageBuffer>> storageBuffers;
		/** behaviors */
		std::unordered_map<BehaviorPhase, std::vector<Behavior>> behaviors;
	};

#pragma region RDGRAPH_IMPL

	auto RDGTexture::devirtualize() noexcept -> void {
		if (!isExternal) {
			guid = Core::ResourceManager::get()->requestRuntimeGUID<GFX::Texture>();
			GFX::GFXManager::get()->registerTextureResource(guid, descriptor);
			texture = Core::ResourceManager::get()->getResource<GFX::Texture>(guid);
			texture->texture->setName(name);
		}
	}

	auto RDGTexture::release() noexcept -> void {
		if (!isExternal) {
			Core::ResourceManager::get()->removeResource<GFX::Texture>(guid);
			texture = nullptr;
			guid = 0;
		}
	}

	auto RDGTexture::consume(ConsumeType type, RHI::TextureLayout layout, RHI::TextureUsage usage) noexcept -> void {
		descriptor.usage |= uint32_t(usage);
	}
	
	auto RDGBuffer::release() noexcept -> void {
		if (!isExternal) {
			Core::ResourceManager::get()->removeResource<GFX::Buffer>(guid);
			buffer = nullptr;
			guid = 0;
		}
	}

	template <class T>
	auto RDGStructuredUniformBuffer<T>::setStructure(T const& x, uint32_t idx) noexcept -> void {
		std::future<bool> mapped = buffer->buffer->mapAsync(0, 0, sizeof(T) * MULTIFRAME_FLIGHTS_COUNT);
		if (mapped.get()) {
			void* data = buffer->buffer->getMappedRange(sizeof(T) * idx, sizeof(T) * MULTIFRAME_FLIGHTS_COUNT);
			memcpy(data, &x, sizeof(T));
			buffer->buffer->unmap();
		}
	}

	auto RDGBuffer::devirtualize() noexcept -> void {
		if (!isExternal) {
			guid = Core::ResourceManager::get()->requestRuntimeGUID<GFX::Buffer>();
			GFX::GFXManager::get()->registerBufferResource(guid, descriptor);
			buffer = Core::ResourceManager::get()->getResource<GFX::Buffer>(guid);
			buffer->buffer->setName(name);
		}
	}

	template <class T>
	auto RDGStructuredUniformBuffer<T>::getBufferBinding(uint32_t idx) noexcept -> RHI::BufferBinding {
		return RHI::BufferBinding{ buffer->buffer.get(), idx * sizeof(T), sizeof(T) };
	}

	template <class T>
	auto RDGStructuredArrayUniformBuffer<T>::setStructure(T* x, uint32_t idx) noexcept -> void {
		std::future<bool> mapped = buffer->buffer->mapAsync(0, 0, sizeof(T) * MULTIFRAME_FLIGHTS_COUNT * size);
		if (mapped.get()) {
			void* data = buffer->buffer->getMappedRange(sizeof(T) * idx * size, sizeof(T) * MULTIFRAME_FLIGHTS_COUNT * size);
			memcpy(data, x, sizeof(T) * size);
			buffer->buffer->unmap();
		}
	}

	template <class T>
	auto RDGStructuredArrayUniformBuffer<T>::getBufferBinding(uint32_t idx) noexcept -> RHI::BufferBinding {
		return RHI::BufferBinding{ buffer->buffer.get(), idx * sizeof(T) * size, sizeof(T) * size };
	}

	template <class T>
	auto RDGStructuredArrayMultiStorageBuffer<T>::setStructure(T* x, uint32_t idx) noexcept -> void {
		std::future<bool> mapped = buffer->buffer->mapAsync(0, 0, sizeof(T) * MULTIFRAME_FLIGHTS_COUNT * size);
		if (mapped.get()) {
			void* data = buffer->buffer->getMappedRange(sizeof(T) * idx * size, sizeof(T) * MULTIFRAME_FLIGHTS_COUNT * size);
			memcpy(data, x, sizeof(T) * size);
			buffer->buffer->unmap();
		}
	}

	template <class T>
	auto RDGStructuredArrayMultiStorageBuffer<T>::getBufferBinding(uint32_t idx) noexcept -> RHI::BufferBinding {
		return RHI::BufferBinding{ buffer->buffer.get(), idx * sizeof(T) * size, sizeof(T) * size };
	}

	auto RDGraph::createTexture(char const* name, RDGTexture::Desc const& desc) noexcept -> RDGTexture* {
		textures[std::string(name)] = std::make_unique<RDGTexture>();
		RDGTexture* texture = textures[std::string(name)].get();
		texture->descriptor = RHI::TextureDescriptor{
			desc.size, desc.mipLevelCount, desc.sampleCount, desc.dimension, desc.format,
			(uint32_t)RHI::TextureUsage::TEXTURE_BINDING,
			{desc.format}
		};
		texture->name = name;
		return texture;
	}

	template <class T>
	auto RDGraph::createStructuredUniformBuffer(char const* name) noexcept -> RDGStructuredUniformBuffer<T>* {
		uniformBuffers[std::string(name)] = std::make_unique<RDGStructuredUniformBuffer<T>>();
		RDGStructuredUniformBuffer<T>* uniform_buffer = static_cast<RDGStructuredUniformBuffer<T>*>(uniformBuffers[std::string(name)].get());
		uniform_buffer->descriptor.size = sizeof(T) * MULTIFRAME_FLIGHTS_COUNT;
		uniform_buffer->descriptor.usage = (uint32_t)RHI::BufferUsage::UNIFORM;
		uniform_buffer->descriptor.memoryProperties = uint32_t(RHI::MemoryProperty::HOST_VISIBLE_BIT | RHI::MemoryProperty::HOST_COHERENT_BIT);
		uniform_buffer->name = std::string(name);
		return uniform_buffer;
	}

	template <class T>
	auto RDGraph::createStructuredArrayUniformBuffer(char const* name, uint32_t size) noexcept -> RDGStructuredArrayUniformBuffer<T>* {
		uniformBuffers[std::string(name)] = std::make_unique<RDGStructuredArrayUniformBuffer<T>>();
		RDGStructuredArrayUniformBuffer<T>* uniform_buffer = static_cast<RDGStructuredArrayUniformBuffer<T>*>(uniformBuffers[std::string(name)].get());
		uniform_buffer->descriptor.size = sizeof(T) * MULTIFRAME_FLIGHTS_COUNT * size;
		uniform_buffer->descriptor.usage = (uint32_t)RHI::BufferUsage::UNIFORM;
		uniform_buffer->descriptor.memoryProperties = uint32_t(RHI::MemoryProperty::HOST_VISIBLE_BIT | RHI::MemoryProperty::HOST_COHERENT_BIT);
		uniform_buffer->name = std::string(name);
		uniform_buffer->size = size;
		return uniform_buffer;
	}
	
	template <class T>
	auto RDGraph::createStructuredArrayMultiStorageBuffer(char const* name, uint32_t size) noexcept -> RDGStructuredArrayMultiStorageBuffer<T>* {
		storageBuffers[std::string(name)] = std::make_unique<RDGStructuredArrayMultiStorageBuffer<T>>();
		RDGStructuredArrayMultiStorageBuffer<T>* uniform_buffer = static_cast<RDGStructuredArrayMultiStorageBuffer<T>*>(storageBuffers[std::string(name)].get());
		uniform_buffer->descriptor.size = sizeof(T) * MULTIFRAME_FLIGHTS_COUNT * size;
		uniform_buffer->descriptor.usage = (uint32_t)RHI::BufferUsage::STORAGE;
		uniform_buffer->descriptor.memoryProperties = uint32_t(RHI::MemoryProperty::HOST_VISIBLE_BIT | RHI::MemoryProperty::HOST_COHERENT_BIT);
		uniform_buffer->name = std::string(name);
		uniform_buffer->size = size;
		return uniform_buffer;
	}

	auto RDGraph::addPass(
		char const* pass_name,
		RDGPassFlag flag,
		CustomPassConsumeFn const& custom_consume,
		CustomPassSetupFn const& custom_setup
	) noexcept -> RDGPassNode* {
		passes[std::string(pass_name)] = std::make_unique<RDGPassNode>();
		RDGPassNode* pass = passes[std::string(pass_name)].get();
		pass->customPassCosume = custom_consume;
		pass->customPassSetup = custom_setup;
		return pass;
	}

	auto RDGraph::getTexture(char const* name) noexcept -> RDGTexture* {
		auto iter = textures.find(std::string(name));
		if (iter != textures.end())
			return iter->second.get();
		else return nullptr;
	}

	template <class T>
	auto RDGraph::getStructuredUniformBuffer(char const* name) noexcept -> RDGStructuredUniformBuffer<T>* {
		auto iter = uniformBuffers.find(std::string(name));
		if (iter != uniformBuffers.end())
			return static_cast<RDGStructuredUniformBuffer<T>*>(iter->second.get());
		else return nullptr;
	}

	template <class T>
	auto RDGraph::getStructuredArrayMultiStorageBuffer(char const* name) noexcept -> RDGStructuredArrayMultiStorageBuffer<T>* {
		auto iter = storageBuffers.find(std::string(name));
		if (iter != storageBuffers.end())
			return static_cast<RDGStructuredArrayMultiStorageBuffer<T>*>(iter->second.get());
		else return nullptr;
	}

	auto RDGraph::addBehavior(Behavior const& behavior, BehaviorPhase phase) noexcept -> void {
		behaviors[phase].push_back(behavior);
	}

	auto RDGraph::compile() noexcept -> void {
		sub_compile_devirtualize();

		for (auto& behavior : behaviors[BehaviorPhase::AfterDevirtualize_BeforePassSetup])
			behavior();

		sub_compile_pass_setup();
	}

	auto RDGraph::sub_compile_devirtualize() noexcept -> void {
		// passes setup
		for (auto& iter : passes)
			iter.second->customPassCosume();
		// devirtualize all resources
		for (auto& iter : textures)
			iter.second->devirtualize();
		for (auto& iter : uniformBuffers)
			iter.second->devirtualize();
		for (auto& iter : storageBuffers)
			iter.second->devirtualize();
	}

	auto RDGraph::sub_compile_pass_setup() noexcept -> void {
		// initialize pass
		for (auto& iter : passes)
			iter.second->customPassExec = iter.second->customPassSetup();
	}

	auto RDGraph::registerExternalTexture(char const* name, Core::GUID guid) noexcept -> RDGTexture* {
		textures[std::string(name)] = std::make_unique<RDGTexture>();
		RDGTexture* texture = textures[std::string(name)].get();
		texture->descriptor = RHI::TextureDescriptor{
			{	texture->texture->texture->width(),
				texture->texture->texture->height(),
				texture->texture->texture->depthOrArrayLayers() },
			texture->texture->texture->mipLevelCount(),
			texture->texture->texture->sampleCount(),
			texture->texture->texture->dimension(),
			texture->texture->texture->format(),
			(uint32_t)RHI::TextureUsage::TEXTURE_BINDING,
			{texture->texture->texture->format()}
		};
		texture->texture = Core::ResourceManager::get()->getResource<GFX::Texture>(guid);
		texture->isExternal = true;
		texture->guid = guid;
		return texture;
	}

	auto RDGraph::execute(RHI::CommandEncoder* cmdEncoder) noexcept -> void {
		for (auto& iter : passes)
			iter.second->customPassExec({}, cmdEncoder);
	}

	auto RDGraph::clear() noexcept -> void {
		// clear passes
		passes.clear();
		// clear textures
		textures.clear();
	}

#pragma endregion
}