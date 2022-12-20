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
		/** consume resource */
		auto consume(ConsumeType type, RHI::TextureLayout layout, RHI::TextureUsage usage) noexcept -> void;
	};

	export struct RDGUniformBuffer :public RDGResourceNode {
		/* buffer descriptor */
		RHI::BufferDescriptor descriptor = {};
		/** guid */
		Core::GUID guid = 0;
		/** devirtualized/external resource */
		GFX::Buffer* buffer = nullptr;
		/** is external */
		bool isExternal = false;
	};

	export template <class T>
	struct RDGStructuredUniformBuffer :public RDGUniformBuffer {
		/** update use structure */
		auto setStructure(T const& x, uint32_t idx) noexcept -> void;
		/** devirtualize resource */
		virtual auto devirtualize() noexcept -> void override;
		/** get buffer binding */
		auto getBufferBinding(uint32_t idx) noexcept -> RHI::BufferBinding;
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
		/** get resource - texture */
		auto getTexture(char const* name) noexcept -> RDGTexture*;
		/** get resource - structured uniform buffer */
		template <class T>
		auto getStructuredUniformBuffer(char const* name) noexcept -> RDGStructuredUniformBuffer<T>*;

		/** add a pass to the RDG */
		auto addPass(
			char const* pass_name, 
			RDGPassFlag flag, 
			CustomPassConsumeFn const& custom_consume, 
			CustomPassSetupFn const& custom_setup) noexcept -> RDGPassNode*;
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

	auto RDGTexture::consume(ConsumeType type, RHI::TextureLayout layout, RHI::TextureUsage usage) noexcept -> void {
		descriptor.usage |= uint32_t(usage);
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

	template <class T>
	auto RDGStructuredUniformBuffer<T>::devirtualize() noexcept -> void {
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

	auto RDGraph::compile() noexcept -> void {
		sub_compile_devirtualize();
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

#pragma endregion
}