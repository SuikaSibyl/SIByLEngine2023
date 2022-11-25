module;
#include <typeinfo>
#include <memory>
#include <vector>
#include <set>
#include <unordered_map>
#include <filesystem>
export module GFX.GFXManager;
import Core.System;
import Core.ECS;
import Core.Memory;
import Core.IO;
import Core.Resource;
import RHI;
import RHI.RHILayer;
import Image.Color;
import Image.Image;
import GFX.Resource;
import GFX.Components;

namespace SIByL::GFX
{
	export struct GFXManager :public Core::Manager {
		/* get singleton */
		static auto get() noexcept -> GFXManager* { return singleton; }
		/** start up the GFX manager */
		virtual auto startUp() noexcept -> void override {
			// set singleton
			singleton = this;
			// register component types
			Core::ComponentManager::get()->registerComponent<GFX::TagComponent>();
			Core::ComponentManager::get()->registerComponent<GFX::TransformComponent>();
			Core::ComponentManager::get()->registerComponent<GFX::MeshReference>();
			Core::ComponentManager::get()->registerComponent<GFX::MeshRenderer>();
			// register resource types
			Core::ResourceManager::get()->registerResource<GFX::Buffer>();
			Core::ResourceManager::get()->registerResource<GFX::Mesh>();
			Core::ResourceManager::get()->registerResource<GFX::Texture>();
			Core::ResourceManager::get()->registerResource<GFX::Sampler>();
			Core::ResourceManager::get()->registerResource<GFX::ShaderModule>();
			Core::ResourceManager::get()->registerResource<GFX::Material>();
			Core::ResourceManager::get()->registerResource<GFX::Scene>();
			Core::ResourceManager::get()->registerResource<GFX::ASGroup>();
		}
		/** shut down the GFX manager */
		virtual auto shutDown() noexcept -> void override {}
		/** register GFX resources */
		auto registerBufferResource(Core::GUID guid, RHI::BufferDescriptor const& desc) noexcept -> void;
		auto registerMeshResource(Core::GUID guid, Core::ORID orid) noexcept -> void;
		auto registerTextureResource(Core::GUID guid, Image::Image<Image::COLOR_R8G8B8A8_UINT>* image) noexcept -> void;
		auto registerTextureResource(Core::GUID guid, RHI::TextureDescriptor const& desc) noexcept -> void;
		auto registerSamplerResource(Core::GUID guid, RHI::SamplerDescriptor const& desc) noexcept -> void;
		auto registerShaderModuleResource(Core::GUID guid, RHI::ShaderModuleDescriptor const& desc) noexcept -> void;
		auto registerShaderModuleResource(Core::GUID guid, std::filesystem::path const& path, RHI::ShaderModuleDescriptor const& desc) noexcept -> void;
		auto registerAsGroupResource(Core::GUID guid, RHI::TLASDescriptor const& desc, std::vector<Core::GUID> const& meshes) noexcept -> void;
		/** RHI layer */
		RHI::RHILayer* rhiLayer = nullptr;
		/** common samplers */
		struct CommonSampler {
			Core::GUID defaultSampler;
		} commonSampler;
	private:
		/** singleton */
		static GFXManager* singleton;
	};

#pragma region GFX_MANAGER_IMPL

	GFXManager* GFXManager::singleton = nullptr;
	
	auto GFXManager::registerBufferResource(Core::GUID guid, RHI::BufferDescriptor const& desc) noexcept -> void {
		GFX::Buffer bufferResource = {};
		bufferResource.buffer = rhiLayer->getDevice()->createBuffer(desc);
		Core::ResourceManager::get()->addResource(guid, std::move(bufferResource));
	}

	auto GFXManager::registerTextureResource(Core::GUID guid, Image::Image<Image::COLOR_R8G8B8A8_UINT>* image) noexcept -> void {
		GFX::Texture textureResource = {};
		RHI::BufferDescriptor stagingBufferDescriptor;
		stagingBufferDescriptor.size = image->data.size;
		stagingBufferDescriptor.usage = (uint32_t)RHI::BufferUsage::COPY_SRC;
		stagingBufferDescriptor.memoryProperties = (uint32_t)RHI::MemoryProperty::HOST_VISIBLE_BIT
			| (uint32_t)RHI::MemoryProperty::HOST_COHERENT_BIT;
		stagingBufferDescriptor.mappedAtCreation = true;
		std::unique_ptr<RHI::Buffer> stagingBuffer = rhiLayer->getDevice()->createBuffer(stagingBufferDescriptor);
		std::future<bool> mapped = stagingBuffer->mapAsync(0, 0, stagingBufferDescriptor.size);
		if (mapped.get()) {
			void* mapdata = stagingBuffer->getMappedRange(0, stagingBufferDescriptor.size);
			memcpy(mapdata, image->data.data, (size_t)stagingBufferDescriptor.size);
			stagingBuffer->unmap();
		}
		std::unique_ptr<RHI::CommandEncoder> commandEncoder = rhiLayer->getDevice()->createCommandEncoder({ nullptr });
		// create texture image
		textureResource.texture = rhiLayer->getDevice()->createTexture(RHI::TextureDescriptor{
			{(uint32_t)image->width,(uint32_t)image->height, 1},
			1,1,RHI::TextureDimension::TEX2D,
			RHI::TextureFormat::RGBA8_UNORM,
			(uint32_t)RHI::TextureUsage::COPY_DST | (uint32_t)RHI::TextureUsage::TEXTURE_BINDING,
			{ RHI::TextureFormat::RGBA8_UNORM }
			});

		commandEncoder->pipelineBarrier(RHI::BarrierDescriptor{
			(uint32_t)RHI::PipelineStages::TOP_OF_PIPE_BIT,
			(uint32_t)RHI::PipelineStages::TRANSFER_BIT,
			(uint32_t)RHI::DependencyType::NONE,
			{}, {},
			{ RHI::TextureMemoryBarrierDescriptor{
				textureResource.texture.get(), RHI::ImageSubresourceRange{(uint32_t)RHI::TextureAspect::COLOR_BIT, 0,1,0,1},
				(uint32_t)RHI::AccessFlagBits::NONE,
				(uint32_t)RHI::AccessFlagBits::TRANSFER_WRITE_BIT,
				RHI::TextureLayout::UNDEFINED,
				RHI::TextureLayout::TRANSFER_DST_OPTIMAL
			}}
			});

		commandEncoder->copyBufferToTexture(
			{ 0, 0, 0, stagingBuffer.get() },
			{ textureResource.texture.get(), 0, {}, (uint32_t)RHI::TextureAspect::COLOR_BIT },
			{ textureResource.texture->width(), textureResource.texture->height(), 1 });

		commandEncoder->pipelineBarrier(RHI::BarrierDescriptor{
			(uint32_t)RHI::PipelineStages::TRANSFER_BIT,
			(uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT,
			(uint32_t)RHI::DependencyType::NONE,
			{}, {},
			{ RHI::TextureMemoryBarrierDescriptor{
				textureResource.texture.get(), RHI::ImageSubresourceRange{(uint32_t)RHI::TextureAspect::COLOR_BIT, 0,1,0,1},
				(uint32_t)RHI::AccessFlagBits::TRANSFER_WRITE_BIT,
				(uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT,
				RHI::TextureLayout::TRANSFER_DST_OPTIMAL,
				RHI::TextureLayout::SHADER_READ_ONLY_OPTIMAL
			}}
			});

		rhiLayer->getDevice()->getGraphicsQueue()->submit({ commandEncoder->finish({}) });
		rhiLayer->getDevice()->getGraphicsQueue()->waitIdle();
		textureResource.originalView = textureResource.texture->createView(RHI::TextureViewDescriptor{
			RHI::TextureFormat::RGBA8_UNORM });
		Core::ResourceManager::get()->addResource(guid, std::move(textureResource));
	}

	auto GFXManager::registerMeshResource(Core::GUID guid, Core::ORID orid) noexcept -> void {
		GFX::Mesh mesh;
		Core::ResourceManager::get()->addResource(guid, std::move(mesh));
		Core::ResourceManager::get()->getResource<GFX::Mesh>(guid)->deserialize(rhiLayer->getDevice(), orid);
	}

	auto GFXManager::registerTextureResource(Core::GUID guid, RHI::TextureDescriptor const& desc) noexcept -> void {
		GFX::Texture textureResource = {};
		// create texture image
		textureResource.texture = rhiLayer->getDevice()->createTexture(desc);
		// transition layout
		RHI::TextureAspectFlags aspectMask = 0;
		RHI::TextureLayout targetLayout = {};
		RHI::AccessFlags targetAccessFlags = {};
		if (desc.usage & (uint32_t)RHI::TextureUsage::COLOR_ATTACHMENT) {
			aspectMask |= (uint32_t)RHI::TextureAspect::COLOR_BIT;
			targetLayout = RHI::TextureLayout::SHADER_READ_ONLY_OPTIMAL;
			targetAccessFlags = (uint32_t)RHI::AccessFlagBits::COLOR_ATTACHMENT_READ_BIT
				| (uint32_t)RHI::AccessFlagBits::COLOR_ATTACHMENT_WRITE_BIT;
		}
		else if (desc.usage & (uint32_t)RHI::TextureUsage::DEPTH_ATTACHMENT) {
			aspectMask |= (uint32_t)RHI::TextureAspect::DEPTH_BIT;
			targetLayout = RHI::TextureLayout::SHADER_READ_ONLY_OPTIMAL;
			targetAccessFlags = (uint32_t)RHI::AccessFlagBits::DEPTH_STENCIL_ATTACHMENT_READ_BIT
				| (uint32_t)RHI::AccessFlagBits::DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
		}
		else if (desc.usage & (uint32_t)RHI::TextureUsage::TEXTURE_BINDING) {
			aspectMask |= (uint32_t)RHI::TextureAspect::COLOR_BIT;
			targetLayout = RHI::TextureLayout::SHADER_READ_ONLY_OPTIMAL;
			targetAccessFlags = (uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT;
		}
		else if (desc.usage & (uint32_t)RHI::TextureUsage::COPY_DST) {
			aspectMask |= (uint32_t)RHI::TextureAspect::COLOR_BIT;
			targetLayout = RHI::TextureLayout::TRANSFER_DST_OPTIMAL;
			targetAccessFlags = (uint32_t)RHI::AccessFlagBits::TRANSFER_WRITE_BIT;
		}
		// do transition commands
		std::unique_ptr<RHI::CommandEncoder> commandEncoder = rhiLayer->getDevice()->createCommandEncoder({ nullptr });
		commandEncoder->pipelineBarrier(RHI::BarrierDescriptor{
			(uint32_t)RHI::PipelineStages::TOP_OF_PIPE_BIT,
			(uint32_t)RHI::PipelineStages::TRANSFER_BIT,
			(uint32_t)RHI::DependencyType::NONE,
			{}, {},
			{ RHI::TextureMemoryBarrierDescriptor{
				textureResource.texture.get(), RHI::ImageSubresourceRange{aspectMask, 0,1,0,1},
				(uint32_t)RHI::AccessFlagBits::NONE,
				targetAccessFlags,
				RHI::TextureLayout::UNDEFINED,
				targetLayout
			}}
			});
		rhiLayer->getDevice()->getGraphicsQueue()->submit({ commandEncoder->finish({}) });
		rhiLayer->getDevice()->getGraphicsQueue()->waitIdle();
		RHI::TextureViewDescriptor viewDesc = { desc.format };
		viewDesc.aspect = RHI::getTextureAspect(desc.format);
		if (!desc.hostVisible) // if host visible we do not create view
			textureResource.originalView = textureResource.texture->createView(viewDesc);
		Core::ResourceManager::get()->addResource(guid, std::move(textureResource));
	}

	auto GFXManager::registerSamplerResource(Core::GUID guid, RHI::SamplerDescriptor const& desc) noexcept -> void {
		GFX::Sampler samplerResource = {};
		samplerResource.sampler = rhiLayer->getDevice()->createSampler(desc);
		Core::ResourceManager::get()->addResource(guid, std::move(samplerResource));
	}

	auto GFXManager::registerShaderModuleResource(Core::GUID guid, RHI::ShaderModuleDescriptor const& desc) noexcept -> void {
		GFX::ShaderModule shaderModuleResource = {};
		shaderModuleResource.shaderModule = rhiLayer->getDevice()->createShaderModule(desc);
		Core::ResourceManager::get()->addResource(guid, std::move(shaderModuleResource));
	}

	auto GFXManager::registerShaderModuleResource(Core::GUID guid, std::filesystem::path const& path, RHI::ShaderModuleDescriptor const& desc) noexcept -> void {
		RHI::ShaderModuleDescriptor smDesc = desc;
		Core::Buffer buffer;
		Core::syncReadFile(path, buffer);
		smDesc.code = &buffer;
		GFX::ShaderModule shaderModuleResource = {};
		shaderModuleResource.shaderModule = rhiLayer->getDevice()->createShaderModule(smDesc);
		Core::ResourceManager::get()->addResource(guid, std::move(shaderModuleResource));
	}

	auto GFXManager::registerAsGroupResource(Core::GUID guid, RHI::TLASDescriptor const& desc, std::vector<Core::GUID> const& meshes) noexcept -> void {
		GFX::ASGroup asGroup = {};
		asGroup.tlas = rhiLayer->getDevice()->createTLAS(desc);
		std::unordered_map<Core::GUID, uint32_t> map;
		uint32_t vertexOffset = 0;
		uint32_t indexOffset = 0;
		std::vector<Core::GUID> meshesToBind = {};
		uint32_t vertexCount = 0;
		uint32_t indexCount = 0;
		for (auto& meshid : meshes) {
			if (map.find(meshid) != map.end()) {
				asGroup.geometryInfo.emplace_back(asGroup.geometryInfo[map[meshid]]);
			}
			GFX::Mesh* mesh = Core::ResourceManager::get()->getResource<GFX::Mesh>(meshid);
			meshesToBind.emplace_back(meshid);
			asGroup.geometryInfo.emplace_back(ASGroup::GeometryInfo{
				uint32_t(vertexOffset / sizeof(float)),
				uint32_t(indexOffset / sizeof(uint16_t)) });
			map[meshid] = asGroup.geometryInfo.size() - 1;
			vertexOffset += mesh->vertexBufferPosOnly->size();
			indexOffset += mesh->indexBuffer->size();
		}
		std::vector<float> vertexBuffer(vertexOffset / sizeof(float));
		std::vector<uint16_t> indexBuffer(indexOffset / sizeof(uint16_t));
		vertexOffset = 0;
		indexOffset = 0;
		for (auto& meshid : meshesToBind) {
			GFX::Mesh* mesh = Core::ResourceManager::get()->getResource<GFX::Mesh>(meshid);
			rhiLayer->getDevice()->readbackDeviceLocalBuffer(mesh->vertexBufferPosOnly.get(), (void*)&(vertexBuffer[vertexOffset / sizeof(float)]), mesh->vertexBufferPosOnly->size());
			rhiLayer->getDevice()->readbackDeviceLocalBuffer(mesh->indexBuffer.get(), (void*)&(indexBuffer[indexOffset / sizeof(uint16_t)]), mesh->indexBuffer->size());
			vertexOffset += mesh->vertexBufferPosOnly->size();
			indexOffset += mesh->indexBuffer->size();
		}
		asGroup.vertexBufferArray = GFX::GFXManager::get()->rhiLayer->getDevice()->createDeviceLocalBuffer(
			(void*)vertexBuffer.data(), vertexBuffer.size() * sizeof(float),
			(uint32_t)RHI::BufferUsage::VERTEX | (uint32_t)RHI::BufferUsage::SHADER_DEVICE_ADDRESS |
			(uint32_t)RHI::BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY | (uint32_t)RHI::BufferUsage::STORAGE);
		asGroup.indexBufferArray = GFX::GFXManager::get()->rhiLayer->getDevice()->createDeviceLocalBuffer(
			(void*)indexBuffer.data(), indexBuffer.size() * sizeof(uint16_t), 
			(uint32_t)RHI::BufferUsage::INDEX | (uint32_t)RHI::BufferUsage::SHADER_DEVICE_ADDRESS |
			(uint32_t)RHI::BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY | (uint32_t)RHI::BufferUsage::STORAGE);
		asGroup.GeometryInfoBuffer = GFX::GFXManager::get()->rhiLayer->getDevice()->createDeviceLocalBuffer(
			(void*)asGroup.geometryInfo.data(), asGroup.geometryInfo.size() * sizeof(GFX::ASGroup::GeometryInfo),
			(uint32_t)RHI::BufferUsage::STORAGE);
		Core::ResourceManager::get()->addResource(guid, std::move(asGroup));
	}

#pragma endregion
}