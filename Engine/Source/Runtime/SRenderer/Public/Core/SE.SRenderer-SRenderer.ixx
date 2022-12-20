module;
#include <array>
#include <vector>
#include <memory>
#include <string>
#include <cstdint>
#include <unordered_map>
#include "../../../Application/Public/SE.Application.Config.h"
export module SE.SRenderer:SRenderer;
import SE.Core.Resource;
import SE.Math.Geometric;
import SE.RHI;
import SE.GFX.Core;
import SE.GFX.RDG;

namespace SIByL
{
	export struct SRenderer {
		/**
		* Config for SRenderer
		*/
		struct Configuration {
			bool enableRayTracing = true;	// whether use ray tracing
		} config;

		/**
		* Standard Data Structure Definition
		* -----------------------------------------------
		* includes following items:
		* 1. InterleavedVertex:		vertex buffer structure
		* 2. Material:				material buffer structure
		* 3. GeometryDrawData:		geometry draw buffer structure
		*/
		/** standard interleaved position data */
		struct InterleavedVertex {
			Math::vec3 position;
			Math::vec3 normal;
			Math::vec3 tangent;
			Math::vec2 texCoords;
		};
		/** standard interleaved mesh data layout. 
		* use this structure to query mesh load. */
		static GFX::MeshDataLayout meshDataLayout;
		/** standard mesh load config to mesh loading */
		static GFX::MeshLoaderConfig meshLoadConfig;
		/** standard vertex buffer layout */
		static RHI::VertexBufferLayout vertexBufferLayout;

		/** standard material data */
		struct MaterialData {
			Math::vec4 albedo_tint;
			Math::vec2 uv_tiling	= { 1.f, 1.f };
			Math::vec2 uv_scaling	= { 1.f, 1.f };
			uint32_t mat_type = 0;
			uint32_t basecolor_opacity_tex;
			uint32_t normal_bump_tex;
			uint32_t roughness_metalic_ao_tex;
		};

		/** mesh / geometry draw call data */
		struct GeometryDrawData {
			uint32_t vertexOffset;
			uint32_t indexOffset;
			uint32_t materialID;
			uint32_t indexSize;
			RHI::AffineTransformMatrix geometryTransform = {};
		};

		/** global uniforms data for render descriptor set */
		struct GlobalUniforms {
			Math::mat4 view;
			Math::mat4 proj;
			Math::mat4 viewInverse;
			Math::mat4 projInverse;
		};

		/**
		* Scene Updating
		* --------------------------------
		* 1. update scene geometry / materials
		* 2. update camera pose / setting
		*/
		struct SceneDataPack {
			// integrated geometry data
			std::unique_ptr<RHI::Buffer> vertex_buffer			= nullptr;
			std::unique_ptr<RHI::Buffer> index_buffer			= nullptr;
			std::unique_ptr<RHI::Buffer> geometry_buffer		= nullptr;
			std::unique_ptr<RHI::Buffer> material_buffer		= nullptr;
			// unbinded textures array to contain all textures
			std::vector<RHI::TextureView*> unbinded_textures = {};
			// cpu data
			std::vector<float>				position_buffer_cpu	= {};	// position buffer cpu
			std::vector<float>				vertex_buffer_cpu	= {};	// vertex buffer cpu
			std::vector<uint32_t>			index_buffer_cpu	= {};	// index buffer cpu
			std::vector<GeometryDrawData>	geometry_buffer_cpu	= {};	// geometries data

			struct MeshReferenceRecord {
				uint32_t submesh_geometry_start;
				uint32_t submesh_geometry_end;
				std::unique_ptr<RHI::BLAS> blases;
			};

			bool geometryDirty = false;
			std::unordered_map<GFX::MeshReference*, MeshReferenceRecord> mesh_ref_record = {};
		} sceneDataPack;
		/** init non-scene resources */
		inline auto init(GFX::RDGraph* rdg) noexcept -> void;
		/** pack scene to scene data pack */
		inline auto packScene(GFX::Scene& scene) noexcept -> void;
		/** update main camera for the scene */
		inline auto updateCamera(GFX::TransformComponent const& transform, GFX::CameraComponent const& camera) noexcept -> void;

		/**
		* SRenderer info
		*/
		/** the rdgraph binded */
		GFX::RDGraph* rdgraph = nullptr;
		/** register passes */
		struct Pass {
			virtual ~Pass() = default;
			virtual auto loadShaders() noexcept -> void = 0;
			virtual auto registerPass(SRenderer* renderer) noexcept -> void = 0;
		};
		/** passes registered */
		std::vector<std::unique_ptr<Pass>> passes;

		/**
		* Common descriptor set definition.
		* -----------------------------------------------
		* Use frequency-based descriptor sets arrangment.
		* Pre-define fixed structures for rasterizer / tracer passes.
		* set 0: global uniforms + unbind textures
		* set 1: material data
		* push constant: drawcall constant
		*/
		struct CommonDescriptorData {
			std::unique_ptr<RHI::BindGroupLayout> set0_layout = 0;
			std::array<std::unique_ptr<RHI::BindGroup>, MULTIFRAME_FLIGHTS_COUNT> set0_flights = {};
			std::array<RHI::BindGroup*, MULTIFRAME_FLIGHTS_COUNT> set0_flights_array = {};
		} commonDescData;
	};

#pragma region SRENDERER_IMPL

	GFX::MeshDataLayout SRenderer::meshDataLayout = {
			{	{RHI::VertexFormat::FLOAT32X3, GFX::MeshDataLayout::VertexInfo::POSITION},
				{RHI::VertexFormat::FLOAT32X3, GFX::MeshDataLayout::VertexInfo::NORMAL},
				{RHI::VertexFormat::FLOAT32X3, GFX::MeshDataLayout::VertexInfo::TANGENT},
				{RHI::VertexFormat::FLOAT32X2, GFX::MeshDataLayout::VertexInfo::UV}, },
			RHI::IndexFormat::UINT32_T
	};

	GFX::MeshLoaderConfig SRenderer::meshLoadConfig = {
		meshDataLayout, true, true, false
	};

	RHI::VertexBufferLayout SRenderer::vertexBufferLayout = GFX::getVertexBufferLayout(meshDataLayout);

	inline auto SRenderer::init(GFX::RDGraph* rdg) noexcept -> void {
		RHI::Device* device = GFX::GFXManager::get()->rhiLayer->getDevice();
		// bind rdgraph
		rdgraph = rdg;
		// create global uniform buffer
		rdg->createStructuredUniformBuffer<GlobalUniforms>("global_uniform_buffer");
		// create common descriptor data
		RHI::ShaderStagesFlags stages =
			  (uint32_t)RHI::ShaderStages::VERTEX
			| (uint32_t)RHI::ShaderStages::FRAGMENT
			| (uint32_t)RHI::ShaderStages::RAYGEN
			| (uint32_t)RHI::ShaderStages::CLOSEST_HIT
			| (uint32_t)RHI::ShaderStages::MISS
			| (uint32_t)RHI::ShaderStages::ANY_HIT
			| (uint32_t)RHI::ShaderStages::COMPUTE;
		commonDescData.set0_layout = device->createBindGroupLayout(
			RHI::BindGroupLayoutDescriptor{ {
				RHI::BindGroupLayoutEntry{ 0, stages, RHI::BufferBindingLayout{RHI::BufferBindingType::UNIFORM}},
				RHI::BindGroupLayoutEntry{ 1, stages, RHI::BufferBindingLayout{RHI::BufferBindingType::STORAGE}},
				RHI::BindGroupLayoutEntry{ 2, stages, RHI::BufferBindingLayout{RHI::BufferBindingType::STORAGE}},
				RHI::BindGroupLayoutEntry{ 3, stages, RHI::BufferBindingLayout{RHI::BufferBindingType::STORAGE}},
				RHI::BindGroupLayoutEntry{ 4, stages, RHI::BindlessTexturesBindingLayout{}},
				} }
		);
	}

	inline auto SRenderer::packScene(GFX::Scene& scene) noexcept -> void {
		RHI::Device* device = GFX ::GFXManager::get()->rhiLayer->getDevice();
		for (auto go_handle : scene.gameObjects) {
			auto* go = scene.getGameObject(go_handle.first);
			GFX::MeshReference* meshref = go->getEntity().getComponent<GFX::MeshReference>();
			if (!meshref) continue;
			Math::mat4 objectMat;
			{	// get mesh transform matrix
				GFX::TransformComponent* transform = go->getEntity().getComponent<GFX::TransformComponent>();
				objectMat = transform->getTransform() * objectMat;
				while (go->parent != Core::NULL_ENTITY) {
					go = scene.getGameObject(go->parent);
					GFX::TransformComponent* transform = go->getEntity().getComponent<GFX::TransformComponent>();
					objectMat = transform->getTransform() * objectMat;
				}
			}
			// if do not have according record, add the record
			auto meshRecordRef = sceneDataPack.mesh_ref_record.find(meshref);
			if (meshRecordRef == sceneDataPack.mesh_ref_record.end()) {
				sceneDataPack.geometryDirty = true;
				sceneDataPack.mesh_ref_record[meshref] = {};
				meshRecordRef = sceneDataPack.mesh_ref_record.find(meshref);
				{	// create mesh record
					uint32_t geometry_beg = sceneDataPack.geometry_buffer_cpu.size();
					uint32_t geometry_end = geometry_beg + meshref->mesh->submeshes.size();
					uint32_t vertex_offset = sceneDataPack.vertex_buffer_cpu.size();
					sceneDataPack.vertex_buffer_cpu.resize(vertex_offset + meshref->mesh->vertexBuffer_host.size / sizeof(float));
					memcpy(&(sceneDataPack.vertex_buffer_cpu[vertex_offset]), meshref->mesh->vertexBuffer_host.data, meshref->mesh->vertexBuffer_host.size);
					uint32_t index_offset = sceneDataPack.index_buffer_cpu.size();
					sceneDataPack.index_buffer_cpu.resize(index_offset + meshref->mesh->indexBuffer_host.size / sizeof(uint32_t));
					memcpy(&(sceneDataPack.index_buffer_cpu[index_offset]), meshref->mesh->indexBuffer_host.data, meshref->mesh->indexBuffer_host.size);
					// 
					for (auto& submesh : meshref->mesh->submeshes) {
						GeometryDrawData geometry;
						geometry.vertexOffset = submesh.baseVertex;
						geometry.indexOffset = submesh.offset;
						geometry.materialID = 0;
						geometry.indexSize = submesh.size;
						geometry.geometryTransform = objectMat;
						sceneDataPack.geometry_buffer_cpu.emplace_back(geometry);
						//blasDesc.triangleGeometries.push_back(RHI::BLASTriangleGeometry{
						//	meshref->mesh->positionBuffer.get(),
						//	meshref->mesh->indexBuffer.get(),
						//	meshref->mesh->vertexBuffer.get(),
						//	RHI::IndexFormat::UINT32_T,
						//	submehs.size,
						//	submehs.baseVertex,
						//	submehs.size / 3,
						//	uint32_t(submehs.offset * sizeof(uint32_t)),
						//	RHI::AffineTransformMatrix(objectMat),
						//	(uint32_t)RHI::BLASGeometryFlagBits::NO_DUPLICATE_ANY_HIT_INVOCATION,
						//	submehs.matID
						//	}
						//);
					//for (auto& triangleGeometry : blasDesc.triangleGeometries) {
					//	ASGroup::GeometryInfo geometryInfo;
					//	// vertex buffer
					//	auto findVertexBuffer = vertexBufferMaps.find(triangleGeometry.vertexBuffer);
					//	if (findVertexBuffer == vertexBufferMaps.end()) {
					//		VertexBufferEntry vbt{ vertexOffset, triangleGeometry.vertexBuffer };
					//		vertexBufferMaps[triangleGeometry.vertexBuffer] = vbt;
					//		vertexBuffers.push_back(vbt);
					//		vertexOffset += triangleGeometry.vertexBuffer->size() / (sizeof(float) * vertexStride);
					//		findVertexBuffer = vertexBufferMaps.find(triangleGeometry.vertexBuffer);
					//	}
					//}

					}

				}

				//// create BLAS
				//if (config.enableRayTracing) {
				//	RHI::BLASDescriptor blasDesc;
				//	for (auto& submehs : meshref->mesh->submeshes) {
				//		blasDesc.triangleGeometries.push_back(RHI::BLASTriangleGeometry{
				//			meshref->mesh->positionBuffer_device.get(),
				//			meshref->mesh->indexBuffer_device.get(),
				//			meshref->mesh->vertexBuffer_device.get(),
				//			RHI::IndexFormat::UINT32_T,
				//			submehs.size,
				//			submehs.baseVertex,
				//			submehs.size / 3,
				//			uint32_t(submehs.offset * sizeof(uint32_t)),
				//			RHI::AffineTransformMatrix(objectMat),
				//			(uint32_t)RHI::BLASGeometryFlagBits::NO_DUPLICATE_ANY_HIT_INVOCATION,
				//			submehs.matID
				//			}
				//		);
				//	}
				//}
			}
		}

		// if geometry is dirty, rebuild all the buffers
		if (sceneDataPack.geometryDirty) {
			RHI::ShaderStagesFlags stages =
				(uint32_t)RHI::ShaderStages::VERTEX
				| (uint32_t)RHI::ShaderStages::FRAGMENT
				| (uint32_t)RHI::ShaderStages::RAYGEN
				| (uint32_t)RHI::ShaderStages::CLOSEST_HIT
				| (uint32_t)RHI::ShaderStages::MISS
				| (uint32_t)RHI::ShaderStages::ANY_HIT
				| (uint32_t)RHI::ShaderStages::COMPUTE;
			sceneDataPack.vertex_buffer = device->createDeviceLocalBuffer(
				sceneDataPack.vertex_buffer_cpu.data(),
				sceneDataPack.vertex_buffer_cpu.size() * sizeof(float), stages);
			sceneDataPack.index_buffer = device->createDeviceLocalBuffer(
				sceneDataPack.index_buffer_cpu.data(),
				sceneDataPack.index_buffer_cpu.size() * sizeof(uint32_t), stages);
			sceneDataPack.geometry_buffer = device->createDeviceLocalBuffer(
				sceneDataPack.geometry_buffer_cpu.data(),
				sceneDataPack.geometry_buffer_cpu.size() * sizeof(GeometryDrawData), stages);
			// rebuild all bind groups
			for (int i = 0; i < 2; ++i) {
				commonDescData.set0_flights[i] = device->createBindGroup(RHI::BindGroupDescriptor{
					commonDescData.set0_layout.get(),
					std::vector<RHI::BindGroupEntry>{
						{0,RHI::BindingResource{rdgraph->getStructuredUniformBuffer<GlobalUniforms>("global_uniform_buffer")->getBufferBinding(i)}},
						{1,RHI::BindingResource{{sceneDataPack.vertex_buffer.get(), 0, sceneDataPack.vertex_buffer->size()}}},
						{2,RHI::BindingResource{{sceneDataPack.index_buffer.get(), 0, sceneDataPack.index_buffer->size()}}},
						{3,RHI::BindingResource{{sceneDataPack.geometry_buffer.get(), 0, sceneDataPack.geometry_buffer->size()}}},
						{4,RHI::BindingResource{sceneDataPack.unbinded_textures,
							Core::ResourceManager::get()->getResource<GFX::Sampler>(GFX::GFXManager::get()->commonSampler.defaultSampler)->sampler.get()}},
				} });
				commonDescData.set0_flights_array[i] = commonDescData.set0_flights[i].get();
			}
			// set scene data pack not dirty
			sceneDataPack.geometryDirty = false;
		}

		//GFX::ASGroup asGroup = {};
		//RHI::TLASDescriptor modified_desc = desc;
		//std::unordered_map<RHI::BLAS*, std::pair<uint32_t, uint32_t>> map;
		//uint32_t vertexOffset = 0;
		//uint32_t indexOffset = 0;
		//struct VertexBufferEntry {
		//	uint32_t	 vertexOffset = 0;
		//	RHI::Buffer* vertexBuffer = nullptr;
		//};
		//struct IndexBufferEntry {
		//	uint32_t	 indexOffset = 0;
		//	RHI::Buffer* indexBuffer = nullptr;
		//};
		//std::unordered_map<RHI::Buffer*, VertexBufferEntry> vertexBufferMaps;
		//std::unordered_map<RHI::Buffer*, IndexBufferEntry> indexBufferMaps;
		//std::vector<VertexBufferEntry> vertexBuffers;
		//std::vector<IndexBufferEntry> indexBuffers;

		//for (int i = 0; i < desc.instances.size(); ++i) {
		//	// if BLAS has pushed
		//	if (map.find(desc.instances[i].blas) != map.end()) {
		//		std::pair<uint32_t, uint32_t> geometryRange = map[desc.instances[i].blas];
		//		modified_desc.instances[i].instanceCustomIndex = geometryRange.first;
		//	}
		//	/// else if BLAS has not pushed
		//	else {
		//		RHI::BLAS* blas = desc.instances[i].blas;
		//		RHI::BLASDescriptor blasDesc = blas->getDescriptor();
		//		uint32_t geometryBegin = asGroup.geometryInfo.size();
		//		uint32_t geometryEnd = geometryBegin;
		//		for (auto& triangleGeometry : blasDesc.triangleGeometries) {
		//			ASGroup::GeometryInfo geometryInfo;
		//			// vertex buffer
		//			auto findVertexBuffer = vertexBufferMaps.find(triangleGeometry.vertexBuffer);
		//			if (findVertexBuffer == vertexBufferMaps.end()) {
		//				VertexBufferEntry vbt{ vertexOffset, triangleGeometry.vertexBuffer };
		//				vertexBufferMaps[triangleGeometry.vertexBuffer] = vbt;
		//				vertexBuffers.push_back(vbt);
		//				vertexOffset += triangleGeometry.vertexBuffer->size() / (sizeof(float) * vertexStride);
		//				findVertexBuffer = vertexBufferMaps.find(triangleGeometry.vertexBuffer);
		//			}
		//			VertexBufferEntry& vertexEntry = findVertexBuffer->second;
		//			geometryInfo.vertexOffset = vertexEntry.vertexOffset + triangleGeometry.firstVertex;
		//			// index buffer
		//			auto findIndexBuffer = indexBufferMaps.find(triangleGeometry.indexBuffer);
		//			if (findIndexBuffer == indexBufferMaps.end()) {
		//				IndexBufferEntry ibe{ indexOffset, triangleGeometry.indexBuffer };
		//				indexBufferMaps[triangleGeometry.indexBuffer] = ibe;
		//				indexBuffers.push_back(ibe);
		//				indexOffset += triangleGeometry.indexBuffer->size() / sizeof(uint32_t);
		//				findIndexBuffer = indexBufferMaps.find(triangleGeometry.indexBuffer);
		//			}
		//			IndexBufferEntry& indexEntry = findIndexBuffer->second;
		//			geometryInfo.indexOffset = indexEntry.indexOffset + triangleGeometry.primitiveOffset / sizeof(uint32_t);
		//			geometryInfo.geometryTransform = Math::transpose(Math::mat4(triangleGeometry.transform));
		//			geometryInfo.materialID = triangleGeometry.materialID;
		//			asGroup.geometryInfo.push_back(geometryInfo);
		//			++geometryEnd;
		//		}
		//		modified_desc.instances[i].instanceCustomIndex = geometryBegin;
		//		map[desc.instances[i].blas] = std::pair<uint32_t, uint32_t>{ geometryBegin, geometryEnd };
		//	}
		//}
		//// create TLAS
		//asGroup.tlas = rhiLayer->getDevice()->createTLAS(modified_desc);
		//// create Buffers
		//std::vector<float> vertexBuffer(vertexOffset * vertexStride);
		//std::vector<uint32_t> indexBuffer(indexOffset);
		//vertexOffset = 0;
		//indexOffset = 0;
		//for (auto iter : vertexBuffers) {
		//	rhiLayer->getDevice()->readbackDeviceLocalBuffer(
		//		iter.vertexBuffer,
		//		(void*)&(vertexBuffer[vertexOffset / sizeof(float)]),
		//		iter.vertexBuffer->size());
		//	vertexOffset += iter.vertexBuffer->size();
		//}
		//for (auto iter : indexBuffers) {
		//	rhiLayer->getDevice()->readbackDeviceLocalBuffer(
		//		iter.indexBuffer,
		//		(void*)&(indexBuffer[indexOffset / sizeof(uint32_t)]),
		//		iter.indexBuffer->size());
		//	indexOffset += iter.indexBuffer->size();
		//}
		//asGroup.vertexBufferArray = GFX::GFXManager::get()->rhiLayer->getDevice()->createDeviceLocalBuffer(
		//	(void*)vertexBuffer.data(), vertexBuffer.size() * sizeof(float),
		//	(uint32_t)RHI::BufferUsage::VERTEX | (uint32_t)RHI::BufferUsage::SHADER_DEVICE_ADDRESS |
		//	(uint32_t)RHI::BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY | (uint32_t)RHI::BufferUsage::STORAGE);
		//asGroup.indexBufferArray = GFX::GFXManager::get()->rhiLayer->getDevice()->createDeviceLocalBuffer(
		//	(void*)indexBuffer.data(), indexBuffer.size() * sizeof(uint32_t),
		//	(uint32_t)RHI::BufferUsage::INDEX | (uint32_t)RHI::BufferUsage::SHADER_DEVICE_ADDRESS |
		//	(uint32_t)RHI::BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY | (uint32_t)RHI::BufferUsage::STORAGE);
		//asGroup.GeometryInfoBuffer = GFX::GFXManager::get()->rhiLayer->getDevice()->createDeviceLocalBuffer(
		//	(void*)asGroup.geometryInfo.data(), asGroup.geometryInfo.size() * sizeof(GFX::ASGroup::GeometryInfo),
		//	(uint32_t)RHI::BufferUsage::STORAGE);
		//Core::ResourceManager::get()->addResource(guid, std::move(asGroup));
	}

	inline auto SRenderer::updateCamera(GFX::TransformComponent const& transform, GFX::CameraComponent const& camera) noexcept -> void {
		RHI::MultiFrameFlights* multiFrameFlights = GFX::GFXManager::get()->rhiLayer->getMultiFrameFlights();
		GlobalUniforms globalUni;
		globalUni.view = Math::transpose(Math::lookAt(transform.translation, transform.translation + transform.getRotatedForward(), Math::vec3(0, 1, 0)).m);
		globalUni.proj = Math::transpose(Math::perspective(22.f, 1.f * 800 / 600, 0.1f, 1000.f).m);
		globalUni.viewInverse = Math::inverse(globalUni.view);
		globalUni.projInverse = Math::inverse(globalUni.proj);
		rdgraph->getStructuredUniformBuffer<GlobalUniforms>("global_uniform_buffer")->setStructure(globalUni, multiFrameFlights->getFlightIndex());
	}

#pragma endregion
}