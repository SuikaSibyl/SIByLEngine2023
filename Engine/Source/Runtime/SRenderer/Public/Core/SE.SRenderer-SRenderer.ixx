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
			std::unique_ptr<RHI::Buffer> position_buffer		= nullptr;
			std::unique_ptr<RHI::Buffer> index_buffer			= nullptr;
			std::unique_ptr<RHI::Buffer> material_buffer		= nullptr;
			// unbinded textures array to contain all textures
			std::vector<RHI::TextureView*> unbinded_textures = {};
			// cpu data
			std::vector<float>				position_buffer_cpu	= {};	// position buffer cpu
			std::vector<float>				vertex_buffer_cpu	= {};	// vertex buffer cpu
			std::vector<uint32_t>			index_buffer_cpu	= {};	// index buffer cpu
			std::vector<GeometryDrawData>	geometry_buffer_cpu	= {};	// geometries data
			RHI::TLASDescriptor				tlas_desc = {};
			std::shared_ptr<RHI::TLAS>		tlas = {};
			std::shared_ptr<RHI::TLAS>		back_tlas = {};

			struct MeshRecord {
				std::vector<GeometryDrawData> submesh_geometry;
				RHI::BLASDescriptor blas_desc;
				std::unique_ptr<RHI::BLAS> blases = nullptr;
			};

			struct MeshReferenceRecord {
				GFX::Mesh* mesh;
				RHI::BLASInstance blasInstance;
				std::vector<uint32_t> geometry_indices;
			};

			bool geometryDirty = false;
			std::unordered_map<GFX::Mesh*, MeshRecord> mesh_record = {};
			std::unordered_map<GFX::MeshReference*, MeshReferenceRecord> mesh_ref_record = {};
		} sceneDataPack;
		/** init non-scene resources */
		inline auto init(GFX::RDGraph* rdg, GFX::Scene& scene) noexcept -> void;
		/** invalid scene */
		inline auto invalidScene(GFX::Scene& scene) noexcept -> void;
		/** pack scene to scene data pack */
		inline auto packScene(GFX::Scene& scene) noexcept -> void;
		/** update main camera for the scene */
		inline auto updateCamera(GFX::TransformComponent const& transform, GFX::CameraComponent const& camera) noexcept -> void;

		struct {
			uint32_t batchIdx = 0;
		} state;
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
		/** resources registered */
		struct TextureRegisterInfo {
			std::string name;
			GFX::RDGTexture::Desc desc;
		};
		std::vector<TextureRegisterInfo> textures;
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
			// set 1 for ray tracing
			std::unique_ptr<RHI::BindGroupLayout> set1_layout_rt = 0;
			std::array<std::unique_ptr<RHI::BindGroup>, MULTIFRAME_FLIGHTS_COUNT> set1_flights_rt = {};
			std::array<RHI::BindGroup*, MULTIFRAME_FLIGHTS_COUNT> set1_flights_rt_array = {};

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
		meshDataLayout, true, true, false, true
	};

	RHI::VertexBufferLayout SRenderer::vertexBufferLayout = GFX::getVertexBufferLayout(meshDataLayout);

	inline auto SRenderer::init(GFX::RDGraph* rdg, GFX::Scene& scene) noexcept -> void {
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
		commonDescData.set1_layout_rt = device->createBindGroupLayout(
			RHI::BindGroupLayoutDescriptor{ {
				RHI::BindGroupLayoutEntry{ 0, stages, RHI::AccelerationStructureBindingLayout{}},
				RHI::BindGroupLayoutEntry{ 1, stages, RHI::StorageTextureBindingLayout{}},
				} }
		);
		// pack scene
		packScene(scene);
		// register textures
		for (auto& texture : textures) {
			rdg->createTexture(texture.name.c_str(), texture.desc);
		}
		// register passes
		for (auto& pass : passes) {
			pass->loadShaders();
			pass->registerPass(this);
		}
	}

	inline auto SRenderer::invalidScene(GFX::Scene& scene) noexcept -> void {
		// 
		RHI::Device* device = GFX::GFXManager::get()->rhiLayer->getDevice();
		sceneDataPack.tlas_desc.instances.clear();
		// for all mesh refs
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
			}
			else {
				for (auto idx : meshRecordRef->second.geometry_indices) {
					sceneDataPack.geometry_buffer_cpu[idx].geometryTransform = objectMat;
				}
				meshRecordRef->second.blasInstance.transform = objectMat;
				sceneDataPack.tlas_desc.instances.push_back(meshRecordRef->second.blasInstance);
			}
		}
		sceneDataPack.back_tlas = sceneDataPack.tlas;
		sceneDataPack.tlas = device->createTLAS(sceneDataPack.tlas_desc);

		RHI::MultiFrameFlights* multiFrameFlights = GFX::GFXManager::get()->rhiLayer->getMultiFrameFlights();
		uint32_t fid = multiFrameFlights->getFlightIndex();
		
		commonDescData.set1_flights_rt[fid]->updateBinding(std::vector<RHI::BindGroupEntry>{
			{0, RHI::BindingResource{ sceneDataPack.tlas.get() }},
		});
		
		auto geometry_buffer = rdgraph->getStructuredArrayMultiStorageBuffer<GeometryDrawData>("geometry_buffer");
		geometry_buffer->setStructure(sceneDataPack.geometry_buffer_cpu.data(), fid);
	}

	inline auto SRenderer::packScene(GFX::Scene& scene) noexcept -> void {
		RHI::Device* device = GFX ::GFXManager::get()->rhiLayer->getDevice();
		for (auto go_handle : scene.gameObjects) {
			auto* go = scene.getGameObject(go_handle.first);
			GFX::MeshReference* meshref = go->getEntity().getComponent<GFX::MeshReference>();
			if (!meshref) continue;
			GFX::Mesh* mesh = meshref->mesh;
			if (!mesh) continue;
			// if do not have according mesh record, add the record
			auto meshRecord = sceneDataPack.mesh_record.find(mesh);
			if (meshRecord == sceneDataPack.mesh_record.end()) {
				sceneDataPack.geometryDirty = true;
				sceneDataPack.mesh_record[mesh] = {};
				meshRecord = sceneDataPack.mesh_record.find(mesh);
				{	// create mesh record, add all data buffers
					uint32_t vertex_offset = sceneDataPack.vertex_buffer_cpu.size();
					uint32_t position_offset = sceneDataPack.position_buffer_cpu.size();
					sceneDataPack.vertex_buffer_cpu.resize(vertex_offset + mesh->vertexBuffer_host.size / sizeof(float));
					memcpy(&(sceneDataPack.vertex_buffer_cpu[vertex_offset]), mesh->vertexBuffer_host.data, mesh->vertexBuffer_host.size);
					if (config.enableRayTracing) {
						sceneDataPack.position_buffer_cpu.resize(position_offset + mesh->positionBuffer_host.size / sizeof(float));
						memcpy(&(sceneDataPack.position_buffer_cpu[position_offset]), mesh->positionBuffer_host.data, mesh->positionBuffer_host.size);
					}
					uint32_t index_offset = sceneDataPack.index_buffer_cpu.size();
					sceneDataPack.index_buffer_cpu.resize(index_offset + mesh->indexBuffer_host.size / sizeof(uint32_t));
					memcpy(&(sceneDataPack.index_buffer_cpu[index_offset]), mesh->indexBuffer_host.data, mesh->indexBuffer_host.size);
					// add all submeshes
					for (auto& submesh : mesh->submeshes) {
						GeometryDrawData geometry;
						geometry.vertexOffset = submesh.baseVertex + vertex_offset * sizeof(float) / SRenderer::vertexBufferLayout.arrayStride;
						geometry.indexOffset = submesh.offset + index_offset;
						geometry.materialID = 0;
						geometry.indexSize = submesh.size;
						geometry.geometryTransform = {};
						meshRecord->second.submesh_geometry.push_back(geometry);

						if (config.enableRayTracing) {
							RHI::BLASDescriptor& blasDesc = meshRecord->second.blas_desc;
							blasDesc.triangleGeometries.push_back(RHI::BLASTriangleGeometry{
								nullptr,
								nullptr,
								nullptr,
								RHI::IndexFormat::UINT32_T,
								uint32_t(mesh->positionBuffer_host.size / (sizeof(float) * 3)),
								geometry.vertexOffset,
								geometry.indexSize / 3,
								uint32_t(geometry.indexOffset * sizeof(uint32_t)),
								RHI::AffineTransformMatrix{},
								(uint32_t)RHI::BLASGeometryFlagBits::NO_DUPLICATE_ANY_HIT_INVOCATION,
								geometry.materialID
								}
							);
						}
					}
				}
			}
			// get transform
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
			// if do not have according mesh ref record, add the record
			auto meshRefRecord = sceneDataPack.mesh_ref_record.find(meshref);
			if (meshRefRecord == sceneDataPack.mesh_ref_record.end()) {
				sceneDataPack.geometryDirty = true;
				sceneDataPack.mesh_ref_record[meshref] = {};
				meshRefRecord = sceneDataPack.mesh_ref_record.find(meshref);
				{	// create mesh record
					meshRefRecord->second.mesh = mesh;
					uint32_t geometry_start = sceneDataPack.geometry_buffer_cpu.size();
					for (auto& iter : meshRecord->second.submesh_geometry) {
						meshRefRecord->second.geometry_indices.push_back(sceneDataPack.geometry_buffer_cpu.size());
						GeometryDrawData geometrydata = iter;
						geometrydata.geometryTransform = objectMat;
						sceneDataPack.geometry_buffer_cpu.emplace_back(geometrydata);
					}
				}
			}
		}

		// if geometry is dirty, rebuild all the buffers
		if (sceneDataPack.geometryDirty) {
			sceneDataPack.vertex_buffer = device->createDeviceLocalBuffer(
				sceneDataPack.vertex_buffer_cpu.data(),
				sceneDataPack.vertex_buffer_cpu.size() * sizeof(float), 
				(uint32_t)RHI::BufferUsage::VERTEX
				| (uint32_t)RHI::BufferUsage::SHADER_DEVICE_ADDRESS
				| (uint32_t)RHI::BufferUsage::STORAGE);
			sceneDataPack.index_buffer = device->createDeviceLocalBuffer(
				sceneDataPack.index_buffer_cpu.data(),
				sceneDataPack.index_buffer_cpu.size() * sizeof(uint32_t), 
				(uint32_t)RHI::BufferUsage::INDEX
				| (uint32_t)RHI::BufferUsage::SHADER_DEVICE_ADDRESS
				| (uint32_t)RHI::BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY
				| (uint32_t)RHI::BufferUsage::STORAGE);
			if (config.enableRayTracing) {
				sceneDataPack.position_buffer = device->createDeviceLocalBuffer(
					sceneDataPack.position_buffer_cpu .data(),
					sceneDataPack.position_buffer_cpu.size() * sizeof(float), 
					(uint32_t)RHI::BufferUsage::VERTEX
					| (uint32_t)RHI::BufferUsage::SHADER_DEVICE_ADDRESS
					| (uint32_t)RHI::BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY
					| (uint32_t)RHI::BufferUsage::STORAGE);
			}
			auto geometry_buffer = rdgraph->createStructuredArrayMultiStorageBuffer<GeometryDrawData>("geometry_buffer", sceneDataPack.geometry_buffer_cpu.size());

			rdgraph->addBehavior([&, device]()->void {
				// rebuild all bind groups
				for (int i = 0; i < 2; ++i) {
					commonDescData.set0_flights[i] = device->createBindGroup(RHI::BindGroupDescriptor{
						commonDescData.set0_layout.get(),
						std::vector<RHI::BindGroupEntry>{
							{0,RHI::BindingResource{rdgraph->getStructuredUniformBuffer<GlobalUniforms>("global_uniform_buffer")->getBufferBinding(i)}},
							{1,RHI::BindingResource{{sceneDataPack.vertex_buffer.get(), 0, sceneDataPack.vertex_buffer->size()}}},
							{2,RHI::BindingResource{{sceneDataPack.index_buffer.get(), 0, sceneDataPack.index_buffer->size()}}},
							{3,RHI::BindingResource{rdgraph->getStructuredArrayMultiStorageBuffer<GeometryDrawData>("geometry_buffer")->getBufferBinding(i)}},
							{4,RHI::BindingResource{sceneDataPack.unbinded_textures,
								Core::ResourceManager::get()->getResource<GFX::Sampler>(GFX::GFXManager::get()->commonSampler.defaultSampler)->sampler.get()}},
					} });
					commonDescData.set0_flights_array[i] = commonDescData.set0_flights[i].get();
				}
				}, GFX::RDGraph::BehaviorPhase::AfterDevirtualize_BeforePassSetup);
			// set scene data pack not dirty
			sceneDataPack.geometryDirty = false;
		}

		// create BLAS / TLAS
		for (auto& iter : sceneDataPack.mesh_record) {
			if (iter.second.blases == nullptr) {
				iter.second.blas_desc.allowRefitting = true;
				iter.second.blas_desc.allowCompaction = true;
				for (auto& mesh : iter.second.blas_desc.triangleGeometries) {
					mesh.positionBuffer = sceneDataPack.position_buffer.get();
					mesh.indexBuffer = sceneDataPack.index_buffer.get();
				}
				iter.second.blases = device->createBLAS(iter.second.blas_desc);
			}
		}
		for (auto& iter : sceneDataPack.mesh_ref_record) {
			// if BLAS is not created
			iter.second.blasInstance.blas = sceneDataPack.mesh_record[iter.second.mesh].blases.get();
			sceneDataPack.tlas_desc.instances.push_back(iter.second.blasInstance);
		}
		sceneDataPack.tlas_desc.allowRefitting = true;
		sceneDataPack.tlas = device->createTLAS(sceneDataPack.tlas_desc);
		rdgraph->addBehavior([&, device]()->void {
			// rebuild all bind groups
			for (int i = 0; i < 2; ++i) {
				commonDescData.set1_flights_rt[i] = device->createBindGroup(RHI::BindGroupDescriptor{
					commonDescData.set1_layout_rt.get(),
					std::vector<RHI::BindGroupEntry>{
						{0,RHI::BindingResource{sceneDataPack.tlas.get()}},
						{1,RHI::BindingResource{rdgraph->getTexture("TracerTarget_Color")->texture->originalView.get()}},
				} });
				commonDescData.set1_flights_rt_array[i] = commonDescData.set1_flights_rt[i].get();
			}
			}, GFX::RDGraph::BehaviorPhase::AfterDevirtualize_BeforePassSetup);
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