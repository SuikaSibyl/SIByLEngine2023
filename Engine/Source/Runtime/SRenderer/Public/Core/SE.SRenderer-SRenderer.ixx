module;
#include <array>
#include <vector>
#include <memory>
#include <string>
#include <cstdint>
#include <unordered_map>
#include "../../../Application/Public/SE.Application.Config.h"
export module SE.SRenderer:SRenderer;
import :RTCommon;
import SE.Core.Resource;
import SE.Core.ECS;
import SE.Math.Geometric;
import SE.Math.Misc;
import SE.RHI;
import SE.GFX;
import SE.RDG;

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
			Math::vec2 uv_tiling = { 1.f, 1.f };
			Math::vec2 uv_scaling = { 1.f, 1.f };
			uint32_t mat_type = 0;
			uint32_t basecolor_opacity_tex;
			uint32_t normal_bump_tex;
			uint32_t roughness_metalic_ao_tex;
			uint32_t padding_0;
			uint32_t padding_1;
			uint32_t padding_2;
			uint32_t bsdf_id = 1;
		};

		/** standard light data */
		struct LightData {
			// 0: diffuse area light - sphere
			// 1: diffuse area light - triangle mesh
			// 2: env map
			uint32_t	lightType;
			Math::vec3	intensity;
			uint32_t	index;						// geometry index (type 0/1) or texture index (type 2)
			uint32_t	sample_dist_size_0;			// sample distribution unit size
			uint32_t	sample_dist_offset_pmf_0;	// sample distribution offset for pmf start
			uint32_t	sample_dist_offset_cdf_0;	// sample distribution offset for cdf start
			float		total_value;
			uint32_t	sample_dist_size_1;			// (another dim of) sample distribution unit size
			uint32_t	sample_dist_offset_pmf_1;	// (another dim of) sample distribution offset for pmf start
			uint32_t	sample_dist_offset_cdf_1;	// (another dim of) sample distribution offset for cdf start
		};

		/** mesh / geometry draw call data */
		struct GeometryDrawData {
			uint32_t vertexOffset;
			uint32_t indexOffset;
			uint32_t materialID;
			uint32_t indexSize;
			float	 surfaceArea;
			uint32_t lightID;
			uint32_t primitiveType;
			float oddNegativeScaling;
			RHI::AffineTransformMatrix geometryTransform = {};
			RHI::AffineTransformMatrix geometryTransformInverse = {};
		};

		/**
		* Universal camera data for both rasterizer & raytracer.
		* Definition & annotations are modified from Falcor:
		* @ref: https://github.com/NVIDIAGameWorks/Falcor
		*/
		struct CameraData {
			Math::mat4	viewMat;                   ///< Camera view matrix.
			Math::mat4	prevViewMat;               ///< Camera view matrix associated to previous frame.
			Math::mat4	projMat;                   ///< Camera projection matrix.
			Math::mat4	viewProjMat;               ///< Camera view-projection matrix.
			Math::mat4	invViewProj;               ///< Camera inverse view-projection matrix.
			Math::mat4	viewProjMatNoJitter;       ///< Camera view-projection matrix. No jittering is applied!
			Math::mat4	prevViewProjMatNoJitter;   ///< Camera view-projection matrix associated to previous frame. No jittering is applied!
			Math::mat4	projMatNoJitter;           ///< Camera projection matrix. No jittering is applied!

			Math::vec3	posW;                      ///< Camera world-space position.
			float		focalLength;               ///< Camera focal length in mm. Default is 59 degree vertical, 90 horizontal FOV at 16:9 aspect ratio.
			Math::vec3	prevPosW;                  ///< Camera world-space position associated to previous frame.
			float		rectArea;                  ///< Recrtangles area A at z=1 plane.
			Math::vec3	up;                        ///< Camera world-space up vector.
			float		aspectRatio;               ///< Camera film frame aspect ratio, equal to frameWidth / frameHeight
			Math::vec3	target;                    ///< Camera target point in world-space.
			float		nearZ;                     ///< Camera near plane.
			Math::vec3	cameraU;                   ///< Camera base vector U. Normalized it indicates the right image plane vector. The length is dependent on the FOV.
			float		farZ;                      ///< Camera far plane.
			Math::vec3	cameraV;                   ///< Camera base vector V. Normalized it indicates the up image plane vector. The length is dependent on the FOV.
			float		jitterX;                   ///< Eventual camera jitter along the x axis expressed as a subpixel offset divided by screen width (positive value shifts the image right).
			Math::vec3	cameraW;                   ///< Camera base vector W. Normalized it indicates the forward direction. The length is the camera focal distance.
			float		jitterY;                   ///< Eventual camera jitter along the y axis expressed as a subpixel offset divided by screen height (positive value shifts the image up).

			float       frameHeight;               ///< Camera film frame height in mm. 24 is the height of a 35mm film
			float       frameWidth;                ///< Camera film frame width in mm.  42 2/3 is the width assuming 24mm height and a 16:9 aspect ratio
			float       focalDistance;             ///< Camera focal distance in scene units.
			float       apertureRadius;            ///< Camera aperture radius in scene units.
			float       shutterSpeed;              ///< Camera shutter speed in seconds.
			float       ISOSpeed;                  ///< Camera film speed based on ISO standards.
			float      _padding1;
			float      _padding2;
		};

		/** global uniforms data for render descriptor set */
		struct alignas(64) GlobalUniforms {
			CameraData cameraData;
		};

		/** scene information uniforms data for render descriptor set */
		struct SceneInfoUniforms {
			uint32_t  light_num;
			uint32_t  light_offset_pmf;
			uint32_t  light_offset_cdf;
			uint32_t  env_map;
			Math::vec4 padding_v0;
			Math::vec4 padding_v1;
			Math::vec4 padding_v2;
		};

		struct TableDist1D {
			TableDist1D() = default;
			TableDist1D(std::vector<float> const& f);
			std::vector<float> pmf;
			std::vector<float> cdf;
		};

		/**
		* Scene Updating
		* --------------------------------
		* 1. update scene geometry / materials
		* 2. update camera pose / setting
		*/
		struct SceneDataPack {
			// integrated geometry data
			std::unique_ptr<RHI::Buffer> vertex_buffer = nullptr;
			std::unique_ptr<RHI::Buffer> position_buffer = nullptr;
			std::unique_ptr<RHI::Buffer> index_buffer = nullptr;
			std::unique_ptr<RHI::Buffer> material_buffer = nullptr;
			std::unique_ptr<RHI::Buffer> light_buffer = nullptr;
			std::unique_ptr<RHI::Buffer> back_light_buffer = nullptr;
			std::unique_ptr<RHI::Buffer> sample_dist_buffer = nullptr;
			std::unique_ptr<RHI::Buffer> back_sample_dist_buffer = nullptr;
			// unbinded textures array to contain all textures
			std::vector<RHI::TextureView*> unbinded_textures = {};
			// cpu data
			std::vector<float>				position_buffer_cpu = {};		// position buffer cpu
			std::vector<float>				vertex_buffer_cpu = {};		// vertex buffer cpu
			std::vector<float>				sample_dist_buffer_cpu = {};		// sample dist buffer cpu
			std::vector<uint32_t>			index_buffer_cpu = {};		// index buffer cpu
			std::vector<GeometryDrawData>	geometry_buffer_cpu = {};		// geometries data
			std::vector<MaterialData>		material_buffer_cpu = {};		// material data
			std::vector<LightData>			light_buffer_cpu = {};		// light data
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
				GFX::MeshReference* meshReference;
				RHI::BLASInstance blasInstance;
				std::vector<uint32_t> geometry_indices;
				uint32_t primitiveType = 0;
			};

			struct EntityLightRecord {
				std::vector<uint32_t> lightIndices;
				std::vector<uint32_t> lightPowers;
				std::vector<TableDist1D> tableDists;
				Math::vec3 scaling;
			};

			bool geometryDirty = false;
			std::unordered_map<GFX::Mesh*, MeshRecord> mesh_record = {};
			std::unordered_map<GFX::Material*, uint32_t> material_record = {};
			std::unordered_map<RHI::TextureView*, uint32_t> textureview_record = {};
			std::unordered_map<Core::EntityHandle, MeshReferenceRecord> mesh_ref_record = {};
			std::unordered_map<Core::EntityHandle, EntityLightRecord> light_comp_record = {};

			SceneInfoUniforms sceneInfoUniform = {};
			TableDist1D lightTableDist = {};
		} sceneDataPack;
		/**
		* Custom Primitive definition
		*/
		struct CustomPrimitive {
			RHI::BLASCustomGeometry customGeometry;
			SceneDataPack::MeshRecord meshRecord = {};
		};
		struct CustomPrimitives {
			CustomPrimitive sphere;
		} customPrimitive;
		/**
		* Default setting definition
		*/
		struct DefaultResources {
			Core::GUID defaultMaterial;
			Core::GUID defaultTexture;
		} defaultResources;
		/** init non-scene resources */
		inline auto init(GFX::Scene& scene) noexcept -> void;
		/** invalid scene */
		inline auto invalidScene(GFX::Scene& scene) noexcept -> void;
		/** pack scene to scene data pack */
		inline auto packScene(GFX::Scene& scene) noexcept -> void;
		/** update main camera for the scene */
		inline auto updateCamera(GFX::TransformComponent const& transform, GFX::CameraComponent const& camera) noexcept -> void;
		/** update render data for the graph */
		inline auto updateRDGData(RDG::Graph* graph) noexcept -> void;

		struct {
			uint32_t width = 1280;
			uint32_t height = 720;
			uint32_t batchIdx = 0;
			uint32_t allBatch = 0;
		} state;
		/**
		* SRenderer info
		*/
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
			std::array<std::vector<RHI::BindGroupEntry>, MULTIFRAME_FLIGHTS_COUNT> set0_flights_resources = {};
			std::array<std::vector<RHI::BindGroupEntry>, MULTIFRAME_FLIGHTS_COUNT> set1_flights_resources = {};
		} commonDescData;

		RTCommon rtCommon;

		struct SceneDataBuffers {
			GFX::StructuredUniformBufferView<GlobalUniforms>					global_uniform_buffer;
			GFX::StructuredUniformBufferView<SceneInfoUniforms>					scene_info_buffer;
			GFX::StructuredArrayMultiStorageBufferView<GeometryDrawData>		geometry_buffer;
		} sceneDataBuffers;

		GlobalUniforms globalUniRecord;
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

	inline auto SRenderer::init(GFX::Scene& scene) noexcept -> void {
		RHI::Device* device = GFX::GFXManager::get()->rhiLayer->getDevice();

		// create common descriptor data
		RHI::ShaderStagesFlags stages =
			(uint32_t)RHI::ShaderStages::VERTEX
			| (uint32_t)RHI::ShaderStages::FRAGMENT
			| (uint32_t)RHI::ShaderStages::RAYGEN
			| (uint32_t)RHI::ShaderStages::CLOSEST_HIT
			| (uint32_t)RHI::ShaderStages::INTERSECTION
			| (uint32_t)RHI::ShaderStages::MISS
			| (uint32_t)RHI::ShaderStages::CALLABLE
			| (uint32_t)RHI::ShaderStages::ANY_HIT
			| (uint32_t)RHI::ShaderStages::COMPUTE;

		sceneDataBuffers.global_uniform_buffer = GFX::GFXManager::get()->createStructuredUniformBuffer<GlobalUniforms>();
		sceneDataBuffers.scene_info_buffer = GFX::GFXManager::get()->createStructuredUniformBuffer<SceneInfoUniforms>();

		// create default setting
		defaultResources.defaultTexture = GFX::GFXManager::get()->registerTextureResource("../Engine/Binaries/Runtime/textures/uv_checker.png");

		// create custom primitive records
		{
			customPrimitive.sphere.customGeometry = RHI::BLASCustomGeometry{
			RHI::AffineTransformMatrix{},
			std::vector<Math::bounds3>{Math::bounds3{{-1,-1,-1}, {+1, +1, +1} } },
				(uint32_t)RHI::BLASGeometryFlagBits::NO_DUPLICATE_ANY_HIT_INVOCATION,
			};
			customPrimitive.sphere.meshRecord.submesh_geometry.push_back(GeometryDrawData{
					0, // vertexOffset;
					0, // indexOffset;
					0, // materialID;
					0, // indexSize;
					0, // padding0;
					0, // padding1;
					0, // padding2;
					1.f, //oddNegativeScaling;
					RHI::AffineTransformMatrix{}, // geometryTransform
					RHI::AffineTransformMatrix{}, // geometryTransformInverse
				});
			customPrimitive.sphere.meshRecord.blas_desc = RHI::BLASDescriptor{};
			customPrimitive.sphere.meshRecord.blas_desc.customGeometries.push_back(customPrimitive.sphere.customGeometry);
			customPrimitive.sphere.meshRecord.blases = device->createBLAS(customPrimitive.sphere.meshRecord.blas_desc);
		}
		// pack scene
		packScene(scene);
	}

	struct SceneDataPackState {
		bool invalidVIPBuffer = false;
		bool invalidGeometryBuffer = false;
		bool dirtyGeometryBuffer = false;
		bool invalidLightBuffer = false;
		bool invalidAccumulation = false;
		bool invalidLightDistBuffer = false;
	};

	inline auto invalid_mesh_record(GFX::Mesh* mesh, SRenderer* srenderer, SceneDataPackState& state) noexcept ->
		std::unordered_map<GFX::Mesh*, SRenderer::SceneDataPack::MeshRecord>::iterator
	{
		srenderer->sceneDataPack.mesh_record[mesh] = {};
		auto meshRecord = srenderer->sceneDataPack.mesh_record.find(mesh);
		{	// create mesh record, add all data buffers
			uint32_t vertex_offset = srenderer->sceneDataPack.vertex_buffer_cpu.size();
			uint32_t position_offset = srenderer->sceneDataPack.position_buffer_cpu.size();
			srenderer->sceneDataPack.vertex_buffer_cpu.resize(vertex_offset + mesh->vertexBuffer_host.size / sizeof(float));
			memcpy(&(srenderer->sceneDataPack.vertex_buffer_cpu[vertex_offset]), mesh->vertexBuffer_host.data, mesh->vertexBuffer_host.size);
			if (srenderer->config.enableRayTracing) {
				srenderer->sceneDataPack.position_buffer_cpu.resize(position_offset + mesh->positionBuffer_host.size / sizeof(float));
				memcpy(&(srenderer->sceneDataPack.position_buffer_cpu[position_offset]), mesh->positionBuffer_host.data, mesh->positionBuffer_host.size);
			}
			uint32_t index_offset = srenderer->sceneDataPack.index_buffer_cpu.size();
			srenderer->sceneDataPack.index_buffer_cpu.resize(index_offset + mesh->indexBuffer_host.size / sizeof(uint32_t));
			memcpy(&(srenderer->sceneDataPack.index_buffer_cpu[index_offset]), mesh->indexBuffer_host.data, mesh->indexBuffer_host.size);
			// add all submeshes
			for (auto& submesh : mesh->submeshes) {
				SRenderer::GeometryDrawData geometry;
				geometry.vertexOffset = submesh.baseVertex + vertex_offset * sizeof(float) / SRenderer::vertexBufferLayout.arrayStride;
				geometry.indexOffset = submesh.offset + index_offset;
				geometry.materialID = 0;
				geometry.indexSize = submesh.size;
				geometry.geometryTransform = {};
				meshRecord->second.submesh_geometry.push_back(geometry);
				if (srenderer->config.enableRayTracing) {
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
			// set dirty of Vertex/Index/Position Buffer:
			state.invalidVIPBuffer = true;
		}
		return meshRecord;
	}

	inline auto luminance(Math::vec3 const& rgb) noexcept -> float {
		return rgb.x * float(0.212671) + rgb.y * float(0.715160) + rgb.z * float(0.072169);
	}

	inline auto invalid_game_object(GFX::GameObject* gameobject, GFX::Scene& scene, SRenderer* srenderer, SceneDataPackState& state) noexcept -> void {
		// fetch components
		GFX::TransformComponent* transform = gameobject->getEntity().getComponent<GFX::TransformComponent>();
		GFX::MeshReference* meshref = gameobject->getEntity().getComponent<GFX::MeshReference>();
		GFX::MeshRenderer* meshrenderer = gameobject->getEntity().getComponent<GFX::MeshRenderer>();
		GFX::LightComponent* lightComponenet = gameobject->getEntity().getComponent<GFX::LightComponent>();
		// we do not cares those game object has no relative components
		if (meshref == nullptr && meshrenderer == nullptr && lightComponenet == nullptr)
			return;
		// get transform
		Math::mat4 objectTransform;
		float oddScaling = 1.f;
		Math::vec3 scaling = Math::vec3{ 1,1,1 };
		{	// get mesh transform matrix
			GFX::GameObject* go = gameobject;
			GFX::TransformComponent* transform = go->getEntity().getComponent<GFX::TransformComponent>();
			objectTransform = transform->getTransform() * objectTransform;
			oddScaling *= transform->scale.x * transform->scale.y * transform->scale.z;
			scaling *= transform->scale;
			while (go->parent != Core::NULL_ENTITY) {
				go = scene.getGameObject(go->parent);
				GFX::TransformComponent* transform = go->getEntity().getComponent<GFX::TransformComponent>();
				objectTransform = transform->getTransform() * objectTransform;
				oddScaling *= transform->scale.x * transform->scale.y * transform->scale.z;
				scaling *= transform->scale;
			}
		}
		// geometry processing
		bool transformChanged = false;
		if (meshref && meshrenderer) {
			// mesh resource
			GFX::Mesh* mesh = meshref->mesh;
			auto meshRecord = srenderer->sceneDataPack.mesh_record.find(mesh);
			{	// insert mesh to the index / vertex buffer if it does not exist now
				if (meshRecord == srenderer->sceneDataPack.mesh_record.end())
					meshRecord = invalid_mesh_record(mesh, srenderer, state);
			}
			// mesh reference
			auto meshRefRecord = srenderer->sceneDataPack.mesh_ref_record.find(gameobject->entity);
			{	// add mesh reference if not exist now
				if (meshRefRecord == srenderer->sceneDataPack.mesh_ref_record.end()) {
					state.invalidGeometryBuffer = true;
				}
				for (auto idx : meshRefRecord->second.geometry_indices) {
					if (srenderer->sceneDataPack.geometry_buffer_cpu[idx].geometryTransform != objectTransform) {
						transformChanged = true;
						state.dirtyGeometryBuffer = true;
						state.invalidAccumulation = true;
						srenderer->sceneDataPack.geometry_buffer_cpu[idx].geometryTransform = objectTransform;
						srenderer->sceneDataPack.geometry_buffer_cpu[idx].geometryTransformInverse = Math::inverse(objectTransform);
						srenderer->sceneDataPack.geometry_buffer_cpu[idx].oddNegativeScaling = oddScaling >= 0 ? 1.f : -1.f;
					}
				}
				meshRefRecord->second.blasInstance.transform = objectTransform;
				srenderer->sceneDataPack.tlas_desc.instances.push_back(meshRefRecord->second.blasInstance);
			}
		}
		// light processing
		if (lightComponenet) {
			// if light is diffuse area light
			if (lightComponenet->type == GFX::LightComponent::LightType::DIFFUSE_AREA_LIGHT) {
				auto meshRefRecord = srenderer->sceneDataPack.mesh_ref_record.find(gameobject->entity);
				if (meshRefRecord != srenderer->sceneDataPack.mesh_ref_record.end()) {
					// now only diffuse area light is supported, we must have mesh record
					auto entityLightCompRecord = srenderer->sceneDataPack.light_comp_record.find(gameobject->entity);
					if (entityLightCompRecord == srenderer->sceneDataPack.light_comp_record.end()) {
					}
					for (size_t index : entityLightCompRecord->second.lightIndices) {
						if (srenderer->sceneDataPack.light_buffer_cpu[index].intensity != lightComponenet->intensity) {
							state.invalidAccumulation = true;
							state.invalidLightBuffer = true;
							srenderer->sceneDataPack.light_buffer_cpu[index].intensity = lightComponenet->intensity;
						}
						// if transform changed, we need to recompute surface area thing
						if (transformChanged && entityLightCompRecord->second.scaling != scaling) {
							state.invalidLightDistBuffer = true;
							auto& lightData = srenderer->sceneDataPack.light_buffer_cpu[index];
							if (meshref->customPrimitiveFlag == 0) {
								float totalArea = 0;
								for (uint32_t geoidx : meshRefRecord->second.geometry_indices) {
									//GFX::Mesh* meshPtr = meshRefRecord->second.meshReference->mesh;
									size_t primitiveNum = meshRefRecord->second.meshReference->mesh->indexBuffer_host.size / (3 * sizeof(uint32_t));
									uint32_t* indexBuffer = static_cast<uint32_t*>(meshRefRecord->second.meshReference->mesh->indexBuffer_host.data);
									float* vertexBuffer = static_cast<float*>(meshRefRecord->second.meshReference->mesh->vertexBuffer_host.data);
									size_t vertexStride = sizeof(SRenderer::InterleavedVertex) / sizeof(float);
									std::vector<float> areas;
									for (size_t i = 0; i < primitiveNum; ++i) {
										uint32_t i0 = indexBuffer[3 * i + 0];
										uint32_t i1 = indexBuffer[3 * i + 1];
										uint32_t i2 = indexBuffer[3 * i + 2];
										Math::vec3 const& pos0 = *(Math::vec3*)(&(vertexBuffer[i0 * vertexStride]));
										Math::vec3 const& pos1 = *(Math::vec3*)(&(vertexBuffer[i1 * vertexStride]));
										Math::vec3 const& pos2 = *(Math::vec3*)(&(vertexBuffer[i2 * vertexStride]));
										Math::vec3 v0 = Math::vec3(objectTransform * Math::vec4(pos0, 0));
										Math::vec3 v1 = Math::vec3(objectTransform * Math::vec4(pos1, 0));
										Math::vec3 v2 = Math::vec3(objectTransform * Math::vec4(pos2, 0));
										Math::vec3 const e1 = v1 - v0;
										Math::vec3 const e2 = v2 - v0;
										float area = Math::length(Math::cross(e1, e2)) / 2;
										areas.push_back(area);
										totalArea += area;
									}
									// create dist1D
									//entityLightCompRecord->second.tableDist = areas;
									entityLightCompRecord->second.tableDists[0] = SRenderer::TableDist1D{ areas };
									entityLightCompRecord->second.lightPowers[0] = totalArea * 3.1415926 * luminance(lightComponenet->intensity);
									srenderer->sceneDataPack.geometry_buffer_cpu[geoidx].surfaceArea = totalArea;
								}
							}
							// if mesh reference is pointing to a sphere mesh
							else if (meshref->customPrimitiveFlag == 1) {
								float const radius = Math::length(Math::vec3(objectTransform * Math::vec4(1, 0, 0, 1)) - Math::vec3(objectTransform * Math::vec4(0, 0, 0, 1)));
								float const surfaceArea = 4 * 3.1415926 * radius * radius;
								entityLightCompRecord->second.lightPowers[0] = surfaceArea * 3.1415926 * luminance(lightComponenet->intensity);
								srenderer->sceneDataPack.geometry_buffer_cpu[meshRefRecord->second.geometry_indices[0]].surfaceArea = surfaceArea;
							}
						}
					}
				}
			}
			else if (lightComponenet->type == GFX::LightComponent::LightType::CUBEMAP_ENV_MAP) {
				lightComponenet->texture;
			}
		}
	}

	inline auto invalidSceneLightingSetting(SRenderer* srenderer) noexcept -> void {
		srenderer->sceneDataPack.sample_dist_buffer_cpu.clear();
		std::vector<float> lightPowerArray = {};
		for (auto& lightCompRecord : srenderer->sceneDataPack.light_comp_record) {
			uint32_t i = 0;
			for (uint32_t subLightIdx : lightCompRecord.second.lightIndices) {
				auto& light = srenderer->sceneDataPack.light_buffer_cpu[subLightIdx];
				// push light power
				lightPowerArray.push_back(lightCompRecord.second.lightPowers[i]);
				// push light dist array
				if (lightCompRecord.second.tableDists.size() == 0) {
					continue;
				}
				else {
					light.sample_dist_size_0 = lightCompRecord.second.tableDists[i].pmf.size();
					light.sample_dist_offset_pmf_0 = srenderer->sceneDataPack.sample_dist_buffer_cpu.size();
					srenderer->sceneDataPack.sample_dist_buffer_cpu.insert(srenderer->sceneDataPack.sample_dist_buffer_cpu.end(),
						lightCompRecord.second.tableDists[i].pmf.begin(), lightCompRecord.second.tableDists[i].pmf.end());
					light.sample_dist_offset_cdf_0 = srenderer->sceneDataPack.sample_dist_buffer_cpu.size();
					srenderer->sceneDataPack.sample_dist_buffer_cpu.insert(srenderer->sceneDataPack.sample_dist_buffer_cpu.end(),
						lightCompRecord.second.tableDists[i].cdf.begin(), lightCompRecord.second.tableDists[i].cdf.end());
					++i;
				}
			}
		}
		srenderer->sceneDataPack.lightTableDist = SRenderer::TableDist1D{ lightPowerArray };
		srenderer->sceneDataPack.sceneInfoUniform.light_num = srenderer->sceneDataPack.lightTableDist.pmf.size();
		srenderer->sceneDataPack.sceneInfoUniform.light_offset_pmf = srenderer->sceneDataPack.sample_dist_buffer_cpu.size();
		srenderer->sceneDataPack.sample_dist_buffer_cpu.insert(srenderer->sceneDataPack.sample_dist_buffer_cpu.end(),
			srenderer->sceneDataPack.lightTableDist.pmf.begin(), srenderer->sceneDataPack.lightTableDist.pmf.end());
		srenderer->sceneDataPack.sceneInfoUniform.light_offset_cdf = srenderer->sceneDataPack.sample_dist_buffer_cpu.size();
		srenderer->sceneDataPack.sample_dist_buffer_cpu.insert(srenderer->sceneDataPack.sample_dist_buffer_cpu.end(),
			srenderer->sceneDataPack.lightTableDist.cdf.begin(), srenderer->sceneDataPack.lightTableDist.cdf.end());
		// also set pmf
		size_t i = 0;
		for (auto& lightCompRecord : srenderer->sceneDataPack.light_comp_record) {
			for (uint32_t subLightIdx : lightCompRecord.second.lightIndices) {
				auto& light = srenderer->sceneDataPack.light_buffer_cpu[subLightIdx];
				light.total_value = srenderer->sceneDataPack.lightTableDist.pmf[i++];
			}
		}
	}

	inline auto SRenderer::invalidScene(GFX::Scene& scene) noexcept -> void {
		// 
		RHI::Device* device = GFX::GFXManager::get()->rhiLayer->getDevice();
		sceneDataPack.tlas_desc.instances.clear();
		// for all mesh refs
		SceneDataPackState packstate;
		for (auto go_handle : scene.gameObjects) {
			invalid_game_object(scene.getGameObject(go_handle.first), scene, this, packstate);
		}
		static bool inited = false;
		if (!inited) {
			sceneDataPack.tlas = device->createTLAS(sceneDataPack.tlas_desc);
			inited = true;
		}
		sceneDataPack.back_tlas = sceneDataPack.tlas;
		sceneDataPack.tlas = device->createTLAS(sceneDataPack.tlas_desc);

		sceneDataPack.back_light_buffer = std::move(sceneDataPack.light_buffer);
		sceneDataPack.light_buffer = device->createDeviceLocalBuffer(
			sceneDataPack.light_buffer_cpu.data(),
			sceneDataPack.light_buffer_cpu.size() * sizeof(LightData),
			(uint32_t)RHI::BufferUsage::SHADER_DEVICE_ADDRESS
			| (uint32_t)RHI::BufferUsage::STORAGE);

		RHI::MultiFrameFlights* multiFrameFlights = GFX::GFXManager::get()->rhiLayer->getMultiFrameFlights();
		uint32_t fid = multiFrameFlights->getFlightIndex();

		// TODO
		commonDescData.set0_flights_resources[fid][5] = { 5, RHI::BindingResource{ {sceneDataPack.light_buffer.get(), 0, sceneDataPack.light_buffer->size()} } };
		commonDescData.set1_flights_resources[fid][0] = { 0, RHI::BindingResource{ sceneDataPack.tlas.get() } };

		sceneDataBuffers.scene_info_buffer.setStructure(sceneDataPack.sceneInfoUniform, fid);

		if (packstate.invalidLightDistBuffer) {
			invalidSceneLightingSetting(this);
			sceneDataPack.back_sample_dist_buffer = std::move(sceneDataPack.sample_dist_buffer);
			sceneDataPack.sample_dist_buffer = device->createDeviceLocalBuffer(
				sceneDataPack.sample_dist_buffer_cpu.data(),
				sceneDataPack.sample_dist_buffer_cpu.size() * sizeof(float),
				(uint32_t)RHI::BufferUsage::SHADER_DEVICE_ADDRESS
				| (uint32_t)RHI::BufferUsage::STORAGE);
		}

		commonDescData.set0_flights_resources[fid][6] = { 6, RHI::BindingResource{ {sceneDataPack.sample_dist_buffer.get(), 0, sceneDataPack.sample_dist_buffer->size()} } };

		if (packstate.invalidAccumulation) {
			state.batchIdx = 0;
		}
		if (sceneDataPack.geometry_buffer_cpu.size() > 0) {
			sceneDataBuffers.geometry_buffer.setStructure(sceneDataPack.geometry_buffer_cpu.data(), fid);
		}

		rtCommon.accumIDX = state.batchIdx;
	}

	auto packTexture(SRenderer* srenderer, Core::GUID guid) -> uint32_t {
		RHI::TextureView* texView = Core::ResourceManager::get()->getResource<GFX::Texture>(guid)->originalView.get();
		auto findTex = srenderer->sceneDataPack.textureview_record.find(texView);
		if (findTex == srenderer->sceneDataPack.textureview_record.end()) {
			uint32_t texID = srenderer->sceneDataPack.unbinded_textures.size();
			srenderer->sceneDataPack.unbinded_textures.push_back(texView);
			srenderer->sceneDataPack.textureview_record[texView] = texID;
			findTex = srenderer->sceneDataPack.textureview_record.find(texView);
		}
		return findTex->second;
	};

	inline auto SRenderer::packScene(GFX::Scene& scene) noexcept -> void {
		RHI::Device* device = GFX::GFXManager::get()->rhiLayer->getDevice();
		for (auto go_handle : scene.gameObjects) {
			auto* go = scene.getGameObject(go_handle.first);
			GFX::MeshReference* meshref = go->getEntity().getComponent<GFX::MeshReference>();
			GFX::MeshRenderer* meshrenderer = go->getEntity().getComponent<GFX::MeshRenderer>();
			GFX::LightComponent* lightComponenet = go->getEntity().getComponent<GFX::LightComponent>();
			if (lightComponenet && !meshref) {
				if (lightComponenet->type == GFX::LightComponent::LightType::CUBEMAP_ENV_MAP) {
					sceneDataPack.sceneInfoUniform.env_map = packTexture(this, lightComponenet->texture->guid);
				}
			}
			// if mesh reference is pointing to a triangle mesh
			if (!meshref) continue;
			GFX::Mesh* mesh = meshref->mesh;
			if (!mesh) continue;
			if (!meshrenderer) continue;
			// if do not have according mesh record, add the record
			auto meshRecord = sceneDataPack.mesh_record.find(mesh);
			if (meshRecord == sceneDataPack.mesh_record.end()) {
				sceneDataPack.geometryDirty = true;
				sceneDataPack.mesh_record[mesh] = {};
				meshRecord = sceneDataPack.mesh_record.find(mesh);
				{
					// create mesh record, add all data buffers
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
						geometry.vertexOffset = vertex_offset * sizeof(float) / SRenderer::vertexBufferLayout.arrayStride;
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
			float oddScaling = 1;
			Math::mat4 objectMat;
			Math::vec3 scaling = Math::vec3{ 1,1,1 };
			{	// get mesh transform matrix
				GFX::TransformComponent* transform = go->getEntity().getComponent<GFX::TransformComponent>();
				objectMat = transform->getTransform() * objectMat;
				oddScaling *= transform->scale.x * transform->scale.y * transform->scale.z;
				scaling *= transform->scale;
				while (go->parent != Core::NULL_ENTITY) {
					go = scene.getGameObject(go->parent);
					GFX::TransformComponent* transform = go->getEntity().getComponent<GFX::TransformComponent>();
					objectMat = transform->getTransform() * objectMat;
					oddScaling *= transform->scale.x * transform->scale.y * transform->scale.z;
					scaling *= transform->scale;
				}
				if (oddScaling != 0) oddScaling / std::abs(oddScaling);
			}
			// if do not have according mesh ref record, add the record
			auto meshRefRecord = sceneDataPack.mesh_ref_record.find(go->entity);
			if (meshRefRecord == sceneDataPack.mesh_ref_record.end()) {
				sceneDataPack.geometryDirty = true;
				sceneDataPack.mesh_ref_record[go->entity] = {};
				meshRefRecord = sceneDataPack.mesh_ref_record.find(go->entity);
				{	// create mesh record
					meshRefRecord->second.mesh = mesh;
					meshRefRecord->second.meshReference = meshref;
					uint32_t geometry_start = sceneDataPack.geometry_buffer_cpu.size();
					uint32_t offset = 0;
					for (auto& iter : meshRecord->second.submesh_geometry) {
						meshRefRecord->second.geometry_indices.push_back(sceneDataPack.geometry_buffer_cpu.size());
						GFX::Material* mat = meshrenderer->materials[offset++];
						// check whether material has been registered
						auto findMat = sceneDataPack.material_record.find(mat);
						if (findMat == sceneDataPack.material_record.end()) {
							uint32_t matID = sceneDataPack.material_buffer_cpu.size();
							MaterialData matData;

							Core::GUID baseTexGUID = mat->textures["base_color"].guid;
							Core::GUID normTexGUID = mat->textures["normal_bump"].guid;

							auto getTexID = [&](Core::GUID guid)->uint32_t {
								RHI::TextureView* normTexView = Core::ResourceManager::get()->getResource<GFX::Texture>(guid)->originalView.get();
								auto findTex = sceneDataPack.textureview_record.find(normTexView);
								if (findTex == sceneDataPack.textureview_record.end()) {
									uint32_t texID = sceneDataPack.unbinded_textures.size();
									sceneDataPack.unbinded_textures.push_back(normTexView);
									sceneDataPack.textureview_record[normTexView] = texID;
									findTex = sceneDataPack.textureview_record.find(normTexView);
								}
								return findTex->second;
							};
							matData.bsdf_id = mat->BxDF;
							matData.basecolor_opacity_tex = getTexID(baseTexGUID);
							matData.normal_bump_tex = getTexID(normTexGUID);
							sceneDataPack.material_buffer_cpu.push_back(matData);
							sceneDataPack.material_record[mat] = matID;
							findMat = sceneDataPack.material_record.find(mat);
						}
						GeometryDrawData geometrydata = iter;
						geometrydata.geometryTransform = objectMat;
						geometrydata.geometryTransformInverse = Math::inverse(objectMat);
						geometrydata.oddNegativeScaling = oddScaling >= 0 ? 1.f : -1.f;
						geometrydata.materialID = findMat->second;
						geometrydata.primitiveType = meshref->customPrimitiveFlag;
						geometrydata.lightID = lightComponenet ? 0 : 4294967295;
						sceneDataPack.geometry_buffer_cpu.emplace_back(geometrydata);
					}
					meshRefRecord->second.blasInstance.instanceCustomIndex = geometry_start;
					meshRefRecord->second.primitiveType = meshref->customPrimitiveFlag;
				}
			}
			// if do not have according light record, add the record
			if (lightComponenet) {
				auto entityLightCompRecord = sceneDataPack.light_comp_record.find(go->entity);
				if (entityLightCompRecord == sceneDataPack.light_comp_record.end()) {
					if (lightComponenet->type == GFX::LightComponent::LightType::CUBEMAP_ENV_MAP) {
						sceneDataPack.sceneInfoUniform.env_map = packTexture(this, lightComponenet->texture->guid);
					}
					else {
						sceneDataPack.light_comp_record[go->entity] = SceneDataPack::EntityLightRecord{};
						entityLightCompRecord = sceneDataPack.light_comp_record.find(go->entity);
						entityLightCompRecord->second.scaling = scaling;
						// if mesh reference is pointing to a triangle mesh
						if (meshref->customPrimitiveFlag == 0) {
							float totalArea = 0;
							for (uint32_t geoidx : meshRefRecord->second.geometry_indices) {
								sceneDataPack.geometry_buffer_cpu[geoidx].lightID = sceneDataPack.light_buffer_cpu.size();
								entityLightCompRecord->second.lightIndices.push_back(sceneDataPack.light_buffer_cpu.size());
								sceneDataPack.light_buffer_cpu.push_back(LightData{
									1, // type 1, sphere area light
									lightComponenet->intensity,
									geoidx,
									0,
									});
								//GFX::Mesh* meshPtr = meshRefRecord->second.meshReference->mesh;
								size_t primitiveNum = meshRefRecord->second.meshReference->mesh->indexBuffer_host.size / (3 * sizeof(uint32_t));
								uint32_t* indexBuffer = static_cast<uint32_t*>(meshRefRecord->second.meshReference->mesh->indexBuffer_host.data);
								float* vertexBuffer = static_cast<float*>(meshRefRecord->second.meshReference->mesh->vertexBuffer_host.data);
								size_t vertexStride = sizeof(InterleavedVertex) / sizeof(float);
								std::vector<float> areas;
								for (size_t i = 0; i < primitiveNum; ++i) {
									uint32_t i0 = indexBuffer[3 * i + 0];
									uint32_t i1 = indexBuffer[3 * i + 1];
									uint32_t i2 = indexBuffer[3 * i + 2];
									Math::vec3 const& pos0 = *(Math::vec3*)(&(vertexBuffer[i0 * vertexStride]));
									Math::vec3 const& pos1 = *(Math::vec3*)(&(vertexBuffer[i1 * vertexStride]));
									Math::vec3 const& pos2 = *(Math::vec3*)(&(vertexBuffer[i2 * vertexStride]));
									Math::vec3 v0 = Math::vec3(objectMat * Math::vec4(pos0, 0));
									Math::vec3 v1 = Math::vec3(objectMat * Math::vec4(pos1, 0));
									Math::vec3 v2 = Math::vec3(objectMat * Math::vec4(pos2, 0));
									Math::vec3 const e1 = v1 - v0;
									Math::vec3 const e2 = v2 - v0;
									float area = Math::length(Math::cross(e1, e2)) / 2;
									areas.push_back(area);
									totalArea += area;
								}
								// create dist1D
								//entityLightCompRecord->second.tableDist = areas;
								entityLightCompRecord->second.tableDists.emplace_back(TableDist1D{ areas });
								entityLightCompRecord->second.lightPowers.emplace_back(totalArea * 3.1415926 * luminance(lightComponenet->intensity));
								sceneDataPack.geometry_buffer_cpu[geoidx].surfaceArea = totalArea;
							}
						}
						// if mesh reference is pointing to a sphere mesh
						else if (meshref->customPrimitiveFlag == 1) {
							entityLightCompRecord->second.lightIndices.push_back(sceneDataPack.light_buffer_cpu.size());
							sceneDataPack.light_buffer_cpu.push_back(LightData{
								0, // type 0, sphere area light
								lightComponenet->intensity,
								meshRefRecord->second.geometry_indices[0],
								0,
								});
							float const radius = Math::length(Math::vec3(objectMat * Math::vec4(1, 0, 0, 1)) - Math::vec3(objectMat * Math::vec4(0, 0, 0, 1)));
							float const surfaceArea = 4 * 3.1415926 * radius * radius;
							entityLightCompRecord->second.lightPowers.emplace_back(surfaceArea * 3.1415926 * luminance(lightComponenet->intensity));
							sceneDataPack.geometry_buffer_cpu[meshRefRecord->second.geometry_indices[0]].surfaceArea = surfaceArea;
						}
					}
				}
			}
		}
		// TODO :: do light sampling precomputation
		{
			std::vector<float> lightPowerArray = {};
			for (auto& lightCompRecord : sceneDataPack.light_comp_record) {
				uint32_t i = 0;
				for (uint32_t subLightIdx : lightCompRecord.second.lightIndices) {
					auto& light = sceneDataPack.light_buffer_cpu[subLightIdx];
					// push light power
					lightPowerArray.push_back(lightCompRecord.second.lightPowers[i]);
					// push light dist array
					if (lightCompRecord.second.tableDists.size() == 0) {
						continue;
					}
					else {
						light.sample_dist_size_0 = lightCompRecord.second.tableDists[i].pmf.size();
						light.sample_dist_offset_pmf_0 = sceneDataPack.sample_dist_buffer_cpu.size();
						sceneDataPack.sample_dist_buffer_cpu.insert(sceneDataPack.sample_dist_buffer_cpu.end(),
							lightCompRecord.second.tableDists[i].pmf.begin(), lightCompRecord.second.tableDists[i].pmf.end());
						light.sample_dist_offset_cdf_0 = sceneDataPack.sample_dist_buffer_cpu.size();
						sceneDataPack.sample_dist_buffer_cpu.insert(sceneDataPack.sample_dist_buffer_cpu.end(),
							lightCompRecord.second.tableDists[i].cdf.begin(), lightCompRecord.second.tableDists[i].cdf.end());
						++i;
					}
				}
			}
			sceneDataPack.lightTableDist = TableDist1D{ lightPowerArray };
			sceneDataPack.sceneInfoUniform.light_num = sceneDataPack.lightTableDist.pmf.size();
			sceneDataPack.sceneInfoUniform.light_offset_pmf = sceneDataPack.sample_dist_buffer_cpu.size();
			sceneDataPack.sample_dist_buffer_cpu.insert(sceneDataPack.sample_dist_buffer_cpu.end(),
				sceneDataPack.lightTableDist.pmf.begin(), sceneDataPack.lightTableDist.pmf.end());
			sceneDataPack.sceneInfoUniform.light_offset_cdf = sceneDataPack.sample_dist_buffer_cpu.size();
			sceneDataPack.sample_dist_buffer_cpu.insert(sceneDataPack.sample_dist_buffer_cpu.end(),
				sceneDataPack.lightTableDist.cdf.begin(), sceneDataPack.lightTableDist.cdf.end());
			// also set pmf
			size_t i = 0;
			for (auto& lightCompRecord : sceneDataPack.light_comp_record) {
				for (uint32_t subLightIdx : lightCompRecord.second.lightIndices) {
					auto& light = sceneDataPack.light_buffer_cpu[subLightIdx];
					light.total_value = sceneDataPack.lightTableDist.pmf[i++];
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
			sceneDataPack.material_buffer = device->createDeviceLocalBuffer(
				sceneDataPack.material_buffer_cpu.data(),
				sceneDataPack.material_buffer_cpu.size() * sizeof(MaterialData),
				(uint32_t)RHI::BufferUsage::SHADER_DEVICE_ADDRESS
				| (uint32_t)RHI::BufferUsage::STORAGE);
			sceneDataPack.light_buffer = device->createDeviceLocalBuffer(
				sceneDataPack.light_buffer_cpu.data(),
				sceneDataPack.light_buffer_cpu.size() * sizeof(LightData),
				(uint32_t)RHI::BufferUsage::SHADER_DEVICE_ADDRESS
				| (uint32_t)RHI::BufferUsage::STORAGE);
			sceneDataPack.sample_dist_buffer = device->createDeviceLocalBuffer(
				sceneDataPack.sample_dist_buffer_cpu.data(),
				sceneDataPack.sample_dist_buffer_cpu.size() * sizeof(float),
				(uint32_t)RHI::BufferUsage::SHADER_DEVICE_ADDRESS
				| (uint32_t)RHI::BufferUsage::STORAGE);
			if (config.enableRayTracing) {
				sceneDataPack.position_buffer = device->createDeviceLocalBuffer(
					sceneDataPack.position_buffer_cpu.data(),
					sceneDataPack.position_buffer_cpu.size() * sizeof(float),
					(uint32_t)RHI::BufferUsage::VERTEX
					| (uint32_t)RHI::BufferUsage::SHADER_DEVICE_ADDRESS
					| (uint32_t)RHI::BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY
					| (uint32_t)RHI::BufferUsage::STORAGE);
			}
			sceneDataBuffers.geometry_buffer = GFX::GFXManager::get()->createStructuredArrayMultiStorageBuffer<GeometryDrawData>(
				sceneDataPack.geometry_buffer_cpu.size());
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
			if (iter.second.primitiveType == 0) {
				iter.second.blasInstance.blas = sceneDataPack.mesh_record[iter.second.mesh].blases.get();
			}
			else if (iter.second.primitiveType == 1) {
				iter.second.blasInstance.blas = customPrimitive.sphere.meshRecord.blases.get();
				iter.second.blasInstance.instanceShaderBindingTableRecordOffset = 1;
			}
			sceneDataPack.tlas_desc.instances.push_back(iter.second.blasInstance);
		}
		sceneDataPack.tlas_desc.allowRefitting = true;
		sceneDataPack.tlas = device->createTLAS(sceneDataPack.tlas_desc);

		//for (int i = 0; i < 2; ++i) {
		//	commonDescData.set1_flights_rt[i] = device->createBindGroup(RHI::BindGroupDescriptor{
		//		commonDescData.set1_layout_rt.get(),
		//		std::vector<RHI::BindGroupEntry>{
		//			{0,RHI::BindingResource{sceneDataPack.tlas.get()}},
		//			{1,RHI::BindingResource{rdgraph->getTexture("TracerTarget_Color")->texture->originalView.get()}},
		//	} });
		//	commonDescData.set1_flights_rt_array[i] = commonDescData.set1_flights_rt[i].get();
		//}


		// rebuild all bind groups
		for (int i = 0; i < 2; ++i) {
			commonDescData.set0_flights_resources[i] = std::vector<RHI::BindGroupEntry>{
							{0,RHI::BindingResource{sceneDataBuffers.global_uniform_buffer.getBufferBinding(i)}},
							{1,RHI::BindingResource{{sceneDataPack.vertex_buffer.get(), 0, sceneDataPack.vertex_buffer->size()}}},
							{2,RHI::BindingResource{{sceneDataPack.index_buffer.get(), 0, sceneDataPack.index_buffer->size()}}},
							{3,RHI::BindingResource{sceneDataBuffers.geometry_buffer.getBufferBinding(i)}},
							{4,RHI::BindingResource{{sceneDataPack.material_buffer.get(), 0, sceneDataPack.material_buffer->size()}}},
							{5,RHI::BindingResource{{sceneDataPack.light_buffer.get(), 0, sceneDataPack.light_buffer->size()}}},
							{6,RHI::BindingResource{{sceneDataPack.sample_dist_buffer.get(), 0, sceneDataPack.sample_dist_buffer->size()}}},
							{7,RHI::BindingResource{sceneDataBuffers.scene_info_buffer.getBufferBinding(i)}},
							{8,RHI::BindingResource{sceneDataPack.unbinded_textures,
								Core::ResourceManager::get()->getResource<GFX::Sampler>(GFX::GFXManager::get()->commonSampler.defaultSampler)->sampler.get()}},
			};

			commonDescData.set0_flights_resources[i] = std::vector<RHI::BindGroupEntry>{
				{0,RHI::BindingResource{sceneDataBuffers.global_uniform_buffer.getBufferBinding(i)}},
				{1,RHI::BindingResource{{sceneDataPack.vertex_buffer.get(), 0, sceneDataPack.vertex_buffer->size()}}},
				{2,RHI::BindingResource{{sceneDataPack.index_buffer.get(), 0, sceneDataPack.index_buffer->size()}}},
				{3,RHI::BindingResource{sceneDataBuffers.geometry_buffer.getBufferBinding(i)}},
				{4,RHI::BindingResource{{sceneDataPack.material_buffer.get(), 0, sceneDataPack.material_buffer->size()}}},
				{5,RHI::BindingResource{{sceneDataPack.light_buffer.get(), 0, sceneDataPack.light_buffer->size()}}},
				{6,RHI::BindingResource{{sceneDataPack.sample_dist_buffer.get(), 0, sceneDataPack.sample_dist_buffer->size()}}},
				{7,RHI::BindingResource{sceneDataBuffers.scene_info_buffer.getBufferBinding(i)}},
				{8,RHI::BindingResource{sceneDataPack.unbinded_textures,
					Core::ResourceManager::get()->getResource<GFX::Sampler>(GFX::GFXManager::get()->commonSampler.defaultSampler)->sampler.get()}},
			};
			commonDescData.set1_flights_resources[i] = std::vector<RHI::BindGroupEntry>{
				{0,RHI::BindingResource{sceneDataPack.tlas.get()}},
			};
		}
	}

	inline auto SRenderer::updateCamera(GFX::TransformComponent const& transform, GFX::CameraComponent const& camera) noexcept -> void {
		RHI::MultiFrameFlights* multiFrameFlights = GFX::GFXManager::get()->rhiLayer->getMultiFrameFlights();
		GlobalUniforms globalUni;

		{
			CameraData& camData = globalUni.cameraData;

			camData.nearZ = camera.near;
			camData.farZ = camera.far;

			camData.posW = transform.translation;
			camData.target = transform.translation + transform.getRotatedForward();

			camData.viewMat = Math::transpose(Math::lookAt(camData.posW, camData.target, Math::vec3(0, 1, 0)).m);
			camData.projMat = Math::transpose(Math::perspective(camera.fovy, camera.aspect, camera.near, camera.far).m);

			camData.viewProjMat = camData.viewMat * camData.projMat;
			camData.invViewProj = Math::inverse(camData.viewProjMat);

			// Ray tracing related vectors
			camData.focalDistance = 1;
			camData.aspectRatio = camera.aspect;
			camData.up = Math::vec3(0, 1, 0);
			camData.cameraW = Math::normalize(camData.target - camData.posW) * camData.focalDistance;
			camData.cameraU = Math::normalize(Math::cross(camData.cameraW, camData.up));
			camData.cameraV = Math::normalize(Math::cross(camData.cameraU, camData.cameraW));
			const float ulen = camData.focalDistance * std::tan(Math::radians(camera.fovy) * 0.5f) * camData.aspectRatio;
			camData.cameraU *= ulen;
			const float vlen = camData.focalDistance * std::tan(Math::radians(camera.fovy) * 0.5f);
			camData.cameraV *= vlen;

			camData.rectArea = 4 * ulen * vlen / (camData.focalDistance * camData.focalDistance);

		}
		if (globalUniRecord.cameraData.viewMat != globalUni.cameraData.viewMat || globalUniRecord.cameraData.projMat != globalUni.cameraData.projMat)
			state.batchIdx = 0;
		globalUniRecord = globalUni;
		sceneDataBuffers.global_uniform_buffer.setStructure(globalUni, multiFrameFlights->getFlightIndex());
	}

	inline auto SRenderer::updateRDGData(RDG::Graph* graph) noexcept -> void {
		uint32_t flightIdx = GFX::GFXManager::get()->rhiLayer->getMultiFrameFlights()->getFlightIndex();
		graph->renderData.setBindGroupEntries("CommonScene", &(commonDescData.set0_flights_resources[flightIdx]));
		graph->renderData.setBindGroupEntries("CommonRT", &(commonDescData.set1_flights_resources[flightIdx]));
		graph->renderData.setUInt("AccumIdx", state.batchIdx++);
		graph->renderData.setUVec2("TargetSize", { state.width , state.height });
		graph->renderData.setPtr("CameraData", &(globalUniRecord.cameraData));
		graph->renderData.setDelegate("IssueAllDrawcalls", [&, flightIdx = flightIdx](RDG::RenderData::DelegateData const& data) {
			if (sceneDataPack.geometry_buffer_cpu.size() > 0) {
				data.passEncoder.render->setIndexBuffer(sceneDataPack.index_buffer.get(),
					RHI::IndexFormat::UINT32_T, 0, sceneDataPack.index_buffer->size());
				data.passEncoder.render->setBindGroup(0, data.pipelinePass->bindgroups[0][flightIdx].get(), 0, 0);
				uint32_t geometry_idx = 0;
				for (auto& geometry : sceneDataPack.geometry_buffer_cpu) {
					data.passEncoder.render->pushConstants(&geometry_idx, (uint32_t)RHI::ShaderStages::VERTEX, 0, sizeof(uint32_t));
					data.passEncoder.render->drawIndexed(geometry.indexSize, 1, geometry.indexOffset, geometry.vertexOffset, 0);
					geometry_idx++;
				}
			}

			});
		graph->renderData.setDelegate("IssueDrawcalls_LightOnly", [&, flightIdx = flightIdx](RDG::RenderData::DelegateData const& data) {
			if (sceneDataPack.geometry_buffer_cpu.size() > 0) {
				data.passEncoder.render->setIndexBuffer(sceneDataPack.index_buffer.get(),
					RHI::IndexFormat::UINT32_T, 0, sceneDataPack.index_buffer->size());
				data.passEncoder.render->setBindGroup(0, data.pipelinePass->bindgroups[0][flightIdx].get(), 0, 0);
				uint32_t geometry_idx = 0;
				for (auto& geometry : sceneDataPack.geometry_buffer_cpu) {
					if (geometry.lightID != 4294967295) {
						data.passEncoder.render->pushConstants(&geometry_idx, (uint32_t)RHI::ShaderStages::VERTEX, 0, sizeof(uint32_t));
						data.passEncoder.render->drawIndexed(geometry.indexSize, 1, geometry.indexOffset, geometry.vertexOffset, 0);
					}
					geometry_idx++;
				}
			}
			});
	}

	SRenderer::TableDist1D::TableDist1D(std::vector<float> const& f) {
		pmf = f;
		cdf = std::vector<float>(f.size() + 1);
		cdf[0] = 0;
		for (int i = 0; i < (int)f.size(); i++)
			cdf[i + 1] = cdf[i] + pmf[i];
		float total = cdf.back();
		if (total > 0)
			for (int i = 0; i < (int)pmf.size(); i++) {
				pmf[i] /= total;
				cdf[i] /= total;
			}
		else {
			for (int i = 0; i < (int)pmf.size(); i++) {
				pmf[i] = float(1) / float(pmf.size());
				cdf[i] = float(i) / float(pmf.size());
			}
			cdf.back() = 1;
		}
		cdf.back() = 1;
	}

#pragma endregion
}