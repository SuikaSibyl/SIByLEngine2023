#pragma once
#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>
#include "../../../Application/Public/SE.Application.Config.h"
#include "SE.SRenderer-RTCommon.hpp"
#include "SE.SRenderer-RACommon.hpp"
#include <Resource/SE.Core.Resource.hpp>
#include <ECS/SE.Core.ECS.hpp>
#include <Misc/SE.Core.Misc.hpp>
#include <SE.Math.Geometric.hpp>
#include <SE.Math.Misc.hpp>
#include <SE.RHI.hpp>
#include <SE.GFX.hpp>
#include <SE.RDG.hpp>
#include <Passes/ComputePasses/SE.SRenderer-Skinning.hpp>

namespace SIByL {
SE_EXPORT struct SRenderer {
  SRenderer() { singleton = this; }
  /**
   * Config for SRenderer
   */
  struct Configuration {
    bool enableRayTracing = true;  // whether use ray tracing
    bool enableUV2 = true;
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
    Math::vec3 baseOrDiffuseColor;
    uint32_t flags;
    Math::vec3 specularColor;
    uint32_t bsdf_id = 1;
    Math::vec3 emissiveColor;
    uint32_t domain = 1;

    float opacity;
    float roughness;
    float metalness;
    float normalTextureScale;

    float occlusionStrength;
    float alphaCutoff;
    float transmissionFactor;
    int baseOrDiffuseTextureIndex;

    int metalRoughOrSpecularTextureIndex;
    int emissiveTextureIndex;
    int normalTextureIndex;
    int occlusionTextureIndex;

    int transmissionTextureIndex;
    int padding1;
    int padding2;
    int padding3;
  };

  struct PolymorphicLightInfo {
    Math::vec3 center;
    uint32_t colorTypeAndFlags;
    uint32_t databyte0;    // geometry id / direction1(oct-encoded)
    uint32_t databyte1;    // index id / direction2(oct-encoded)
    uint32_t scalars;      // 2x float16
    uint32_t logRadiance;  // uint16
    uint32_t iesProfileIndex;
    uint32_t primaryAxis;              // oct-encoded
    uint32_t cosConeAngleAndSoftness;  // 2x float16
    uint32_t shadowMapIndex;           // shadow map index
  };

  /** mesh / geometry draw call data */
  struct GeometryDrawData {
    uint32_t vertexOffset;
    uint32_t indexOffset;
    uint32_t materialID;
    uint32_t indexSize;
    float surfaceArea;
    uint32_t lightID;
    uint32_t primitiveType;
    struct BitField3 {
      uint32_t oddNegativeScaling : 1;
      uint32_t gameObjectID : 31;
    } bitfeild3;
    RHI::AffineTransformMatrix geometryTransform = {};
    RHI::AffineTransformMatrix geometryTransformInverse = {};
  };

  /**
   * Universal camera data for both rasterizer & raytracer.
   * Definition & annotations are modified from Falcor:
   * @ref: https://github.com/NVIDIAGameWorks/Falcor
   */
  struct CameraData {
    Math::mat4 viewMat;  ///< Camera view matrix.
    Math::mat4 invViewMat;     ///< Inversed camera view matrix.
    Math::mat4 projMat;  ///< Camera projection matrix.
    Math::mat4 invProjMat;  ///< Camera projection matrix.
    Math::mat4 viewProjMat;          ///< Camera view-projection matrix.
    Math::mat4 invViewProj;          ///< Camera inverse view-projection matrix.
    Math::mat4 viewProjMatNoJitter;  ///< Camera view-projection matrix. No
                                     ///< jittering is applied!
    Math::mat4 projMatNoJitter;  ///< Camera projection matrix. No jittering is
                                 ///< applied!

    Math::vec3 posW;      ///< Camera world-space position.
    float focalLength;    ///< Camera focal length in mm. Default is 59 degree
                          ///< vertical, 90 horizontal FOV at 16:9 aspect ratio.
    Math::vec3 prevPosW;  ///< Camera world-space position associated to
                          ///< previous frame.
    float rectArea;       ///< Recrtangles area A at z=1 plane.
    Math::vec3 up;        ///< Camera world-space up vector.
    float aspectRatio;  ///< Camera film frame aspect ratio, equal to frameWidth
                        ///< / frameHeight
    Math::vec3 target;  ///< Camera target point in world-space.
    float nearZ;        ///< Camera near plane.
    Math::vec3
        cameraU;  ///< Camera base vector U. Normalized it indicates the right
                  ///< image plane vector. The length is dependent on the FOV.
    float farZ;   ///< Camera far plane.
    Math::vec3
        cameraV;    ///< Camera base vector V. Normalized it indicates the up
                    ///< image plane vector. The length is dependent on the FOV.
    float jitterX;  ///< Eventual camera jitter along the x axis expressed as a
                    ///< subpixel offset divided by screen width (positive value
                    ///< shifts the image right).
    Math::vec3
        cameraW;  ///< Camera base vector W. Normalized it indicates the forward
                  ///< direction. The length is the camera focal distance.
    float jitterY;  ///< Eventual camera jitter along the y axis expressed as a
                    ///< subpixel offset divided by screen height (positive
                    ///< value shifts the image up).

    float frameHeight;  ///< Camera film frame height in mm. 24 is the height of
                        ///< a 35mm film
    float frameWidth;   ///< Camera film frame width in mm.  42 2/3 is the width
                        ///< assuming 24mm height and a 16:9 aspect ratio
    float focalDistance;   ///< Camera focal distance in scene units.
    float apertureRadius;  ///< Camera aperture radius in scene units.
    float shutterSpeed;    ///< Camera shutter speed in seconds.
    float ISOSpeed;        ///< Camera film speed based on ISO standards.
    float _padding1;
    float _padding2;

    Math::vec2 clipToWindowScale;
    Math::vec2 clipToWindowBias;

  };

  /** global uniforms data for render descriptor set */
  struct alignas(64) GlobalUniforms {
    CameraData cameraData;
  };

  /** scene information uniforms data for render descriptor set */
  struct SceneInfoUniforms {
    uint32_t light_num;
    uint32_t light_offset_pmf;
    uint32_t light_offset_cdf;
    uint32_t env_map;
    Math::vec4 padding_v0;
    Math::vec4 padding_v1;
    Math::vec4 padding_v2;
  };

  /**
   * Scene Updating
   * --------------------------------
   * 1. update scene geometry / materials
   * 2. update camera pose / setting
   */
  struct SceneDataPack {
    // integrated geometry data
    GFX::HostDeviceBuffer<float> vertex_buffer;
    GFX::HostDeviceBuffer<float> position_buffer;
    GFX::HostDeviceBuffer<float> uv2_buffer;
    GFX::HostDeviceBuffer<uint32_t> index_buffer;
    GFX::HostDeviceBuffer<MaterialData> material_buffer;
    GFX::HostDeviceBuffer<GeometryDrawData> geometry_buffer;
    // integrated skeleton animation data
    GFX::HostDeviceBuffer<float> skelpos_buffer;
    GFX::HostDeviceBuffer<float> skelnormal_buffer;
    GFX::HostDeviceBuffer<float> skelweights_buffer;
    GFX::HostDeviceBuffer<uint32_t> skeloffset_buffer;
    GFX::HostDeviceBuffer<uint32_t> skeljoints_buffer;
    GFX::HostDeviceBuffer<Math::mat4> skeltransform_buffer;
    GFX::HostDeviceBuffer<Math::mat4> skelinvtrans_buffer;
    // integrated skeleton animation data
    GFX::HostDeviceBuffer<PolymorphicLightInfo> light_buffer;
    // unbinded textures array to contain all textures
    std::vector<RHI::TextureView*> unbinded_textures = {};
    std::vector<RHI::Sampler*> unbinded_samplers = {};
    // tlas of the scene for ray tracing
    RHI::TLASDescriptor tlas_desc = {};
    GFX::DualResource<RHI::TLAS> tlas = {};


    struct MeshRecord {
      std::vector<GeometryDrawData> submesh_geometry;
      RHI::BLASDescriptor blas_desc;
      std::unique_ptr<RHI::BLAS> blases = nullptr;
      std::unique_ptr<RHI::BLAS> blases_back = nullptr;
      bool need_rebuild = false;
    };

    struct MeshReferenceRecord {
      GFX::MeshReference* meshReference;
      // meshes and geometry indices for all LoDs
      std::vector<GFX::Mesh*> meshes;
      std::vector<std::vector<uint32_t>> geometry_indices;
      std::vector<RHI::BLASInstance> blasInstance;
      uint32_t primitiveType = 0;
      uint32_t lod_shown = 0;
      std::vector<uint32_t> basic_drawcall_offset;
      std::vector<uint32_t> bsdf_drawcall_offset;
    };

    struct EntityLightRecord {
      std::vector<uint32_t> lightIndices;
      std::vector<uint32_t> lightPowers;
      Math::vec3 scaling;
    };
    
    bool geometryDirty = false;
    std::unordered_map<GFX::Mesh*, MeshRecord> mesh_record = {};
    std::unordered_map<GFX::Material*, uint32_t> material_record = {};
    std::unordered_map<GFX::Texture*, uint32_t> texture_record = {};
    std::unordered_map<Core::EntityHandle, MeshReferenceRecord> mesh_ref_record = {};
    std::unordered_map<Core::EntityHandle, EntityLightRecord> light_comp_record = {};
    std::unordered_map<Core::EntityHandle, uint32_t> skinning_record = {};
    SkinningPass skinning_pass;

    SceneInfoUniforms sceneInfoUniform = {};
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
  auto init(GFX::Scene& scene) noexcept -> void;
  /** invalid scene */
  auto invalidScene(GFX::Scene& scene) noexcept -> void;
  /** update main camera for the scene */
  auto updateCamera(GFX::TransformComponent const& transform,
                    GFX::CameraComponent const& camera,
                    Math::ivec2 const& viewport) noexcept -> void;
  /** update render data for the graph */
  auto updateRDGData(RDG::Graph* graph) noexcept -> void;

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
    std::array<std::vector<RHI::BindGroupEntry>, MULTIFRAME_FLIGHTS_COUNT>
        set0_flights_resources = {};
    std::array<std::vector<RHI::BindGroupEntry>, MULTIFRAME_FLIGHTS_COUNT>
        set1_flights_resources = {};
  } commonDescData;

  RTCommon rtCommon;
  RACommon raCommon;
  Core::Timer* timer;

  struct SceneDataBuffers {
    GFX::StructuredUniformBufferView<GlobalUniforms> global_uniform_buffer;
    GFX::StructuredUniformBufferView<SceneInfoUniforms> scene_info_buffer;
  } sceneDataBuffers;

  struct SceneStatisticsData {
    Math::bounds3 aabb;
  } statisticsData;

  GlobalUniforms globalUniRecord;

  static SRenderer* singleton;
};
}  // namespace SIByL