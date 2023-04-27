module;
#include <array>
#include <vector>
#include <memory>
#include <optional>
#include <compare>
#include <typeinfo>
#include <filesystem>
#include <unordered_map>
#include <functional>
#include "../../../Application/Public/SE.Application.Config.h"
export module SE.SRenderer:RACommon;
import SE.Core.Resource;
import SE.Math.Geometric;
import SE.RHI;
import SE.GFX;

namespace SIByL
{
	export struct RACommon {

		RACommon() {
			singleton = this;
		}

		static auto get() noexcept -> RACommon* { return singleton; }

		struct DirectionalLightInfo {
			Math::mat4 transform;
			uint32_t lightID;
		};
		std::optional<DirectionalLightInfo> mainDirectionalLight = std::nullopt;

		struct ShadowmapInfo {
			Math::mat4 viewProj;
		};
		std::vector<ShadowmapInfo> shadowmapData;


		struct alignas(64) CascadeShadowmapData {
			Math::mat4 cascade_transform_0;
			Math::mat4 cascade_transform_1;
			Math::mat4 cascade_transform_2;
			Math::mat4 cascade_transform_3;
			Math::vec4 cascade_depths;
		} mainLightCSM;
		GFX::StructuredUniformBufferView<CascadeShadowmapData> csm_info_device;

		std::vector<Math::mat4> cascade_views;
		Math::vec4 cascade_distances;

		Math::bounds3 sceneAABB;

		//indexed 
		struct DrawIndexedIndirectCmd {
			uint32_t    indexCount;
			uint32_t    instanceCount;
			uint32_t    firstIndex;
			int32_t     vertexOffset;
			uint32_t    firstInstance;
		};

		struct DrawIndexedIndirectEX {
			uint32_t    indexCount;
			uint32_t    instanceCount;
			uint32_t    firstIndex;
			int32_t     vertexOffset;
			uint32_t    firstInstance;
			uint32_t    geometryID;
		};

		//non indexed
		struct DrawIndirectCmd {
			uint32_t    vertexCount;
			uint32_t    instanceCount;
			uint32_t    firstVertex;
			uint32_t    firstInstance;
		};

		struct IndirectDrawcall {
			uint64_t    offset;
			uint32_t    drawCount;
		};

		struct DrawcallData {
			std::vector<DrawIndexedIndirectEX> opaque_drawcalls_host;
			std::vector<DrawIndexedIndirectEX> alphacut_drawcalls_host;
			std::unordered_map<uint32_t, std::vector<DrawIndexedIndirectEX>> bsdf_drawcalls_host;


			IndirectDrawcall opaque_drawcall;
			IndirectDrawcall alphacut_drawcall;
			std::unordered_map<uint32_t, IndirectDrawcall> bsdfs_drawcalls;

			std::vector<DrawIndexedIndirectEX> all_drawcall_host;
			GFX::Buffer* all_drawcall_device = nullptr;

			auto buildIndirectDrawcalls() noexcept -> void {
				// release the previous buffer
				if (all_drawcall_device)
					all_drawcall_device->release();
				all_drawcall_host.clear();
				bsdfs_drawcalls.clear();
				// opaque pass
				opaque_drawcall = IndirectDrawcall{ (uint32_t)(all_drawcall_host.size() * sizeof(DrawIndexedIndirectEX)), (uint32_t)opaque_drawcalls_host.size() };
				all_drawcall_host.insert(all_drawcall_host.end(), opaque_drawcalls_host.begin(), opaque_drawcalls_host.end());
				// opaque pass
				alphacut_drawcall = IndirectDrawcall{ (uint32_t)(all_drawcall_host.size() * sizeof(DrawIndexedIndirectEX)), (uint32_t)alphacut_drawcalls_host.size() };
				all_drawcall_host.insert(all_drawcall_host.end(), alphacut_drawcalls_host.begin(), alphacut_drawcalls_host.end());
				// bsdf passes
				for (auto const& iter : bsdf_drawcalls_host) {
					std::vector<DrawIndexedIndirectEX> const& drawcall_host = iter.second;
					IndirectDrawcall bsdf_drawcall = IndirectDrawcall{ (uint32_t)(all_drawcall_host.size() * sizeof(DrawIndexedIndirectEX)), (uint32_t)drawcall_host.size() };
					all_drawcall_host.insert(all_drawcall_host.end(), drawcall_host.begin(), drawcall_host.end());
					bsdfs_drawcalls[iter.first] = bsdf_drawcall;
				}
				// create buffer
				//all_drawcall_device
				Core::GUID guid = Core::ResourceManager::get()->requestRuntimeGUID<GFX::Buffer>();
				GFX::GFXManager::get()->registerBufferResource(guid, all_drawcall_host.data(), all_drawcall_host.size() * sizeof(DrawIndexedIndirectEX),
					(uint32_t)RHI::BufferUsage::STORAGE | (uint32_t)RHI::BufferUsage::INDIRECT);
				all_drawcall_device = Core::ResourceManager::get()->getResource<GFX::Buffer>(guid);
			}

		} structured_drawcalls;

		GFX::CameraComponent const* mainCamera;
		struct {
			Math::mat4 view;
		} mainCameraInfo;
	private:
		static RACommon* singleton;

	};

	RACommon* RACommon::singleton = nullptr;
}