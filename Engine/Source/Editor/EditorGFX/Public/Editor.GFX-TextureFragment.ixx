module;
#include <vector>
#include <string>
#include <unordered_map>
#include <cstdint>
#include <typeinfo>
#include <imgui.h>
#include <imgui_internal.h>
export module SE.Editor.GFX:TextureFragment;
import :Utils;
import SE.Core.Resource;
import SE.RHI;
import SE.GFX;
import SE.Editor.Core;

namespace SIByL::Editor
{
	export struct TextureUtils {
		static auto getImGuiTexture(Core::GUID guid) noexcept -> ImGuiTexture* {
			auto& pool = ImGuiLayer::get()->ImGuiTexturePool;
			auto iter = pool.find(guid);
			if (iter == pool.end()) {
				GFX::Texture* texture = Core::ResourceManager::get()->getResource<GFX::Texture>(guid);
				pool.insert({ guid, ImGuiLayer::get()->createImGuiTexture(
					Core::ResourceManager::get()->getResource<GFX::Sampler>(GFX::GFXManager::get()->commonSampler.defaultSampler)->sampler.get(),
					texture->getSRV(0, 1, 0, 1),
					RHI::TextureLayout::SHADER_READ_ONLY_OPTIMAL) });
				return pool[guid].get();
			}
			else {
				return iter->second.get();
			}
		}
	};

	export struct TextureFragment :public Fragment {
		/** virtual draw gui*/
		virtual auto onDrawGui(uint32_t flags, void* data) noexcept -> void {

		}
	};
}