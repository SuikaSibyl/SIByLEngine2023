module;
#include <vector>
#include <string>
#include <unordered_map>
#include <cstdint>
#include <typeinfo>
#include <memory>
#include <functional>
#include <concepts>
#include <imgui.h>
#include <imgui_internal.h>
export module SE.Editor.GFX:ResourceViewer;
import :TextureFragment;
import SE.Core.ECS;
import SE.Core.Log;
import SE.Core.Resource;
import SE.Image;
import SE.GFX.Core;
import SE.Editor.Core;

namespace SIByL::Editor
{
	export struct ResourceElucidator {
		/** override draw gui */
		virtual auto onDrawGui(Core::GUID guid) noexcept -> void = 0;
	};

	export struct ResourceViewer {
		/** override draw gui */
		auto onDrawGui(char const* type, Core::GUID guid) noexcept -> void {
			for (auto const& pair : elucidatorMaps) {
				if (strcmp(pair.first, type) == 0) {
					pair.second->onDrawGui(guid);
					break;
				}
			}
		}

		template <class T>
		requires std::derived_from<T, ResourceElucidator>
		auto registerElucidator(char const* type) noexcept -> void {
			elucidatorMaps[type] = std::make_unique<T>();
		}

	private:
		std::unordered_map<char const*, std::unique_ptr<ResourceElucidator>> elucidatorMaps;
	};

	export struct TextureElucidator :public ResourceElucidator {
		/** override draw gui */
		virtual auto onDrawGui(Core::GUID guid) noexcept -> void {
			GFX::Texture* tex = Core::ResourceManager::get()->getResource<GFX::Texture>(guid);
			float const texw = (float)tex->texture->width();
			float const texh = (float)tex->texture->height();
			float const wa = ImGui::GetContentRegionAvail().x / texw;
			float const ha = ImGui::GetContentRegionAvail().y / texh;
			float a = std::min(1.f, std::min(wa, ha));
			ImGui::Image(
				Editor::TextureUtils::getImGuiTexture(guid)->getTextureID(),
				{ a * texw,a * texh }, 
				{ 0,0 }, { 1, 1 });
			ImGui::Text("Texture: ");
			ImGui::Text((std::string("- GUID: ") + std::to_string(tex->guid)).c_str());
			ImGui::Text((std::string("- name: ") + tex->texture->getName()).c_str());
			ImGui::Text((std::string("- width: ") + std::to_string(tex->texture->width())).c_str());
			ImGui::Text((std::string("- height: ") + std::to_string(tex->texture->height())).c_str());
		}
	};
}