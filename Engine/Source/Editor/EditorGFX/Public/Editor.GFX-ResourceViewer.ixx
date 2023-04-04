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
import :Utils;
import :TextureFragment;
import SE.Core.ECS;
import SE.Core.Log;
import SE.Core.Resource;
import SE.RHI;
import SE.Image;
import SE.GFX;
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

	export struct MeshElucidator :public ResourceElucidator {
		/** override draw gui */
		virtual auto onDrawGui(Core::GUID guid) noexcept -> void override {
			onDrawGui_GUID(guid);
		}

		static auto onDrawGui_GUID(Core::GUID guid) noexcept -> void {
			GFX::Mesh* mesh = Core::ResourceManager::get()->getResource<GFX::Mesh>(guid);
			onDrawGui_PTR(mesh);
		}

		static auto onDrawGui_PTR(GFX::Mesh* mesh) noexcept -> void {
			const int index_size = mesh->primitiveState.stripIndexFormat == RHI::IndexFormat::UINT16_t ? sizeof(uint16_t) : sizeof(uint32_t);
			const int index_count = mesh->indexBufferInfo.size / index_size;
			const int primitive_count = index_count / 3;
			if (ImGui::TreeNode("Vertex Buffer")) {
				ImGui::BulletText((std::string("Size (bytes): ") + std::to_string(mesh->vertexBufferInfo.size)).c_str());
				if (ImGui::TreeNode("Buffer Layout")) {
					ImGui::BulletText((std::string("Array Stride: ") + std::to_string(mesh->vertexBufferLayout.arrayStride)).c_str());
					ImGui::BulletText((std::string("Step Mode: ") + to_string(mesh->vertexBufferLayout.stepMode)).c_str());
					if (ImGui::TreeNode("Attributes Layouts:")) {
						for (auto& item : mesh->vertexBufferLayout.attributes) {
							ImGui::BulletText((to_string(item.format) + std::string(" | OFF: ") +
								std::to_string(item.offset) + std::string(" | LOC: ") +
								std::to_string(item.shaderLocation)).c_str());
						}
						ImGui::TreePop();
					}
					ImGui::TreePop();
				}
				ImGui::TreePop();
			}
			if (ImGui::TreeNode("Index Buffer")) {
				ImGui::BulletText((std::string("Size (bytes): ") + std::to_string(mesh->indexBufferInfo.size)).c_str());
				ImGui::BulletText((std::string("Index Count: ") + std::to_string(index_count)).c_str());
				ImGui::TreePop();
			}
			if (ImGui::TreeNode("Primitive Status")) {
				ImGui::BulletText((std::string("Primitive Count: ") + std::to_string(primitive_count)).c_str());
				ImGui::BulletText((std::string("Topology: ") + to_string(mesh->primitiveState.topology)).c_str());

				ImGui::TreePop();
			}


			//float const texw = (float)tex->texture->width();
			//float const texh = (float)tex->texture->height();
			//float const wa = std::max(1.f, ImGui::GetContentRegionAvail().x - 15) / texw;
			//float const ha = 1;
			//float a = std::min(1.f, std::min(wa, ha));
			//ImGui::Image(
			//	Editor::TextureUtils::getImGuiTexture(tex->guid)->getTextureID(),
			//	{ a * texw,a * texh },
			//	{ 0,0 }, { 1, 1 });
			//ImGui::Text("Texture: ");
			//ImGui::Text((std::string("- GUID: ") + std::to_string(tex->guid)).c_str());
			//ImGui::Text((std::string("- name: ") + tex->texture->getName()).c_str());
			//ImGui::Text((std::string("- width: ") + std::to_string(tex->texture->width())).c_str());
			//ImGui::Text((std::string("- height: ") + std::to_string(tex->texture->height())).c_str());
		}
	};


	export struct TextureElucidator :public ResourceElucidator {
		/** override draw gui */
		virtual auto onDrawGui(Core::GUID guid) noexcept -> void override {
			onDrawGui_GUID(guid);
		}

		static auto onDrawGui_GUID(Core::GUID guid) noexcept -> void {
			GFX::Texture* tex = Core::ResourceManager::get()->getResource<GFX::Texture>(guid);
			onDrawGui_PTR(tex);
		}

		static auto onDrawGui_PTR(GFX::Texture* tex) noexcept -> void {
			float const texw = (float)tex->texture->width();
			float const texh = (float)tex->texture->height();
			float const wa = std::max(1.f, ImGui::GetContentRegionAvail().x - 15) / texw;
			float const ha = 1;
			float a = std::min(1.f, std::min(wa, ha));
			ImGui::Image(
				Editor::TextureUtils::getImGuiTexture(tex->guid)->getTextureID(),
				{ a * texw,a * texh },
				{ 0,0 }, { 1, 1 });
			ImGui::Text("Texture: ");
			ImGui::Text((std::string("- GUID: ") + std::to_string(tex->guid)).c_str());
			ImGui::Text((std::string("- name: ") + tex->texture->getName()).c_str());
			ImGui::Text((std::string("- width: ") + std::to_string(tex->texture->width())).c_str());
			ImGui::Text((std::string("- height: ") + std::to_string(tex->texture->height())).c_str());
		}
	};

	export struct MaterialElucidator :public ResourceElucidator {
		/** override draw gui */
		virtual auto onDrawGui(Core::GUID guid) noexcept -> void override {
			onDrawGui_GUID(guid);
		}

		static auto onDrawGui_GUID(Core::GUID guid) noexcept -> void {
			GFX::Material* mat = Core::ResourceManager::get()->getResource<GFX::Material>(guid);
			onDrawGui_PTR(mat);
		}

		/** override draw gui */
		static auto onDrawGui_PTR(GFX::Material* material) noexcept -> void {
			ImGui::BulletText(("Name: " + material->name).c_str());
			ImGui::BulletText(("Path: " + material->path).c_str());
			if (ImGui::TreeNode("Textures:")) {
				uint32_t id = 0;
				for (auto& [name, texture] : material->textures) {
					ImGui::PushID(id);
					if (ImGui::TreeNode((("Texture - " + std::to_string(id) + " - " + name).c_str()))) {
						TextureElucidator::onDrawGui_GUID(texture);
						ImGui::TreePop();
					}
					++id;
					ImGui::PopID();
				}
				ImGui::TreePop();
			}
			ImGui::BulletText(("Emissive: " + std::to_string(material->isEmissive)).c_str());
		}
	};
}