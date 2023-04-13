module;
#include <vector>
#include <string>
#include <unordered_map>
#include <cstdint>
#include <typeinfo>
#include <imgui.h>
#include <algorithm>
#include <functional>
#include <compare>
#include <filesystem>
#include <initializer_list>
#include <imgui_internal.h>
export module SE.Editor.GFX:ContentWidget;
import :InspectorWidget;
import :TextureFragment;
import :ResourceViewer;
import SE.Core.Resource;
import SE.Image;
import SE.GFX;
import SE.Editor.Core;

namespace SIByL::Editor
{
	export struct ContentWidget :public Widget {
		/** draw gui*/
		virtual auto onDrawGui() noexcept -> void override;
        /** contents source */
        enum struct Source {
            FileSystem,
            RuntimeResources,
        } source = Source::FileSystem;
		/** resources registry */
		struct ResourceRegistry {
			std::string resourceName;
			Core::GUID	resourceIcon;
			std::vector<std::string> possibleExtensions;
			using GUIDFinderFn = std::function<Core::GUID(Core::ORID)>;
			GUIDFinderFn guidFinder = nullptr;
			using ResourceLoadFn = std::function<Core::GUID(char const*)>;
			ResourceLoadFn resourceLoader = nullptr;
			inline auto matchExtensions(std::string const& ext) const noexcept -> bool {
				bool match = false;
				for (auto const& e : possibleExtensions)
					if (e == ext) match = true;
				return match;
			}
		};
		std::vector<ResourceRegistry> resourceRegistries;
		/** register a resource type */
		template <Core::StructResource T>
		auto registerResource(Core::GUID icon, std::initializer_list<std::string> extensions, ResourceRegistry::GUIDFinderFn guidFinder = nullptr, ResourceRegistry::ResourceLoadFn resourceLoader = nullptr) noexcept -> void;
		/** icon resources */
		struct IconResource {
			Core::GUID back;
			Core::GUID folder;
			Core::GUID file;
			Core::GUID mesh;
			Core::GUID scene;
			Core::GUID material;
			Core::GUID shader;
			Core::GUID image;
			Core::GUID video;
		} icons;
		/* register icon resources*/
		auto reigsterIconResources() noexcept -> void;
        /** current directory */
		std::filesystem::path currentDirectory = "./content";
		/** current resource */
		struct {
			char const* type = nullptr;
			std::vector<Core::GUID> const* GUIDs = nullptr;
		} runtimeResourceInfo;
		/** inspector widget to show gameobject detail */
		InspectorWidget* inspectorWidget = nullptr;

		struct modalState {
			bool showModal = false;
			std::string path;
			ResourceRegistry const* entry_ptr;
		} modal_state;
	};

	template <Core::StructResource T>
	auto ContentWidget::registerResource(Core::GUID icon, std::initializer_list<std::string> extensions, ResourceRegistry::GUIDFinderFn guidFinder, ResourceRegistry::ResourceLoadFn resourceLoader) noexcept -> void {
		resourceRegistries.emplace_back(ResourceRegistry{ typeid(T).name(), icon, std::vector<std::string>(extensions), guidFinder, resourceLoader });
	}

	auto ContentWidget::onDrawGui() noexcept -> void {
        if (!ImGui::Begin("Content Browser", 0)) {
            ImGui::End();
            return;
        }

		if (modal_state.showModal)
			ImGui::OpenPopup("Warning");
		if (ImGui::BeginPopupModal("Warning")) {
			ImGui::Text("Selected resource not registered in resource libarary!");
			if (ImGui::Button("OK", ImVec2(120, 0))) {
				ImGui::CloseCurrentPopup();
				modal_state.showModal = false;
			}
			ImGui::SameLine();
			if (ImGui::Button("Load", ImVec2(120, 0))) {
				if (modal_state.entry_ptr->resourceLoader == nullptr) {
					Core::LogManager::Error("GFXEditor :: Resource loading of this type is not supported yet!");
				}
				else {
					Core::GUID guid = modal_state.entry_ptr->resourceLoader(modal_state.path.c_str());
					inspectorWidget->setCustomDraw(std::bind(
						&(ResourceViewer::onDrawGui),
						&(inspectorWidget->resourceViewer),
						modal_state.entry_ptr->resourceName.c_str(),
						guid));
					ImGui::CloseCurrentPopup();
					modal_state.showModal = false;
				}
			}
		}

		{
			ImGuiWindowFlags window_flags = ImGuiWindowFlags_None;
			ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 5.0f);
			ImGui::BeginChild("ChildL", ImVec2(ImGui::GetContentRegionAvail().x * 0.2f, 0), true, window_flags);
			// Select content browser source
			{
				ImGui::AlignTextToFramePadding();
				bool const selectFileExplorer = ImGui::TreeNodeEx("File Exploror", ImGuiTreeNodeFlags_Leaf);
				if (ImGui::IsItemClicked()) {
					source = Source::FileSystem;
				}
				if (selectFileExplorer) {
					ImGui::TreePop();
				}
				if (ImGui::TreeNodeEx("Runtime Resources")) {
					auto const& resourcePool = Core::ResourceManager::get()->getResourcePool();
					for (auto const& pair : resourcePool) {
						bool const selectResourceType = ImGui::TreeNodeEx(pair.first, ImGuiTreeNodeFlags_Leaf);
						if (ImGui::IsItemClicked()) {
							runtimeResourceInfo.type = pair.first;
							runtimeResourceInfo.GUIDs = &pair.second->getAllGUID();
							source = Source::RuntimeResources;
						}
						if (selectResourceType) {
							ImGui::TreePop();
						}
					}
					ImGui::TreePop();
				}
			}
			ImGui::EndChild();
			ImGui::PopStyleVar();
		}
		ImGui::SameLine();

		// Child 2: rounded border
		{
			ImGuiWindowFlags window_flags = ImGuiWindowFlags_None;
			ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 5.0f);
			ImGui::BeginChild("ChildR", ImVec2(0, 0), true, window_flags);
			{
				if (source == Source::FileSystem) {
					static std::filesystem::path root("./content");
					int iconCount = 0;
					if (root.compare(currentDirectory))
						++iconCount;
					//// calculate column number
					static float padding = 16.0f * ImGuiLayer::get()->getDPI();
					static float thumbnailSize = 64.f * ImGuiLayer::get()->getDPI();
					float cellSize = thumbnailSize + padding;
					float panelWidth = ImGui::GetContentRegionAvail().x;
					int columnCount = (int)(panelWidth / cellSize);
					if (columnCount < 1)
						columnCount = 1;
					// Do Columns
					int subdirCount = std::distance(
						std::filesystem::directory_iterator(currentDirectory),
						std::filesystem::directory_iterator{});
					ImGui::Text(("File Explorer Path: " + currentDirectory.string()).c_str());
					if (ImGui::BeginTable("FileSystemBrowser", columnCount, ImGuiTableFlags_Resizable | ImGuiTableFlags_NoBordersInBody)) {
						ImGui::TableNextRow();
						{
							int iconIdx = 0;
							// for back
							if (root.compare(currentDirectory)) {
								ImGui::TableSetColumnIndex(iconIdx);
								ImGui::PushID("Back");
								ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
								ImGuiTexture* icon = Editor::TextureUtils::getImGuiTexture(icons.back);
								ImGui::ImageButton(icon->getTextureID(), { thumbnailSize,thumbnailSize }, { 0,0 }, { 1,1 });
								ImGui::PopStyleColor();
								if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
									currentDirectory = currentDirectory.parent_path();
								}
								ImGui::TextWrapped("       Back");
								ImGui::PopID();
								// set table position
								++iconIdx;
								if (iconIdx >= columnCount) {
									iconIdx = 0;
									ImGui::TableNextRow();
								}
							}
							// for each subdir
							std::vector<std::filesystem::directory_entry> path_sorted = {};
							std::vector<std::filesystem::directory_entry> folder_pathes = {};
							std::vector<std::vector<std::filesystem::directory_entry>> other_pathes(resourceRegistries.size());
							std::vector<std::filesystem::directory_entry> unkown_pathes = {};
							for (auto& directoryEntry : std::filesystem::directory_iterator(currentDirectory)) {
								const auto& path = directoryEntry.path();
								std::string filenameExtension = path.extension().string();
								if (directoryEntry.is_directory()) {
									folder_pathes.push_back(directoryEntry);
								}
								else {
									size_t idx = 0;
									for (auto const& entry : resourceRegistries) {
										if (entry.matchExtensions(filenameExtension)) {
											other_pathes[idx].push_back(directoryEntry);
											break;
										}
										++idx;
									}
								}
							}
							path_sorted.insert(path_sorted.end(), folder_pathes.begin(), folder_pathes.end());
							for (auto& vec : other_pathes) 
								path_sorted.insert(path_sorted.end(), vec.begin(), vec.end());
							path_sorted.insert(path_sorted.end(), unkown_pathes.begin(), unkown_pathes.end());
							for (auto& directoryEntry : path_sorted) {
								ImGui::TableSetColumnIndex(iconIdx);
								const auto& path = directoryEntry.path();
								std::string pathString = path.string();
								auto relativePath = std::filesystem::relative(directoryEntry.path(), root);
								std::string relativePathString = relativePath.string();
								std::string filenameString = relativePath.filename().string();
								std::string filenameExtension = relativePath.extension().string();

								ImGui::PushID(filenameString.c_str());
								// If is directory
								ImGuiTexture* icon = Editor::TextureUtils::getImGuiTexture(icons.file);
								ResourceRegistry const* entry_ptr = nullptr;
								if (directoryEntry.is_directory()) {
									icon = Editor::TextureUtils::getImGuiTexture(icons.folder);
								}
								else {
									for (auto const& entry : resourceRegistries) {
										if (entry.matchExtensions(filenameExtension)) {
											icon = Editor::TextureUtils::getImGuiTexture(entry.resourceIcon);
											entry_ptr = &entry;
											break;
										}
									}
								}

								ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
								ImGui::ImageButton(icon->getTextureID(), { thumbnailSize,thumbnailSize }, { 0,0 }, { 1,1 });

								//if (ImGui::BeginDragDropSource()) {
								//	ImGui::Text("Dragged");

								//	const wchar_t* itemPath = relativePath.c_str();
								//	ImGui::SetDragDropPayload("ASSET", itemPath, (wcslen(itemPath) + 1) * sizeof(wchar_t));
								//	ImGui::EndDragDropSource();
								//}
								ImGui::PopStyleColor();
								if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
									if (directoryEntry.is_directory()) {
										currentDirectory /= path.filename();
									}
									else if (entry_ptr && entry_ptr->guidFinder) {
										Core::ORID orid = Core::ResourceManager::get()->database.findResourcePath(path.string().c_str());
										if (orid != Core::ORID_NONE) {
											Core::GUID guid = entry_ptr->guidFinder(orid);
											inspectorWidget->setCustomDraw(std::bind(
												&(ResourceViewer::onDrawGui),
												&(inspectorWidget->resourceViewer),
												entry_ptr->resourceName.c_str(),
												guid));
										}
										else {
											modal_state.showModal = true;
											modal_state.path = path.string();
											modal_state.entry_ptr = entry_ptr;
										}
									}
								}
								ImGui::TextWrapped(filenameString.c_str());
								ImGui::PopID();
								// set table position
								++iconIdx;
								if (iconIdx >= columnCount) {
									iconIdx = 0;
									ImGui::TableNextRow();
								}
							}
						}
						ImGui::EndTable();
					}
				}
				else if (source == Source::RuntimeResources) {
					for (auto const& GUID : *runtimeResourceInfo.GUIDs) {
						std::string GUIDstr = std::to_string(GUID);
						if (GUIDstr.length() < 20) {
							GUIDstr = std::string(20 - GUIDstr.length(), '0') + GUIDstr;
						}
						bool const selectResource = ImGui::TreeNodeEx(
							(GUIDstr + " :: " +
							Core::ResourceManager::get()->getResourceName(runtimeResourceInfo.type, GUID)).c_str(),
							ImGuiTreeNodeFlags_Leaf);
						if (ImGui::IsItemClicked()) {
							inspectorWidget->setCustomDraw(std::bind(
								&(ResourceViewer::onDrawGui),
								&(inspectorWidget->resourceViewer),
								runtimeResourceInfo.type,
								GUID));
						}
						if (selectResource) {
							ImGui::TreePop();
						}
					}
				}
			}
			ImGui::EndChild();
			ImGui::PopStyleVar();
		}
        ImGui::End();
	}

	auto ContentWidget::reigsterIconResources() noexcept -> void {
		icons.back		= GFX::GFXManager::get()->registerTextureResource("../Engine/Binaries/Runtime/icons/back.png");
		icons.folder	= GFX::GFXManager::get()->registerTextureResource("../Engine/Binaries/Runtime/icons/folder.png");
		icons.file		= GFX::GFXManager::get()->registerTextureResource("../Engine/Binaries/Runtime/icons/file.png");
		icons.image		= GFX::GFXManager::get()->registerTextureResource("../Engine/Binaries/Runtime/icons/image.png");
		icons.material	= GFX::GFXManager::get()->registerTextureResource("../Engine/Binaries/Runtime/icons/material.png");
		icons.mesh		= GFX::GFXManager::get()->registerTextureResource("../Engine/Binaries/Runtime/icons/mesh.png");
		icons.scene		= GFX::GFXManager::get()->registerTextureResource("../Engine/Binaries/Runtime/icons/scene.png");
		icons.shader	= GFX::GFXManager::get()->registerTextureResource("../Engine/Binaries/Runtime/icons/shader.png");
		icons.video		= GFX::GFXManager::get()->registerTextureResource("../Engine/Binaries/Runtime/icons/video.png");
		Core::ResourceManager::get()->getResource<GFX::Texture>(icons.back	  )->texture->setName("editor_icon_back");
		Core::ResourceManager::get()->getResource<GFX::Texture>(icons.folder  )->texture->setName("editor_icon_folder");
		Core::ResourceManager::get()->getResource<GFX::Texture>(icons.file	  )->texture->setName("editor_icon_file");
		Core::ResourceManager::get()->getResource<GFX::Texture>(icons.image	  )->texture->setName("editor_icon_image");
		Core::ResourceManager::get()->getResource<GFX::Texture>(icons.material)->texture->setName("editor_icon_material");
		Core::ResourceManager::get()->getResource<GFX::Texture>(icons.mesh	  )->texture->setName("editor_icon_mesh");
		Core::ResourceManager::get()->getResource<GFX::Texture>(icons.scene	  )->texture->setName("editor_icon_scene");
		Core::ResourceManager::get()->getResource<GFX::Texture>(icons.shader  )->texture->setName("editor_icon_shader");
		Core::ResourceManager::get()->getResource<GFX::Texture>(icons.video	  )->texture->setName("editor_icon_videor");
	}

}