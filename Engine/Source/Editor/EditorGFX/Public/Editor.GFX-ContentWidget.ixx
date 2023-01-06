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
#include <imgui_internal.h>
export module SE.Editor.GFX:ContentWidget;
import :InspectorWidget;
import :TextureFragment;
import :ResourceViewer;
import SE.Core.Resource;
import SE.Image;
import SE.GFX.Core;
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
	};

	auto ContentWidget::onDrawGui() noexcept -> void {
        if (!ImGui::Begin("Content Browser", 0)) {
            ImGui::End();
            return;
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
							for (auto& directoryEntry : std::filesystem::directory_iterator(currentDirectory)) {
								ImGui::TableSetColumnIndex(iconIdx);
								const auto& path = directoryEntry.path();
								std::string pathString = path.string();
								auto relativePath = std::filesystem::relative(directoryEntry.path(), root);
								std::string relativePathString = relativePath.string();
								std::string filenameString = relativePath.filename().string();

								if (filenameString == ".adb" || filenameString == ".iadb")
									continue;

								ImGui::PushID(filenameString.c_str());
								// If is directory
								ImGuiTexture* icon = Editor::TextureUtils::getImGuiTexture(icons.file);
								if (directoryEntry.is_directory()) {
									icon = Editor::TextureUtils::getImGuiTexture(icons.folder);
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
		std::unique_ptr<Image::Image<Image::COLOR_R8G8B8A8_UINT>> img = nullptr;
		icons.back = Core::hashGUID("../Engine/Binaries/Runtime/icons/back.png");
		icons.folder = Core::hashGUID("../Engine/Binaries/Runtime/icons/folder.png");
		icons.file = Core::hashGUID("../Engine/Binaries/Runtime/icons/file.png");
		img = Image::PNG::fromPNG("../Engine/Binaries/Runtime/icons/back.png");
		GFX::GFXManager::get()->registerTextureResource(icons.back, img.get());
		Core::ResourceManager::get()->getResource<GFX::Texture>(icons.back)->texture->setName("editor_icon_back");
		img = Image::PNG::fromPNG("../Engine/Binaries/Runtime/icons/folder.png");
		GFX::GFXManager::get()->registerTextureResource(icons.folder, img.get());
		Core::ResourceManager::get()->getResource<GFX::Texture>(icons.folder)->texture->setName("editor_icon_folder");
		img = Image::PNG::fromPNG("../Engine/Binaries/Runtime/icons/file.png");
		GFX::GFXManager::get()->registerTextureResource(icons.file, img.get());
		Core::ResourceManager::get()->getResource<GFX::Texture>(icons.file)->texture->setName("editor_icon_file");
	}

}