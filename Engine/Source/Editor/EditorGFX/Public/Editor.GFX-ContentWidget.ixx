module;
#include <vector>
#include <string>
#include <unordered_map>
#include <cstdint>
#include <typeinfo>
#include <imgui.h>
#include <algorithm>
#include <compare>
#include <filesystem>
#include <imgui_internal.h>
export module Editor.GFX:ContentWidget;
import :InspectorWidget;
import :TextureFragment;
import Core.Resource;
import Image.Color;
import Image.Image;
import Image.FileFormat;
import GFX.Resource;
import GFX.GFXManager;
import Editor.Core;
import Editor.Framework;

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

        bool openWidget = true;
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
	};

	auto ContentWidget::onDrawGui() noexcept -> void {
        if (!ImGui::Begin("Content Browser", &openWidget)) {
            ImGui::End();
            return;
        }
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(2, 2));
        if (ImGui::BeginTable("split", 2, ImGuiTableFlags_BordersOuter | ImGuiTableFlags_Resizable)) {
            // Text and Tree nodes are less high than framed widgets, using AlignTextToFramePadding() we add vertical spacing to make the tree lines equal high.
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
			// Select content browser source
			{
				ImGui::AlignTextToFramePadding();
				ImGui::Text("File Exploror");
				ImGui::Text("Runtime Resources");
			}
            //bool node_open = ImGui::TreeNode("Object", "%s_%u", 1, 1);
            ImGui::TableSetColumnIndex(1);
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
            ImGui::EndTable();
        }
        ImGui::PopStyleVar();
        ImGui::End();
	}

	auto ContentWidget::reigsterIconResources() noexcept -> void {
		std::unique_ptr<Image::Image<Image::COLOR_R8G8B8A8_UINT>> img = nullptr;
		icons.back = Core::hashGUID("../Engine/Binaries/Runtime/icons/back.png");
		icons.folder = Core::hashGUID("../Engine/Binaries/Runtime/icons/folder.png");
		icons.file = Core::hashGUID("../Engine/Binaries/Runtime/icons/file.png");
		img = Image::PNG::fromPNG("../Engine/Binaries/Runtime/icons/back.png");
		GFX::GFXManager::get()->registerTextureResource(icons.back, img.get());
		img = Image::PNG::fromPNG("../Engine/Binaries/Runtime/icons/folder.png");
		GFX::GFXManager::get()->registerTextureResource(icons.folder, img.get());
		img = Image::PNG::fromPNG("../Engine/Binaries/Runtime/icons/file.png");
		GFX::GFXManager::get()->registerTextureResource(icons.file, img.get());
	}

}