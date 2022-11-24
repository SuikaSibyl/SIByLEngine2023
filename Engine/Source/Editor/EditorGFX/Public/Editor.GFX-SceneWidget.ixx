module;
#include <vector>
#include <string>
#include <unordered_map>
#include <cstdint>
#include <typeinfo>
#include <imgui.h>
#include <imgui_internal.h>
export module Editor.GFX:SceneWidget;
import :InspectorWidget;
import Editor.Core;
import Editor.Framework;
import GFX.Resource;

namespace SIByL::Editor
{
	export struct GameObjectInspector : public CustomInspector {
		/** custom data to be parsed to each fragment */
		struct GameObjectData {
			GFX::GameObjectHandle handle;
			GFX::Scene* scene;
		} data;
		/** draw each fragments */
		virtual auto onDrawGui() noexcept -> void {
			for (auto& frag : fragmentSequence) {
				frag->onDrawGui(0, &data);
			}
		}
	};

	export struct SceneWidget :public Widget {
		/** virtual draw gui*/
		virtual auto onDrawGui() noexcept -> void override;
		/** draw a scene gameobject and its descents, return whether is deleted */
		auto drawNode(GFX::GameObjectHandle const& node) -> bool;
		/** bind scene to the widget */
		auto bindScene(GFX::Scene* scene) noexcept -> void { this->scene = scene; }
		/** scene binded to be shown on widget */
		GFX::Scene* scene = nullptr;
		/** the scene node that should be opened */
		GFX::GameObjectHandle forceNodeOpen = GFX::NULL_GO;
		/** the scene node that is inspected */
		GFX::GameObjectHandle inspected = GFX::NULL_GO;
		/** inspector widget to show gameobject detail */
		InspectorWidget* inspectorWidget = nullptr;
		/** game object inspector */
		GameObjectInspector gameobjectInspector = {};
	};

	auto SceneWidget::onDrawGui() noexcept -> void {
		ImGui::Begin("Scene", 0, ImGuiWindowFlags_MenuBar);
		ImGui::PushItemWidth(ImGui::GetFontSize() * -12);
		if (ImGui::BeginMenuBar()) {
			if (ImGui::Button("New")) {

			}
			if (ImGui::Button("Save")) {
				if (scene != nullptr) {
					std::string name = scene->name + ".scene";
					std::string path = ImGuiLayer::get()->rhiLayer->getRHILayerDescriptor().windowBinded->saveFile(nullptr, name);
					scene->serialize(path);
					scene->isDirty = false;
				}
			}
			if (ImGui::Button("Load")) {
				if (scene != nullptr) {
					std::string path = ImGuiLayer::get()->rhiLayer->getRHILayerDescriptor().windowBinded->openFile("scene");
					scene->deserialize(path);
					//scene->isDirty = false;
				}
			}
			//// Menu - Scene
			//if (ImGui::BeginMenu("Scene")) {
			//	// Menu - File - Load
			//	if (ImGui::MenuItem("New")) {
			//		hold_scene = MemNew<GFX::Scene>();
			//		bindScene(hold_scene.get());
			//	}
			//	ImGui::EndMenu();
			//}
			//// Menu - File
			//if (ImGui::BeginMenu("File")) {
			//	// Menu - File - Load
			//	if (ImGui::MenuItem("Load")) {
			//		std::string path = windowLayer->getWindow()->openFile("");
			//		hold_scene = MemNew<GFX::Scene>();
			//		bindScene(hold_scene.get());
			//		binded_scene->deserialize(path, assetLayer);
			//	}
			//	// Menu - File - Save
			//	bool should_save_as = false;
			//	if (ImGui::MenuItem("Save")) {
			//		if (currentPath == std::string()) should_save_as = true;
			//		else {}
			//	}
			//	// Menu - File - Save as
			//	if (ImGui::MenuItem("Save as") || should_save_as) {
			//		std::string path = windowLayer->getWindow()->saveFile("");
			//		binded_scene->serialize(path);
			//	}
			//	ImGui::EndMenu();
			//}
			ImGui::EndMenuBar();
		}
		ImGui::PopItemWidth();

		// Left-clock on blank space
		if (ImGui::IsMouseDown(0) && ImGui::IsWindowHovered()) {
			if (inspectorWidget) inspectorWidget->setEmpty();
			//if (viewport) viewport->selectedEntity = {};
		}
		// Right-click on blank space
		if (ImGui::BeginPopupContextWindow(0, 1, false)) {
			if (ImGui::MenuItem("Create Empty Entity") && scene) {
				scene->createGameObject(GFX::NULL_GO);
				ImGui::SetNextItemOpen(true, ImGuiCond_Always);
			}
			ImGui::EndPopup();
		}
		// Draw scene hierarchy
		if (scene) {
			std::vector<GFX::GameObjectHandle> rootHandles = {};
			for (auto iter = scene->gameObjects.begin(); iter != scene->gameObjects.end(); ++iter) {
				if (iter->second.parent == GFX::NULL_GO)
					rootHandles.push_back(iter->first);
			}
			for (auto handle : rootHandles)
				drawNode(handle);
		}
		ImGui::End();
	}

	auto SceneWidget::drawNode(GFX::GameObjectHandle const& node) -> bool {
		ImGui::PushID(node);
		GFX::TagComponent* tag = scene->getGameObject(node)->getEntity().getComponent<GFX::TagComponent>();
		ImGuiTreeNodeFlags node_flags = 0;
		if (scene->getGameObject(node)->children.size() == 0) node_flags |= ImGuiTreeNodeFlags_Leaf;
		if (node == forceNodeOpen) {
			ImGui::SetNextItemOpen(true, ImGuiCond_Always);
			forceNodeOpen = GFX::NULL_GO;
		}
		bool opened = ImGui::TreeNodeEx(tag->name.c_str(), node_flags);
		ImGuiID uid = ImGui::GetID(tag->name.c_str());
		ImGui::TreeNodeBehaviorIsOpen(uid);
		// Clicked
		if (ImGui::IsItemClicked()) {
			//ECS::Entity entity = binded_scene->tree.getNodeEntity(node);
			inspected = node;
			gameobjectInspector.data.scene = scene;
			gameobjectInspector.data.handle = node;
			if (inspectorWidget) gameobjectInspector.setInspectorWidget(inspectorWidget);
			//if (viewport) viewport->selectedEntity = entity;
		}
		// Right-click on blank space
		bool entityDeleted = false;
		if (ImGui::BeginPopupContextItem()) {
			if (ImGui::MenuItem("Create Empty Entity")) {
				scene->createGameObject(node);
				forceNodeOpen = node;
			}
			if (ImGui::MenuItem("Delete Entity"))
				entityDeleted = true;
			ImGui::EndPopup();
		}
		// If draged
		if (ImGui::BeginDragDropSource()) {
			ImGui::Text(tag->name.c_str());
			ImGui::SetDragDropPayload("SceneEntity", &node, sizeof(node));
			ImGui::EndDragDropSource();
		}
		// If dragged to
		if (ImGui::BeginDragDropTarget()) {
			if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("SceneEntity")) {
				GFX::GameObjectHandle* dragged_handle = (uint64_t*)payload->Data;
				//binded_scene->tree.moveNode(*dragged_handle, node);
			}
			ImGui::EndDragDropTarget();
		}
		// Opened
		if (opened) {
			ImGui::NextColumn();
			for (int i = 0; i < scene->getGameObject(node)->children.size(); i++) {
				drawNode(scene->getGameObject(node)->children[i]);
			}
			ImGui::TreePop();
		}
		ImGui::PopID();
		if (entityDeleted) {
			bool isParentOfInspected = false;
			GFX::GameObjectHandle inspected_parent = inspected;
			while (inspected_parent != GFX::NULL_GO) {
				inspected_parent = scene->getGameObject(inspected_parent)->parent;
				if (node == node) { isParentOfInspected = true; break; } // If set ancestor as child, no movement;
			}
			if (node == inspected || isParentOfInspected) {
				if (inspectorWidget) inspectorWidget->setEmpty();
				//if (viewport) viewport->selectedEntity = {};
				inspected = 0;
			}
			scene->removeGameObject(node);
			return true;
		}
		return false;
	}

}