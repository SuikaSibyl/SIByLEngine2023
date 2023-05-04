module;
#include <vector>
#include <string>
#include <unordered_map>
#include <cstdint>
#include <typeinfo>
#include <imgui.h>
#include <memory>
#include <utility>
#include <tuple>
#include <functional>
#include <imgui_internal.h>
export module SE.Editor.GFX:SceneWidget;
import :InspectorWidget;
import SE.Core.ECS;
import SE.Editor.Core;
import SE.GFX;

namespace SIByL::Editor
{
	export struct GameObjectInspector : public CustomInspector {
		/** custom data to be parsed to each fragment */
		struct GameObjectData {
			GFX::GameObjectHandle handle;
			GFX::Scene* scene;
		} data;

		struct IComponentOperator {
			virtual auto addComponent(Core::Entity&) -> void = 0;
			virtual auto getComponent(Core::Entity&) -> void* = 0;
		};
		template <class T>
		struct ComponentOperator :IComponentOperator {
			virtual auto addComponent(Core::Entity& entity) -> void override {
				entity.addComponent<T>();
			}
			virtual auto getComponent(Core::Entity& entity) -> void* override {
				return (void*)entity.getComponent<T>();
			}
		};
		std::vector<std::pair<std::string, std::unique_ptr<IComponentOperator>>> componentsRegister = {};
		template <class T>
		inline auto registerComponent(std::string const& name) -> void {
			componentsRegister.emplace_back(std::pair<std::string, std::unique_ptr<IComponentOperator>>(name, std::make_unique<ComponentOperator<T>>()));
		}

		/** draw each fragments */
		virtual auto onDrawGui() noexcept -> void {
			for (auto& frag : fragmentSequence) {
				frag->onDrawGui(0, &data);
			}
			{	// add component
				ImGui::Separator();
				ImVec2 contentRegionAvailable = ImGui::GetContentRegionAvail();
				ImVec2 buttonSize(200, 30);
				ImGui::SetCursorPosX(contentRegionAvailable.x / 2 - 100 + 20);
				if (ImGui::Button(" Add Component", buttonSize))
					ImGui::OpenPopup("AddComponent");
				if (ImGui::BeginPopup("AddComponent")) {
					Core::Entity entity = data.scene->getGameObject(data.handle)->getEntity();
					for (auto& pair : componentsRegister) {
						if (pair.second.get()->getComponent(entity) == nullptr) {
							if (ImGui::MenuItem(pair.first.c_str())) {
								pair.second.get()->addComponent(entity);
								ImGui::CloseCurrentPopup();
							}
						}
					}
					ImGui::EndPopup();
				}
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
					if (path != "") {
						scene->serialize(path);
						scene->isDirty = false;
					}
				}
			}
			if (ImGui::Button("Load")) {
				if (scene != nullptr) {
					std::string path = ImGuiLayer::get()->rhiLayer->getRHILayerDescriptor().windowBinded->openFile("scene");
					if (path != "")
						scene->deserialize(path);
					//scene->isDirty = false;
				}
			}
			if (ImGui::BeginMenu("Import")) {
				// Menu - File - Load
				if (ImGui::MenuItem("glTF 2.0 (.glb/.gltf)")) {
					std::string path = ImGuiLayer::get()->rhiLayer->getRHILayerDescriptor().windowBinded->openFile(".gltf");
					//GFX::SceneNodeLoader_obj::loadSceneNode(path, *scene, SRenderer::meshLoadConfig);
				}
				if (ImGui::MenuItem("Wavefront (.obj)")) {
					std::string path = ImGuiLayer::get()->rhiLayer->getRHILayerDescriptor().windowBinded->openFile(".obj");
					GFX::SceneNodeLoader_obj::loadSceneNode(path, *scene, GFX::GFXManager::get()->config.meshLoaderConfig);
				}
				if (ImGui::MenuItem("FBX (.fbx)")) {
					std::string path = ImGuiLayer::get()->rhiLayer->getRHILayerDescriptor().windowBinded->openFile(".fbx");
					GFX::SceneNodeLoader_assimp::loadSceneNode(path, *scene, GFX::GFXManager::get()->config.meshLoaderConfig);
				}
				ImGui::EndMenu();
			}
			if (ImGui::BeginMenu("Export")) {

				ImGui::EndMenu();
			}
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