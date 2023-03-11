module;
#include <vector>
#include <string>
#include <unordered_map>
#include <cstdint>
#include <typeinfo>
#include <imgui.h>
#include <algorithm>
#include <imgui_internal.h>
export module SE.Editor.GFX:InspectorFragments;
import :Utils;
import :SceneWidget;
import :ResourceViewer;
import SE.Math.Geometric;
import SE.RHI;
import SE.GFX.Core;
import SE.Editor.Core;

namespace SIByL::Editor
{
	export struct ComponentElucidator :public Fragment {
		/** override draw gui */
		virtual auto onDrawGui(uint32_t flags, void* data) noexcept -> void override {
			GameObjectInspector::GameObjectData* goData = static_cast<GameObjectInspector::GameObjectData*>(data);
			elucidateComponent(goData);
		}
		/** elucidate component */
		virtual auto elucidateComponent(GameObjectInspector::GameObjectData* data) noexcept -> void = 0;
	};

	export template<class T, class UIFunction>
	auto drawComponent(GFX::GameObject* gameObject, std::string const& name, UIFunction uiFunction, bool couldRemove = true) noexcept -> void {
		const ImGuiTreeNodeFlags treeNodeFlags =
			ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_Framed | ImGuiTreeNodeFlags_FramePadding |
			ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_AllowItemOverlap;
		T* component = gameObject->getEntity().getComponent<T>();
		if (component) {
			ImVec2 contentRegionAvailable = ImGui::GetContentRegionAvail();
			ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2{ 4,4 });
			float lineHeight = GImGui->Font->FontSize + GImGui->Style.FramePadding.y * 2.0f;
			ImGui::Separator();
			bool open = ImGui::TreeNodeEx((void*)typeid(T).hash_code(), treeNodeFlags, name.c_str());
			ImGui::PopStyleVar();
			bool removeComponent = false;
			if (couldRemove) {
				ImGui::SameLine(contentRegionAvailable.x - lineHeight * 0.5);
				if (ImGui::Button("+", ImVec2{ lineHeight, lineHeight })) {
					ImGui::OpenPopup("ComponentSettings");
				}
				if (ImGui::BeginPopup("ComponentSettings")) {
					if (ImGui::MenuItem("Remove Component"))
						removeComponent = true;
					ImGui::EndPopup();
				}
			}
			if (open) {
				T* component = gameObject->getEntity().getComponent<T>();
				uiFunction(component);
				ImGui::Dummy(ImVec2(0.0f, 20.0f));
				ImGui::TreePop();
			}
			if (couldRemove && removeComponent)
				gameObject->getEntity().removeComponent<T>();
		}
	}

	export struct TagComponentFragment :public ComponentElucidator {
		virtual auto elucidateComponent(GameObjectInspector::GameObjectData* data) noexcept -> void {
			GFX::GameObject* go = data->scene->getGameObject(data->handle);
			GFX::TagComponent* tagComponent = go->getEntity().getComponent<GFX::TagComponent>();
			if (tagComponent) {
				char buffer[256];
				memset(buffer, 0, sizeof(buffer));
				strcpy_s(buffer, tagComponent->name.c_str());
				if (ImGui::InputText(" ", buffer, sizeof(buffer)))
					tagComponent->name = std::string(buffer);
			}
		}
	};

	export struct TransformComponentFragment :public ComponentElucidator {
		virtual auto elucidateComponent(GameObjectInspector::GameObjectData* data) noexcept -> void {
			GFX::GameObject* go = data->scene->getGameObject(data->handle);
			GFX::TransformComponent* transformComponent = go->getEntity().getComponent<GFX::TransformComponent>();
			if (transformComponent) {
				drawComponent<GFX::TransformComponent>(go, "Transform", [](GFX::TransformComponent* component) {
					// set translation
					Math::vec3 translation = component->translation;
					drawVec3Control("Translation", translation, 0, 100);
					bool translation_modified = (component->translation != translation);
					if (translation_modified) component->translation = translation;
					// set rotation
					Math::vec3 rotation = component->eulerAngles;
					drawVec3Control("Rotation", rotation, 0, 100);
					bool rotation_modified = (component->eulerAngles != rotation);
					if (rotation_modified) component->eulerAngles = rotation;
					// set scale
					Math::vec3 scaling = component->scale;
					drawVec3Control("Scaling", scaling, 1, 100);
					bool scale_modified = (component->scale != scaling);
					if (scale_modified) component->scale = scaling;
				}, false);
			}
		}
	};

	export struct MeshReferenceComponentFragment :public ComponentElucidator {
		virtual auto elucidateComponent(GameObjectInspector::GameObjectData* data) noexcept -> void {
			GFX::GameObject* go = data->scene->getGameObject(data->handle);
			GFX::MeshReference* meshReference = go->getEntity().getComponent<GFX::MeshReference>();
			if (meshReference) {
				drawComponent<GFX::MeshReference>(go, "MeshReference", [](GFX::MeshReference* component) {
					if (component->mesh) {
						const int index_size = component->mesh->primitiveState.stripIndexFormat == RHI::IndexFormat::UINT16_t ? sizeof(uint16_t) : sizeof(uint32_t);
						const int index_count = component->mesh->indexBufferInfo.size / index_size;
						const int primitive_count = index_count / 3;
						if (ImGui::TreeNode("Vertex Buffer")) {
							ImGui::BulletText((std::string("Size (bytes): ") + std::to_string(component->mesh->vertexBufferInfo.size)).c_str());
							if (ImGui::TreeNode("Buffer Layout")) {
								ImGui::BulletText((std::string("Array Stride: ") + std::to_string(component->mesh->vertexBufferLayout.arrayStride)).c_str());
								ImGui::BulletText((std::string("Step Mode: ") + to_string(component->mesh->vertexBufferLayout.stepMode)).c_str());
								if (ImGui::TreeNode("Attributes Layouts:")) {
									for (auto& item : component->mesh->vertexBufferLayout.attributes) {
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
							ImGui::BulletText((std::string("Size (bytes): ") + std::to_string(component->mesh->indexBufferInfo.size)).c_str());
							ImGui::BulletText((std::string("Index Count: ") + std::to_string(index_count)).c_str());
							ImGui::TreePop();
						}
						if (ImGui::TreeNode("Primitive Status")) {
							ImGui::BulletText((std::string("Primitive Count: ") + std::to_string(primitive_count)).c_str());
							ImGui::BulletText((std::string("Topology: ") + to_string(component->mesh->primitiveState.topology)).c_str());

							ImGui::TreePop();
						}
					}
					ImGui::BulletText("Custom Primitive ID: ");
					ImGui::SameLine();
					ImGui::PushItemWidth(std::max(int(ImGui::GetContentRegionAvail().x), 5));
					int i = static_cast<int>(component->customPrimitiveFlag);
					ImGui::InputInt(" ", &i, 1, 10);
					ImGui::PopItemWidth();
					component->customPrimitiveFlag = static_cast<size_t>(i);
					});
			}
		}
	};

	export struct MeshRendererComponentFragment :public ComponentElucidator {
		virtual auto elucidateComponent(GameObjectInspector::GameObjectData* data) noexcept -> void {
			GFX::GameObject* go = data->scene->getGameObject(data->handle);
			GFX::MeshRenderer* meshRenderer = go->getEntity().getComponent<GFX::MeshRenderer>();
			if (meshRenderer) {
				drawComponent<GFX::MeshRenderer>(go, "MeshRenderer", [](GFX::MeshRenderer* component) {
					uint32_t idx = 0;
					for (auto& material : component->materials) {
						if (ImGui::TreeNode(("Material ID: " + std::to_string(idx)).c_str())) {
							MaterialElucidator::onDrawGui_PTR(material);
							ImGui::TreePop();
						}
						++idx;
					}
				});
			}
		}
	};

	export struct LightComponentFragment :public ComponentElucidator {
		virtual auto elucidateComponent(GameObjectInspector::GameObjectData* data) noexcept -> void {
			GFX::GameObject* go = data->scene->getGameObject(data->handle);
			GFX::LightComponent* lightComp = go->getEntity().getComponent<GFX::LightComponent>();
			if (lightComp) {
				drawComponent<GFX::LightComponent>(go, "LightComponent", [](GFX::LightComponent* component) {
					// set type
					static std::string lightTypeNames;
					static bool inited = false;
					if (!inited) {
						for (uint32_t i = 0; i < static_cast<uint32_t>(GFX::LightComponent::LightType::MAX_ENUM); ++i) {
							if (i != 0)
								lightTypeNames.push_back('\0');
							lightTypeNames += GFX::to_string(static_cast<GFX::LightComponent::LightType>(i));
						}
						lightTypeNames.push_back('\0');
						lightTypeNames.push_back('\0');
						inited = true;
					}
					int item_current = static_cast<int>(component->type);
					ImGui::Combo("Light Type", &item_current, lightTypeNames.c_str());
					if (item_current != static_cast<int>(component->type))
						component->type = GFX::LightComponent::LightType(item_current);
					// set scale
					Math::vec3 intensity = component->intensity;
					drawVec3Control("Intensity", intensity, 0, 100);
					if (intensity != component->intensity)
						component->intensity = intensity;
				});
			}
		}
	};

	export struct CameraComponentFragment :public ComponentElucidator {
		virtual auto elucidateComponent(GameObjectInspector::GameObjectData* data) noexcept -> void {
			GFX::GameObject* go = data->scene->getGameObject(data->handle);
			GFX::CameraComponent* cameraComp = go->getEntity().getComponent<GFX::CameraComponent>();
			if (cameraComp) {
				drawComponent<GFX::CameraComponent>(go, "CameraComponent", [](GFX::CameraComponent* component) {
					if (component->projectType == GFX::CameraComponent::ProjectType::PERSPECTIVE) {
						ImGui::BulletText("Project Type: PERSPECTIVE");
						bool isDirty = false;
						float fovy = component->fovy;
						drawFloatControl("FoV", fovy, 45);
						if (fovy != component->fovy) {
							component->fovy = fovy;
							isDirty = true;
						}
						float near = component->near;
						drawFloatControl("Near", near, 0.001);
						if (near != component->near) {
							component->near = near;
							isDirty = true;
						}
						float far = component->far;
						drawFloatControl("Far", far, 1000);
						if (far != component->far) {
							component->far = far;
							isDirty = true;
						}
						bool isPrimary = component->isPrimaryCamera;
						drawBoolControl("Is Primary", isPrimary, 100);
						if (isPrimary != component->isPrimaryCamera) {
							component->isPrimaryCamera = isPrimary;
						}
					}
					else if (component->projectType == GFX::CameraComponent::ProjectType::ORTHOGONAL) {
						ImGui::BulletText("Project Type: ORTHOGONAL");
					}
				});
			}
		}
	};
}