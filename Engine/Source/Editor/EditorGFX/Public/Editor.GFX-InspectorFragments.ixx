module;
#include <vector>
#include <string>
#include <unordered_map>
#include <cstdint>
#include <typeinfo>
#include <imgui.h>
#include <imgui_internal.h>
export module Editor.GFX:InspectorFragments;
import :SceneWidget;
import Math.Vector;
import RHI;
import GFX.Resource;
import GFX.Components;
import Editor.Framework;
import Editor.Utils;

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
	auto drawComponent(GFX::GameObject* gameObject, std::string const& name, UIFunction uiFunction) noexcept -> void {
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
			ImGui::SameLine(contentRegionAvailable.x - lineHeight * 0.5);
			if (ImGui::Button("+", ImVec2{ lineHeight, lineHeight })) {
				ImGui::OpenPopup("ComponentSettings");
			}
			bool removeComponent = false;
			if (ImGui::BeginPopup("ComponentSettings")) {
				if (ImGui::MenuItem("Remove Component"))
					removeComponent = true;
				ImGui::EndPopup();
			}
			if (open) {
				T* component = gameObject->getEntity().getComponent<T>();
				uiFunction(component);
				ImGui::Dummy(ImVec2(0.0f, 20.0f));
				ImGui::TreePop();
			}
			if (removeComponent)
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
				});
			}
		}
	};

	export inline auto to_string(RHI::PrimitiveTopology topology) noexcept -> std::string {
		switch (topology) {
		case SIByL::RHI::PrimitiveTopology::TRIANGLE_STRIP:	return "TRIANGLE_STRIP";
		case SIByL::RHI::PrimitiveTopology::TRIANGLE_LIST:return "TRIANGLE_LIST";
		case SIByL::RHI::PrimitiveTopology::LINE_STRIP:return "LINE_STRIP";
		case SIByL::RHI::PrimitiveTopology::LINE_LIST:return "LINE_LIST";
		case SIByL::RHI::PrimitiveTopology::POINT_LIST:return "POINT_LIST";
		default: return "UNKNOWN"; }
	}

	export inline auto to_string(RHI::VertexStepMode stepMode) noexcept -> std::string {
		switch (stepMode) {
		case SIByL::RHI::VertexStepMode::VERTEX:	return "VERTEX";
		case SIByL::RHI::VertexStepMode::INSTANCE:	return "INSTANCE";
		default: return "UNKNOWN"; }
	}

	export inline auto to_string(RHI::VertexFormat vertexFormat) noexcept -> std::string {
		switch (vertexFormat) {
		case SIByL::RHI::VertexFormat::UINT8X2:		return"UINT8X2";
		case SIByL::RHI::VertexFormat::UINT8X4:		return"UINT8X4";
		case SIByL::RHI::VertexFormat::SINT8X2:		return"SINT8X2";
		case SIByL::RHI::VertexFormat::SINT8X4:		return"SINT8X4";
		case SIByL::RHI::VertexFormat::UNORM8X2:	return"UNORM8X2";
		case SIByL::RHI::VertexFormat::UNORM8X4:	return"UNORM8X4";
		case SIByL::RHI::VertexFormat::SNORM8X2:	return"SNORM8X2";
		case SIByL::RHI::VertexFormat::SNORM8X4:	return"SNORM8X4";
		case SIByL::RHI::VertexFormat::UINT16X2:	return"UINT16X2";
		case SIByL::RHI::VertexFormat::UINT16X4:	return"UINT16X4";
		case SIByL::RHI::VertexFormat::SINT16X2:	return"SINT16X2";
		case SIByL::RHI::VertexFormat::SINT16X4:	return"SINT16X4";
		case SIByL::RHI::VertexFormat::UNORM16X2:	return"UNORM16X2";
		case SIByL::RHI::VertexFormat::UNORM16X4:	return"UNORM16X4";
		case SIByL::RHI::VertexFormat::SNORM16X2:	return"SNORM16X2";
		case SIByL::RHI::VertexFormat::SNORM16X4:	return"SNORM16X4";
		case SIByL::RHI::VertexFormat::FLOAT16X2:	return"FLOAT16X2";
		case SIByL::RHI::VertexFormat::FLOAT16X4:	return"FLOAT16X4";
		case SIByL::RHI::VertexFormat::FLOAT32:		return"FLOAT32";
		case SIByL::RHI::VertexFormat::FLOAT32X2:	return"FLOAT32X2";
		case SIByL::RHI::VertexFormat::FLOAT32X3:	return"FLOAT32X3";
		case SIByL::RHI::VertexFormat::FLOAT32X4:	return"FLOAT32X4";
		case SIByL::RHI::VertexFormat::UINT32:		return"UINT32";
		case SIByL::RHI::VertexFormat::UINT32X2:	return"UINT32X2";
		case SIByL::RHI::VertexFormat::UINT32X3:	return"UINT32X3";
		case SIByL::RHI::VertexFormat::UINT32X4:	return"UINT32X4";
		case SIByL::RHI::VertexFormat::SINT32:		return"SINT32";
		case SIByL::RHI::VertexFormat::SINT32X2:	return"SINT32X2";
		case SIByL::RHI::VertexFormat::SINT32X3:	return"SINT32X3";
		case SIByL::RHI::VertexFormat::SINT32X4:	return"SINT32X4";
		default: return "UNKNOWN"; }
	}

	export struct MeshReferenceComponentFragment :public ComponentElucidator {
		virtual auto elucidateComponent(GameObjectInspector::GameObjectData* data) noexcept -> void {
			GFX::GameObject* go = data->scene->getGameObject(data->handle);
			GFX::MeshReference* meshReference = go->getEntity().getComponent<GFX::MeshReference>();
			if (meshReference) {
				drawComponent<GFX::MeshReference>(go, "MeshReference", [](GFX::MeshReference* component) {
					const int index_size = component->mesh->primitiveState.stripIndexFormat == RHI::IndexFormat::UINT16_t ? sizeof(uint16_t) : sizeof(uint32_t);
					const int index_count = component->mesh->indexBuffer->size() / index_size;
					const int primitive_count = index_count / 3;
					if (ImGui::TreeNode("Vertex Buffer")) {
						ImGui::BulletText((std::string("Size (bytes): ") + std::to_string(component->mesh->vertexBuffer->size())).c_str());
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
						ImGui::BulletText((std::string("Size (bytes): ") + std::to_string(component->mesh->indexBuffer->size())).c_str());
						ImGui::BulletText((std::string("Index Count: ") + std::to_string(index_count)).c_str());
						ImGui::TreePop();
					}
					if (ImGui::TreeNode("Primitive Status")) {
						ImGui::BulletText((std::string("Primitive Count: ") + std::to_string(primitive_count)).c_str());
						ImGui::BulletText((std::string("Topology: ") + to_string(component->mesh->primitiveState.topology)).c_str());

						ImGui::TreePop();
					}
				});
			}
		}
	};
}