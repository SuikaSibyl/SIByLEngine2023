module;
#include <vector>
#include <string>
#include <unordered_map>
#include <cstdint>
#include <typeinfo>
#include <memory>
#include <functional>
#include <imgui.h>
#include <imgui_internal.h>
export module SE.Editor.GFX:Utils;
import SE.Editor.Core;
import SE.Core.ECS;
import SE.Math.Geometric;
import SE.RHI;
import SE.GFX;

namespace SIByL::Editor
{
	export inline auto drawBoolControl(std::string const& label, bool& value, float labelWidth = 50) noexcept -> void {
		ImGuiIO& io = ImGui::GetIO();
		auto boldFont = io.Fonts->Fonts[0];
		ImGui::PushID(label.c_str());
		ImGui::Columns(2);
		// First Column
		{	ImGui::SetColumnWidth(0, labelWidth);
			ImGui::Text(label.c_str());
			ImGui::NextColumn(); }
		// Second Column
		{	ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2{ 0,0 });
			ImGui::Checkbox(label.c_str(), &value);
			ImGui::PopStyleVar();
		}
		ImGui::Columns(1);
		ImGui::PopID();
	}

	export inline auto drawFloatControl(std::string const& label, float& value, float resetValue = 0, float columeWidth = 100) noexcept -> void {
		ImGuiIO& io = ImGui::GetIO();
		auto boldFont = io.Fonts->Fonts[0];
		ImGui::PushID(label.c_str());
		ImGui::Columns(2);
		// First Column
		{	ImGui::SetColumnWidth(0, columeWidth);
			ImGui::Text(label.c_str());
			ImGui::NextColumn(); }
		// Second Column
		{	ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2{ 0,0 });

			float lineHeight = GImGui->Font->FontSize + GImGui->Style.FramePadding.y * 2.0f;
			ImVec2 buttonSize = { lineHeight + 3.0f, lineHeight };
			ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{ 0.8,0.1f,0.15f,1.0f });
			ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{ 0.9,0.2f,0.2f,1.0f });
			ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{ 0.8,0.1f,0.15f,1.0f });
			ImGui::PushFont(boldFont);
			if (ImGui::Button("X", buttonSize))
				value = resetValue;
			ImGui::PopFont();
			ImGui::PopStyleColor(3);
			ImGui::SameLine();
			ImGui::DragFloat("##x", &value, 0.1f);
			ImGui::SameLine();
			ImGui::PopStyleVar();
		}
		ImGui::Columns(1);
		ImGui::PopID();
	}

	export inline auto drawVec3Control(const std::string& label, Math::vec3& values, float resetValue = 0, float columeWidth = 100) noexcept -> void {
		ImGuiIO& io = ImGui::GetIO();
		auto boldFont = io.Fonts->Fonts[0];
		ImGui::PushID(label.c_str());
		ImGui::Columns(2);
		// First Column
		{	ImGui::SetColumnWidth(0, columeWidth);
			ImGui::Text(label.c_str());
			ImGui::NextColumn(); }
		// Second Column
		{
			ImGui::PushMultiItemsWidths(3, ImGui::CalcItemWidth());
			ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2{ 0,0 });

			float lineHeight = GImGui->Font->FontSize + GImGui->Style.FramePadding.y * 2.0f;
			ImVec2 buttonSize = { lineHeight + 3.0f, lineHeight };

			ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{ 0.8,0.1f,0.15f,1.0f });
			ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{ 0.9,0.2f,0.2f,1.0f });
			ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{ 0.8,0.1f,0.15f,1.0f });
			ImGui::PushFont(boldFont);
			if (ImGui::Button("X", buttonSize))
				values.x = resetValue;
			ImGui::PopFont();
			ImGui::PopStyleColor(3);
			ImGui::SameLine();
			ImGui::DragFloat("##x", &values.x, 0.1f);
			ImGui::PopItemWidth();
			ImGui::SameLine();

			ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{ 0.2,0.7f,0.2f,1.0f });
			ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{ 0.3,0.8f,0.3f,1.0f });
			ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{ 0.2,0.7f,0.2f,1.0f });
			ImGui::PushFont(boldFont);
			if (ImGui::Button("Y", buttonSize))
				values.y = resetValue;
			ImGui::PopFont();
			ImGui::PopStyleColor(3);
			ImGui::SameLine();
			ImGui::DragFloat("##y", &values.y, 0.1f);
			ImGui::PopItemWidth();
			ImGui::SameLine();

			ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{ 0.1,0.26f,0.8f,1.0f });
			ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{ 0.2,0.35f,0.9f,1.0f });
			ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{ 0.1,0.26f,0.8f,1.0f });
			ImGui::PushFont(boldFont);
			if (ImGui::Button("Z", buttonSize))
				values.z = resetValue;
			ImGui::PopFont();
			ImGui::PopStyleColor(3);
			ImGui::SameLine();
			ImGui::DragFloat("##z", &values.z, 0.1f);
			ImGui::PopItemWidth();
			//ImGui::SameLine();
			ImGui::PopStyleVar();
		}
		ImGui::Columns(1);
		ImGui::PopID();
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
		default: return "UNKNOWN";
		}
	}

	export inline auto to_string(RHI::PrimitiveTopology topology) noexcept -> std::string {
		switch (topology) {
		case SIByL::RHI::PrimitiveTopology::TRIANGLE_STRIP:	return "TRIANGLE_STRIP";
		case SIByL::RHI::PrimitiveTopology::TRIANGLE_LIST:return "TRIANGLE_LIST";
		case SIByL::RHI::PrimitiveTopology::LINE_STRIP:return "LINE_STRIP";
		case SIByL::RHI::PrimitiveTopology::LINE_LIST:return "LINE_LIST";
		case SIByL::RHI::PrimitiveTopology::POINT_LIST:return "POINT_LIST";
		default: return "UNKNOWN";
		}
	}

	export inline auto to_string(RHI::VertexStepMode stepMode) noexcept -> std::string {
		switch (stepMode) {
		case SIByL::RHI::VertexStepMode::VERTEX:	return "VERTEX";
		case SIByL::RHI::VertexStepMode::INSTANCE:	return "INSTANCE";
		default: return "UNKNOWN";
		}
	}
}