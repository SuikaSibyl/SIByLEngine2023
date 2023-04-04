module;
#include <string>
#include <imgui.h>
#include <imgui_internal.h>
export module SE.Editor.RDG:RDGViewer;
import SE.Editor.Core;
import SE.Editor.GFX;
import SE.RHI;
import SE.GFX;
import SE.RDG;

namespace SIByL::Editor
{
	export struct RDGViewerWidget :public Widget {

		RDG::Graph* graph = nullptr;

		/** draw gui*/
		virtual auto onDrawGui() noexcept -> void override {
			ImGui::Begin("RenderGraph Viewer", 0, ImGuiWindowFlags_MenuBar);
			ImGui::PushItemWidth(ImGui::GetFontSize() * -12);
			if (ImGui::BeginMenuBar()) {
				if (ImGui::Button("capture", { 200, 100 })) {
					//	captureImage(rdg->getTexture("RasterizerTarget_Color")->guid);
				}
				ImGui::EndMenuBar();
			}
			ImGui::PopItemWidth();


			ImGui::End();
		}
	};
}
