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

		RDG::Pipeline* pipeline = nullptr;

		/** draw gui*/
		virtual auto onDrawGui() noexcept -> void override {
			ImGui::Begin("Render Graph");

			RDG::Graph* graph = pipeline->getActiveGraph();
			for (size_t i : graph->flattenedPasses) {
				if (ImGui::TreeNode(graph->passes[i]->identifier.c_str())) {
					graph->passes[i]->renderUI();
					ImGui::TreePop();
				}
			}

			ImGui::End();
		}
	};
}
