#include <SE.Editor.RDG.hpp>
#include <imgui.h>
#include <imgui_internal.h>

namespace SIByL::Editor {
auto RDGViewerWidget::onDrawGui() noexcept -> void {
  ImGui::Begin("Render Pipeline");
  // Render pipeline UI
  pipeline->renderUI();
  ImGui::Separator();
  std::vector<RDG::Graph*> graphs = pipeline->getActiveGraphs();
  for (auto* graph : graphs)
      for (size_t i : graph->flattenedPasses) {
        if (ImGui::TreeNode(graph->passes[i]->identifier.c_str())) {
          graph->passes[i]->renderUI();
          ImGui::TreePop();
        }
      }
  ImGui::End();
}
}  // namespace SIByL::Editor
