#define SIByL_API __declspec(dllexport)
#include <seditor-rdg.hpp>
#undef SIByL_API
#define SIByL_API __declspec(dllimport)
#include <se.editor.hpp>
#include <seditor-base.hpp>
#undef SIByL_API

namespace se::editor {
auto RDGViewerWidget::onDrawGui() noexcept -> void {
  ImGui::Begin("Render Pipeline");
  // Render pipeline UI
  if (pipeline != nullptr) {
    pipeline->renderUI();
    ImGui::Separator();
    std::vector<rdg::Graph*> graphs = pipeline->getActiveGraphs();
    for (auto* graph : graphs)
      for (size_t i : graph->flattenedPasses) {
        if (ImGui::TreeNode(graph->passes[i]->identifier.c_str())) {
          graph->passes[i]->renderUI();
          ImGui::TreePop();
        }
      }
  }
  ImGui::End();
}
}