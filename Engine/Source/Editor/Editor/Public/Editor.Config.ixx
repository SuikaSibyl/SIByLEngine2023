module;
#include <compare>
#include <filesystem>
export module Editor.Config;
import Core.Resource.RuntimeManage;
import GFX.Resource;
import GFX.GFXManager;
import Editor.Core;
import Editor.Framework;
import Editor.GFX;

namespace SIByL::Editor
{
	export struct Config {
		/** build editor layer up in a pre-defined way */
		static inline auto buildEditorLayer(Editor::EditorLayer* layer) noexcept -> void {
			layer->registerWidget<Editor::SceneWidget>();
			layer->registerWidget<Editor::InspectorWidget>();
			layer->registerWidget<Editor::ContentWidget>();
			layer->registerWidget<Editor::LogWidget>();

			layer->getWidget<Editor::SceneWidget>()->bindScene(nullptr);
			layer->getWidget<Editor::SceneWidget>()->inspectorWidget = layer->getWidget<Editor::InspectorWidget>();
			layer->getWidget<Editor::SceneWidget>()->gameobjectInspector.registerFragment<Editor::TagComponentFragment>();
			layer->getWidget<Editor::SceneWidget>()->gameobjectInspector.registerFragment<Editor::TransformComponentFragment>();
			layer->getWidget<Editor::SceneWidget>()->gameobjectInspector.registerFragment<Editor::MeshReferenceComponentFragment>();

			layer->getWidget<Editor::ContentWidget>()->reigsterIconResources();
		}
	};
}