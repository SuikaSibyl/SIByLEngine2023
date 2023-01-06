module;
#include <compare>
#include <filesystem>
export module SE.Editor.Config;
import SE.Core.Resource;
import SE.GFX.Core;
import SE.Editor.Core;
import SE.Editor.GFX;

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
			layer->getWidget<Editor::SceneWidget>()->gameobjectInspector.registerFragment<Editor::MeshRendererComponentFragment>();
			layer->getWidget<Editor::SceneWidget>()->gameobjectInspector.registerFragment<Editor::CameraComponentFragment>();

			layer->getWidget<Editor::InspectorWidget>()->resourceViewer.registerElucidator<Editor::TextureElucidator>("struct SIByL::GFX::Texture");
			layer->getWidget<Editor::InspectorWidget>()->resourceViewer.registerElucidator<Editor::MaterialElucidator>("struct SIByL::GFX::Material");

			layer->getWidget<Editor::ContentWidget>()->inspectorWidget = layer->getWidget<Editor::InspectorWidget>();
			layer->getWidget<Editor::ContentWidget>()->reigsterIconResources();
		}
	};
}