module;
#include <string>
#include <typeinfo>
#include <imgui.h>
#include <imgui_internal.h>
export module SE.Editor.GFX:Status;
import :Utils;
import :TextureFragment;
import SE.Editor.Core;
import SE.Core.Resource;
import SE.Core.Misc;
import SE.Core.Misc;
import SE.Core.Log;
import SE.RHI;
import SE.Image;
import SE.GFX;
import SE.Math.Geometric;
import SE.Platform.Window;

namespace SIByL::Editor
{
	export struct StatusWidget :public Widget {

		Core::Timer* timer;

		/** draw gui*/
		virtual auto onDrawGui() noexcept -> void override {
			ImGui::Begin("Status");

			ImGui::Text(("fps:\t" + std::to_string(1. / timer->deltaTime())).c_str());
			ImGui::Text(("time:\t" + std::to_string(timer->deltaTime())).c_str());

			ImGui::End();
		}
	};
}
