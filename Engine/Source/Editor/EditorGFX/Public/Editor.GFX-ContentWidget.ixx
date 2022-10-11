module;
#include <vector>
#include <string>
#include <unordered_map>
#include <cstdint>
#include <typeinfo>
#include <imgui.h>
#include <imgui_internal.h>
export module Editor.GFX:ContentWidget;
import :InspectorWidget;
import Editor.Framework;
import GFX.Resource;

namespace SIByL::Editor
{
	export struct ContentWidget :public Widget {

	};
}