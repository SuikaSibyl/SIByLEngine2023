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
export module SE.Editor.GFX:LogWidget;
import SE.Editor.Core;
import SE.Core.ECS;
import SE.Core.Log;
import SE.GFX;

namespace SIByL::Editor
{
	export struct LogWidget :public Widget {
		/** constructor */
		LogWidget();
		/** destructor */
		~LogWidget();
		/** draw gui*/
		virtual auto onDrawGui() noexcept -> void override;
		/** set custom draw */
		auto setCustomDraw(std::function<void()> func) noexcept -> void { customDraw = func; }
		/** set empty */
		auto clear() noexcept -> void;
        /** add line */
        auto addline(std::string const& str) -> void;
		/** custom draw on inspector widget */
		std::function<void()> customDraw;

		ImGuiTextBuffer     _buf;
		ImGuiTextFilter     filter;
		ImVector<int>       lineOffsets; // Index to lines offset. We maintain this with AddLog() calls.
		bool                autoScroll;  // Keep scrolling if already at the bottom.
	};

#pragma region LOG_WIDGE_IMPL

	LogWidget::LogWidget() {
		autoScroll = true;
        clear();
        Core::LogManager::get()->editorCallback = std::bind(&LogWidget::addline, this, std::placeholders::_1);
	}

	LogWidget::~LogWidget() {
        clear();
        Core::LogManager::get()->editorCallback = nullptr;
	}

	auto LogWidget::clear() noexcept -> void {
        _buf.clear();
		lineOffsets.clear();
		lineOffsets.push_back(0);
	}

    auto LogWidget::addline(std::string const& str) -> void {
        int old_size = _buf.size();
        _buf.append(str.c_str());
        old_size = _buf.size();
        lineOffsets.push_back(old_size);
    }

	auto LogWidget::onDrawGui() noexcept -> void {
        ImGui::Begin("Log", 0);

        // Options menu
        if (ImGui::BeginPopup("Options")) {
            ImGui::Checkbox("Auto-scroll", &autoScroll);
            ImGui::EndPopup();
        }

        // Main window
        if (ImGui::Button("Options"))
            ImGui::OpenPopup("Options");
        ImGui::SameLine();
        bool bclear = ImGui::Button("Clear");
        ImGui::SameLine();
        bool copy = ImGui::Button("Copy");
        ImGui::SameLine();
        filter.Draw("Filter", -100.0f);

        ImGui::Separator();
        ImGui::BeginChild("scrolling", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar);

        if (bclear)
            clear();
        if (copy)
            ImGui::LogToClipboard();

        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 0));
        const char* buf = _buf.begin();
        const char* buf_end = _buf.end();

        for (int line_no = 0; line_no < lineOffsets.Size; line_no++) {
            const char* line_start = buf + lineOffsets[line_no];
            const char* line_end = (line_no + 1 < lineOffsets.Size) ? (buf + lineOffsets[line_no + 1] - 1) : buf_end;
            if (!filter.IsActive() || filter.PassFilter(line_start, line_end)) {
                if (line_start[1] == 'W') {
                    ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 190, 0, 255));
                    ImGui::TextUnformatted(line_start, line_end);
                    ImGui::PopStyleColor();
                }
                else if (line_start[1] == 'D') {
                    ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(50, 180, 255, 255));
                    ImGui::TextUnformatted(line_start, line_end);
                    ImGui::PopStyleColor();
                }
                else if (line_start[1] == 'E') {
                    ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 30, 61, 255));
                    ImGui::TextUnformatted(line_start, line_end);
                    ImGui::PopStyleColor();
                }
                else {
                    ImGui::TextUnformatted(line_start, line_end);
                }
            }
        }
        ImGui::PopStyleVar();

        if (autoScroll && ImGui::GetScrollY() >= ImGui::GetScrollMaxY())
            ImGui::SetScrollHereY(1.0f);

        ImGui::EndChild();
        ImGui::End();
	}

#pragma endregion

}