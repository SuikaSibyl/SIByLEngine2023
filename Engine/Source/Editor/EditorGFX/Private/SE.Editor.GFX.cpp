#pragma once
#include <SE.Editor.GFX.hpp>
#include <imgui.h>
#include <imgui_internal.h>
#include <SE.Editor.Core.hpp>
#include <SE.GFX.hpp>
#include <SE.Math.Geometric.hpp>
#include <SE.RHI.hpp>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <typeinfo>
#include <unordered_map>
#include <vector>

namespace SIByL::Editor {
auto drawBoolControl(std::string const& label, bool& value,
                     float labelWidth) noexcept -> void {
  ImGuiIO& io = ImGui::GetIO();
  auto boldFont = io.Fonts->Fonts[0];
  ImGui::PushID(label.c_str());
  ImGui::Columns(2);
  // First Column
  {
    ImGui::SetColumnWidth(0, labelWidth);
    ImGui::Text(label.c_str());
    ImGui::NextColumn();
  }
  // Second Column
  {
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2{0, 0});
    ImGui::Checkbox(label.c_str(), &value);
    ImGui::PopStyleVar();
  }
  ImGui::Columns(1);
  ImGui::PopID();
}

auto drawFloatControl(std::string const& label, float& value, float resetValue,
                      float columeWidth) noexcept -> void {
  ImGuiIO& io = ImGui::GetIO();
  auto boldFont = io.Fonts->Fonts[0];
  ImGui::PushID(label.c_str());
  ImGui::Columns(2);
  // First Column
  {
    ImGui::SetColumnWidth(0, columeWidth);
    ImGui::Text(label.c_str());
    ImGui::NextColumn();
  }
  // Second Column
  {
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2{0, 0});

    float lineHeight =
        GImGui->Font->FontSize + GImGui->Style.FramePadding.y * 2.0f;
    ImVec2 buttonSize = {lineHeight + 3.0f, lineHeight};
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{0.8, 0.1f, 0.15f, 1.0f});
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered,
                          ImVec4{0.9, 0.2f, 0.2f, 1.0f});
    ImGui::PushStyleColor(ImGuiCol_ButtonActive,
                          ImVec4{0.8, 0.1f, 0.15f, 1.0f});
    ImGui::PushFont(boldFont);
    if (ImGui::Button("X", buttonSize)) value = resetValue;
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

auto drawVec3Control(const std::string& label, Math::vec3& values,
                     float resetValue, float columeWidth) noexcept -> void {
  ImGuiIO& io = ImGui::GetIO();
  auto boldFont = io.Fonts->Fonts[0];
  ImGui::PushID(label.c_str());
  ImGui::Columns(2);
  // First Column
  {
    ImGui::SetColumnWidth(0, columeWidth);
    ImGui::Text(label.c_str());
    ImGui::NextColumn();
  }
  // Second Column
  {
    ImGui::PushMultiItemsWidths(3, ImGui::CalcItemWidth());
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2{0, 0});

    float lineHeight =
        GImGui->Font->FontSize + GImGui->Style.FramePadding.y * 2.0f;
    ImVec2 buttonSize = {lineHeight + 3.0f, lineHeight};

    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{0.8, 0.1f, 0.15f, 1.0f});
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered,
                          ImVec4{0.9, 0.2f, 0.2f, 1.0f});
    ImGui::PushStyleColor(ImGuiCol_ButtonActive,
                          ImVec4{0.8, 0.1f, 0.15f, 1.0f});
    ImGui::PushFont(boldFont);
    if (ImGui::Button("X", buttonSize)) values.x = resetValue;
    ImGui::PopFont();
    ImGui::PopStyleColor(3);
    ImGui::SameLine();
    ImGui::DragFloat("##x", &values.x, 0.1f);
    ImGui::PopItemWidth();
    ImGui::SameLine();

    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{0.2, 0.7f, 0.2f, 1.0f});
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered,
                          ImVec4{0.3, 0.8f, 0.3f, 1.0f});
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{0.2, 0.7f, 0.2f, 1.0f});
    ImGui::PushFont(boldFont);
    if (ImGui::Button("Y", buttonSize)) values.y = resetValue;
    ImGui::PopFont();
    ImGui::PopStyleColor(3);
    ImGui::SameLine();
    ImGui::DragFloat("##y", &values.y, 0.1f);
    ImGui::PopItemWidth();
    ImGui::SameLine();

    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{0.1, 0.26f, 0.8f, 1.0f});
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered,
                          ImVec4{0.2, 0.35f, 0.9f, 1.0f});
    ImGui::PushStyleColor(ImGuiCol_ButtonActive,
                          ImVec4{0.1, 0.26f, 0.8f, 1.0f});
    ImGui::PushFont(boldFont);
    if (ImGui::Button("Z", buttonSize)) values.z = resetValue;
    ImGui::PopFont();
    ImGui::PopStyleColor(3);
    ImGui::SameLine();
    ImGui::DragFloat("##z", &values.z, 0.1f);
    ImGui::PopItemWidth();
    // ImGui::SameLine();
    ImGui::PopStyleVar();
  }
  ImGui::Columns(1);
  ImGui::PopID();
}

auto to_string(RHI::VertexFormat vertexFormat) noexcept -> std::string {
  switch (vertexFormat) {
    case SIByL::RHI::VertexFormat::UINT8X2:
      return "UINT8X2";
    case SIByL::RHI::VertexFormat::UINT8X4:
      return "UINT8X4";
    case SIByL::RHI::VertexFormat::SINT8X2:
      return "SINT8X2";
    case SIByL::RHI::VertexFormat::SINT8X4:
      return "SINT8X4";
    case SIByL::RHI::VertexFormat::UNORM8X2:
      return "UNORM8X2";
    case SIByL::RHI::VertexFormat::UNORM8X4:
      return "UNORM8X4";
    case SIByL::RHI::VertexFormat::SNORM8X2:
      return "SNORM8X2";
    case SIByL::RHI::VertexFormat::SNORM8X4:
      return "SNORM8X4";
    case SIByL::RHI::VertexFormat::UINT16X2:
      return "UINT16X2";
    case SIByL::RHI::VertexFormat::UINT16X4:
      return "UINT16X4";
    case SIByL::RHI::VertexFormat::SINT16X2:
      return "SINT16X2";
    case SIByL::RHI::VertexFormat::SINT16X4:
      return "SINT16X4";
    case SIByL::RHI::VertexFormat::UNORM16X2:
      return "UNORM16X2";
    case SIByL::RHI::VertexFormat::UNORM16X4:
      return "UNORM16X4";
    case SIByL::RHI::VertexFormat::SNORM16X2:
      return "SNORM16X2";
    case SIByL::RHI::VertexFormat::SNORM16X4:
      return "SNORM16X4";
    case SIByL::RHI::VertexFormat::FLOAT16X2:
      return "FLOAT16X2";
    case SIByL::RHI::VertexFormat::FLOAT16X4:
      return "FLOAT16X4";
    case SIByL::RHI::VertexFormat::FLOAT32:
      return "FLOAT32";
    case SIByL::RHI::VertexFormat::FLOAT32X2:
      return "FLOAT32X2";
    case SIByL::RHI::VertexFormat::FLOAT32X3:
      return "FLOAT32X3";
    case SIByL::RHI::VertexFormat::FLOAT32X4:
      return "FLOAT32X4";
    case SIByL::RHI::VertexFormat::UINT32:
      return "UINT32";
    case SIByL::RHI::VertexFormat::UINT32X2:
      return "UINT32X2";
    case SIByL::RHI::VertexFormat::UINT32X3:
      return "UINT32X3";
    case SIByL::RHI::VertexFormat::UINT32X4:
      return "UINT32X4";
    case SIByL::RHI::VertexFormat::SINT32:
      return "SINT32";
    case SIByL::RHI::VertexFormat::SINT32X2:
      return "SINT32X2";
    case SIByL::RHI::VertexFormat::SINT32X3:
      return "SINT32X3";
    case SIByL::RHI::VertexFormat::SINT32X4:
      return "SINT32X4";
    default:
      return "UNKNOWN";
  }
}

auto to_string(RHI::PrimitiveTopology topology) noexcept -> std::string {
  switch (topology) {
    case SIByL::RHI::PrimitiveTopology::TRIANGLE_STRIP:
      return "TRIANGLE_STRIP";
    case SIByL::RHI::PrimitiveTopology::TRIANGLE_LIST:
      return "TRIANGLE_LIST";
    case SIByL::RHI::PrimitiveTopology::LINE_STRIP:
      return "LINE_STRIP";
    case SIByL::RHI::PrimitiveTopology::LINE_LIST:
      return "LINE_LIST";
    case SIByL::RHI::PrimitiveTopology::POINT_LIST:
      return "POINT_LIST";
    default:
      return "UNKNOWN";
  }
}

auto to_string(RHI::VertexStepMode stepMode) noexcept -> std::string {
  switch (stepMode) {
    case SIByL::RHI::VertexStepMode::VERTEX:
      return "VERTEX";
    case SIByL::RHI::VertexStepMode::INSTANCE:
      return "INSTANCE";
    default:
      return "UNKNOWN";
  }
}
}  // namespace SIByL::Editor

namespace SIByL::Editor {
auto TextureUtils::getImGuiTexture(Core::GUID guid) noexcept -> ImGuiTexture* {
  auto& pool = ImGuiLayer::get()->ImGuiTexturePool;
  auto iter = pool.find(guid);
  if (iter == pool.end()) {
    GFX::Texture* texture =
        Core::ResourceManager::get()->getResource<GFX::Texture>(guid);
    pool.insert(
        {guid, ImGuiLayer::get()->createImGuiTexture(
                   Core::ResourceManager::get()
                       ->getResource<GFX::Sampler>(
                           GFX::GFXManager::get()->commonSampler.defaultSampler)
                       ->sampler.get(),
                   texture->getSRV(0, 1, 0, 1),
                   RHI::TextureLayout::SHADER_READ_ONLY_OPTIMAL)});
    return pool[guid].get();
  } else {
    return iter->second.get();
  }
}
}  // namespace SIByL::Editor

namespace SIByL::Editor {
auto ResourceViewer::onDrawGui(char const* type, Core::GUID guid) noexcept
    -> void {
  for (auto const& pair : elucidatorMaps) {
    if (strcmp(pair.first, type) == 0) {
      pair.second->onDrawGui(guid);
      break;
    }
  }
}
auto MeshElucidator::onDrawGui(Core::GUID guid) noexcept -> void {
  onDrawGui_GUID(guid);
}

auto MeshElucidator::onDrawGui_GUID(Core::GUID guid) noexcept -> void {
  GFX::Mesh* mesh = Core::ResourceManager::get()->getResource<GFX::Mesh>(guid);
  onDrawGui_PTR(mesh);
}

auto MeshElucidator::onDrawGui_PTR(GFX::Mesh* mesh) noexcept -> void {
  const int index_size =
      mesh->primitiveState.stripIndexFormat == RHI::IndexFormat::UINT16_t
          ? sizeof(uint16_t)
          : sizeof(uint32_t);
  const int index_count = mesh->indexBufferInfo.size / index_size;
  const int primitive_count = index_count / 3;
  if (ImGui::TreeNode("Vertex Buffer")) {
    ImGui::BulletText((std::string("Size (bytes): ") +
                       std::to_string(mesh->vertexBufferInfo.size))
                          .c_str());
    if (ImGui::TreeNode("Buffer Layout")) {
      ImGui::BulletText((std::string("Array Stride: ") +
                         std::to_string(mesh->vertexBufferLayout.arrayStride))
                            .c_str());
      ImGui::BulletText((std::string("Step Mode: ") +
                         to_string(mesh->vertexBufferLayout.stepMode))
                            .c_str());
      if (ImGui::TreeNode("Attributes Layouts:")) {
        for (auto& item : mesh->vertexBufferLayout.attributes) {
          ImGui::BulletText((to_string(item.format) + std::string(" | OFF: ") +
                             std::to_string(item.offset) +
                             std::string(" | LOC: ") +
                             std::to_string(item.shaderLocation))
                                .c_str());
        }
        ImGui::TreePop();
      }
      ImGui::TreePop();
    }
    ImGui::TreePop();
  }
  if (ImGui::TreeNode("Index Buffer")) {
    ImGui::BulletText((std::string("Size (bytes): ") +
                       std::to_string(mesh->indexBufferInfo.size))
                          .c_str());
    ImGui::BulletText(
        (std::string("Index Count: ") + std::to_string(index_count)).c_str());
    ImGui::TreePop();
  }
  if (ImGui::TreeNode("Primitive Status")) {
    ImGui::BulletText(
        (std::string("Primitive Count: ") + std::to_string(primitive_count))
            .c_str());
    ImGui::BulletText(
        (std::string("Topology: ") + to_string(mesh->primitiveState.topology))
            .c_str());

    ImGui::TreePop();
  }
}

auto TextureElucidator::onDrawGui(Core::GUID guid) noexcept -> void {
  onDrawGui_GUID(guid);
}

auto TextureElucidator::onDrawGui_GUID(Core::GUID guid) noexcept -> void {
  GFX::Texture* tex =
      Core::ResourceManager::get()->getResource<GFX::Texture>(guid);
  onDrawGui_PTR(tex);
}

auto TextureElucidator::onDrawGui_PTR(GFX::Texture* tex) noexcept -> void {
  float const texw = (float)tex->texture->width();
  float const texh = (float)tex->texture->height();
  float const wa = std::max(1.f, ImGui::GetContentRegionAvail().x - 15) / texw;
  float const ha = 1;
  float a = std::min(1.f, std::min(wa, ha));
  ImGui::Image(Editor::TextureUtils::getImGuiTexture(tex->guid)->getTextureID(),
               {a * texw, a * texh}, {0, 0}, {1, 1});
  ImGui::Text("Texture: ");
  ImGui::Text((std::string("- GUID: ") + std::to_string(tex->guid)).c_str());
  ImGui::Text((std::string("- name: ") + tex->texture->getName()).c_str());
  ImGui::Text((std::string("- width: ") + std::to_string(tex->texture->width()))
                  .c_str());
  ImGui::Text(
      (std::string("- height: ") + std::to_string(tex->texture->height()))
          .c_str());
}

auto MaterialElucidator::onDrawGui(Core::GUID guid) noexcept -> void {
  onDrawGui_GUID(guid);
}

auto MaterialElucidator::onDrawGui_GUID(Core::GUID guid) noexcept -> void {
  GFX::Material* mat =
      Core::ResourceManager::get()->getResource<GFX::Material>(guid);
  onDrawGui_PTR(mat);
}

auto MaterialElucidator::onDrawGui_PTR(GFX::Material* material) noexcept
    -> void {
  ImGui::BulletText(("Name: " + material->name).c_str());
  ImGui::BulletText(("Path: " + material->path).c_str());
  if (ImGui::TreeNode("Textures:")) {
    uint32_t id = 0;
    for (auto& [name, texture] : material->textures) {
      ImGui::PushID(id);
      if (ImGui::TreeNode(
              (("Texture - " + std::to_string(id) + " - " + name).c_str()))) {
        TextureElucidator::onDrawGui_GUID(texture.guid);
        ImGui::TreePop();
      }
      ++id;
      ImGui::PopID();
    }
    ImGui::TreePop();
  }
  ImGui::BulletText(
      ("Emissive: " + std::to_string(material->isEmissive)).c_str());
}

auto VideoClipElucidator::onDrawGui(Core::GUID guid) noexcept -> void {
  onDrawGui_GUID(guid);
}

auto VideoClipElucidator::onDrawGui_GUID(Core::GUID guid) noexcept -> void {
  GFX::VideoClip* vc =
      Core::ResourceManager::get()->getResource<GFX::VideoClip>(guid);
  onDrawGui_PTR(vc);
}

auto VideoClipElucidator::onDrawGui_PTR(GFX::VideoClip* vc) noexcept -> void {
  ImGui::BulletText(("Name: " + vc->name).c_str());
  ImGui::BulletText(("Path: " + vc->resourcePath.value()).c_str());
  ImGui::BulletText(("Width: " + std::to_string(vc->decoder.width)).c_str());
  ImGui::BulletText(("Height: " + std::to_string(vc->decoder.height)).c_str());
  if (ImGui::TreeNode("Texture Binded:")) {
    TextureElucidator::onDrawGui_PTR(vc->decoder.device_texture);
    ImGui::TreePop();
  }
}
}  // namespace SIByL::Editor

namespace SIByL::Editor {
 auto InspectorWidget::onDrawGui() noexcept -> void {
   ImGui::Begin("Inspector", 0, ImGuiWindowFlags_MenuBar);
   if (customDraw) customDraw();
   ImGui::End();
 }

 auto InspectorWidget::setCustomDraw(std::function<void()> func) noexcept ->
 void {
   customDraw = func;
 }

 auto InspectorWidget::setEmpty() noexcept -> void {
   setCustomDraw([]() {});
 }

 auto CustomInspector::setInspectorWidget(InspectorWidget* widget) noexcept
     -> void {
   widget->setCustomDraw(std::bind(&CustomInspector::onDrawGui, this));
 }
}  // namespace SIByL::Editor

namespace SIByL::Editor {
auto StatusWidget::onDrawGui() noexcept -> void {
   ImGui::Begin("Status");
   ImGui::Text(("fps:\t" + std::to_string(1. / timer->deltaTime())).c_str());
   ImGui::Text(("time:\t" + std::to_string(timer->deltaTime())).c_str());
   ImGui::End();
}
}  // namespace SIByL::Editor

namespace SIByL::Editor {
auto LogWidget::setCustomDraw(std::function<void()> func) noexcept -> void {
   customDraw = func;
}

LogWidget::LogWidget() {
   autoScroll = true;
   clear();
   Core::LogManager::get()->editorCallback =
       std::bind(&LogWidget::addline, this, std::placeholders::_1);
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
   if (ImGui::Button("Options")) ImGui::OpenPopup("Options");
   ImGui::SameLine();
   bool bclear = ImGui::Button("Clear");
   ImGui::SameLine();
   bool copy = ImGui::Button("Copy");
   ImGui::SameLine();
   filter.Draw("Filter", -100.0f);

   ImGui::Separator();
   ImGui::BeginChild("scrolling", ImVec2(0, 0), false,
                     ImGuiWindowFlags_HorizontalScrollbar);

   if (bclear) clear();
   if (copy) ImGui::LogToClipboard();

   ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 0));
   const char* buf = _buf.begin();
   const char* buf_end = _buf.end();

   for (int line_no = 0; line_no < lineOffsets.Size; line_no++) {
    const char* line_start = buf + lineOffsets[line_no];
    const char* line_end = (line_no + 1 < lineOffsets.Size)
                               ? (buf + lineOffsets[line_no + 1] - 1)
                               : buf_end;
    if (!filter.IsActive() || filter.PassFilter(line_start, line_end)) {
      if (line_start[1] == 'W') {
        ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 190, 0, 255));
        ImGui::TextUnformatted(line_start, line_end);
        ImGui::PopStyleColor();
      } else if (line_start[1] == 'D') {
        ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(50, 180, 255, 255));
        ImGui::TextUnformatted(line_start, line_end);
        ImGui::PopStyleColor();
      } else if (line_start[1] == 'E') {
        ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 30, 61, 255));
        ImGui::TextUnformatted(line_start, line_end);
        ImGui::PopStyleColor();
      } else {
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
}  // namespace SIByL::Editor

namespace SIByL::Editor {
auto captureImage(Core::GUID src) noexcept -> void {
   RHI::RHILayer* rhiLayer = ImGuiLayer::get()->rhiLayer;
   Platform::Window* mainWindow =
       rhiLayer->getRHILayerDescriptor().windowBinded;
   GFX::Texture* tex =
       Core::ResourceManager::get()->getResource<GFX::Texture>(src);
   size_t width = tex->texture->width();
   size_t height = tex->texture->height();

   RHI::TextureFormat format;
   size_t pixelSize;
   if (tex->texture->format() == RHI::TextureFormat::RGBA32_FLOAT) {
    format = RHI::TextureFormat::RGBA32_FLOAT;
    pixelSize = sizeof(Math::vec4);
   } else if (tex->texture->format() == RHI::TextureFormat::RGBA8_UNORM) {
    format = RHI::TextureFormat::RGBA8_UNORM;
    pixelSize = sizeof(uint8_t) * 4;
   } else {
    Core::LogManager::Error(
        "Editor :: ViewportWidget :: captureImage() :: Unsupported format to "
        "capture.");
    return;
   }

   std::unique_ptr<RHI::CommandEncoder> commandEncoder =
       rhiLayer->getDevice()->createCommandEncoder({});

   static Core::GUID copyDst = 0;
   if (copyDst == 0) {
    copyDst = Core::ResourceManager::get()->requestRuntimeGUID<GFX::Texture>();
    RHI::TextureDescriptor desc{
        {width, height, 1},
        1,
        1,
        RHI::TextureDimension::TEX2D,
        format,
        (uint32_t)RHI::TextureUsage::COPY_DST |
            (uint32_t)RHI::TextureUsage::TEXTURE_BINDING,
        {format},
        RHI::TextureFlags::HOSTI_VISIBLE};
    GFX::GFXManager::get()->registerTextureResource(copyDst, desc);

    GFX::Texture* copydst =
        Core::ResourceManager::get()->getResource<GFX::Texture>(copyDst);
    commandEncoder->pipelineBarrier(RHI::BarrierDescriptor{
        (uint32_t)RHI::PipelineStages::ALL_GRAPHICS_BIT,
        (uint32_t)RHI::PipelineStages::TRANSFER_BIT,
        (uint32_t)RHI::DependencyType::NONE,
        {},
        {},
        {RHI::TextureMemoryBarrierDescriptor{
            copydst->texture.get(),
            RHI::ImageSubresourceRange{(uint32_t)RHI::TextureAspect::COLOR_BIT,
                                       0, 1, 0, 1},
            (uint32_t)RHI::AccessFlagBits::NONE,
            (uint32_t)RHI::AccessFlagBits::TRANSFER_WRITE_BIT,
            RHI::TextureLayout::SHADER_READ_ONLY_OPTIMAL,
            RHI::TextureLayout::TRANSFER_DST_OPTIMAL}}});
   }
   rhiLayer->getDevice()->waitIdle();
   commandEncoder->pipelineBarrier(RHI::BarrierDescriptor{
       (uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT,
       (uint32_t)RHI::PipelineStages::TRANSFER_BIT,
       (uint32_t)RHI::DependencyType::NONE,
       {},
       {},
       {RHI::TextureMemoryBarrierDescriptor{
           Core::ResourceManager::get()
               ->getResource<GFX::Texture>(src)
               ->texture.get(),
           RHI::ImageSubresourceRange{(uint32_t)RHI::TextureAspect::COLOR_BIT,
                                      0, 1, 0, 1},
           (uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT |
               (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT,
           (uint32_t)RHI::AccessFlagBits::TRANSFER_READ_BIT,
           RHI::TextureLayout::SHADER_READ_ONLY_OPTIMAL,
           RHI::TextureLayout::TRANSFER_SRC_OPTIMAL}}});
   commandEncoder->copyTextureToTexture(
       RHI::ImageCopyTexture{Core::ResourceManager::get()
                                 ->getResource<GFX::Texture>(src)
                                 ->texture.get()},
       RHI::ImageCopyTexture{Core::ResourceManager::get()
                                 ->getResource<GFX::Texture>(copyDst)
                                 ->texture.get()},
       RHI::Extend3D{uint32_t(width), uint32_t(height), 1});
   commandEncoder->pipelineBarrier(RHI::BarrierDescriptor{
       (uint32_t)RHI::PipelineStages::TRANSFER_BIT,
       (uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT,
       (uint32_t)RHI::DependencyType::NONE,
       {},
       {},
       {RHI::TextureMemoryBarrierDescriptor{
           Core::ResourceManager::get()
               ->getResource<GFX::Texture>(src)
               ->texture.get(),
           RHI::ImageSubresourceRange{(uint32_t)RHI::TextureAspect::COLOR_BIT,
                                      0, 1, 0, 1},
           (uint32_t)RHI::AccessFlagBits::TRANSFER_READ_BIT,
           (uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT |
               (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT,
           RHI::TextureLayout::TRANSFER_SRC_OPTIMAL,
           RHI::TextureLayout::SHADER_READ_ONLY_OPTIMAL}}});
   commandEncoder->pipelineBarrier(RHI::BarrierDescriptor{
       (uint32_t)RHI::PipelineStages::TRANSFER_BIT,
       (uint32_t)RHI::PipelineStages::HOST_BIT,
       (uint32_t)RHI::DependencyType::NONE,
       {},
       {},
       {RHI::TextureMemoryBarrierDescriptor{
           Core::ResourceManager::get()
               ->getResource<GFX::Texture>(copyDst)
               ->texture.get(),
           RHI::ImageSubresourceRange{(uint32_t)RHI::TextureAspect::COLOR_BIT,
                                      0, 1, 0, 1},
           (uint32_t)RHI::AccessFlagBits::TRANSFER_WRITE_BIT,
           (uint32_t)RHI::AccessFlagBits::HOST_READ_BIT,
           RHI::TextureLayout::TRANSFER_DST_OPTIMAL,
           RHI::TextureLayout::TRANSFER_DST_OPTIMAL}}});
   rhiLayer->getDevice()->getGraphicsQueue()->submit(
       {commandEncoder->finish({})});
   rhiLayer->getDevice()->waitIdle();
   std::future<bool> mapped =
       Core::ResourceManager::get()
           ->getResource<GFX::Texture>(copyDst)
           ->texture->mapAsync((uint32_t)RHI::MapMode::READ, 0,
                               width * height * pixelSize);
   if (mapped.get()) {
    void* data = Core::ResourceManager::get()
                     ->getResource<GFX::Texture>(copyDst)
                     ->texture->getMappedRange(0, width * height * pixelSize);
    if (tex->texture->format() == RHI::TextureFormat::RGBA32_FLOAT) {
      std::string filepath = mainWindow->saveFile(
          "", Core::WorldTimePoint::get().to_string() + ".hdr");
      Image::HDR::writeHDR(filepath, width, height, 4,
                           reinterpret_cast<float*>(data));
    } else if (tex->texture->format() == RHI::TextureFormat::RGBA8_UNORM) {
      std::string filepath = mainWindow->saveFile(
          "", Core::WorldTimePoint::get().to_string() + ".bmp");
      Image::BMP::writeBMP(filepath, width, height, 4,
                           reinterpret_cast<float*>(data));
    }
    Core::ResourceManager::get()
        ->getResource<GFX::Texture>(copyDst)
        ->texture->unmap();
   }
}

auto ViewportWidget::setTarget(std::string const& name,
                               GFX::Texture* tex) noexcept -> void {
   this->name = name;
   texture = tex;
}

/** draw gui*/
auto ViewportWidget::onDrawGui() noexcept -> void {
   ImGui::Begin(name.c_str(), 0, ImGuiWindowFlags_MenuBar);

   ImGui::PushItemWidth(ImGui::GetFontSize() * -12);
   if (ImGui::BeginMenuBar()) {
    if (ImGui::Button("capture")) {
      if (texture) {
        captureImage(texture->guid);
      }
    }
    // menuBarSize = ImGui::GetWindowSize();
    ImGui::EndMenuBar();
   }
   ImGui::PopItemWidth();
   commonOnDrawGui();
   auto currPos = ImGui::GetCursorPos();
   info.mousePos = ImGui::GetMousePos();
   info.mousePos.x -= info.windowPos.x + currPos.x;
   info.mousePos.y -= info.windowPos.y + currPos.y;
   // info.contentPos = info.windowPos;
   // info.contentPos.x += currPos.x;
   // info.contentPos.y += currPos.y;

   if (texture) {
    ImGui::Image(
        Editor::TextureUtils::getImGuiTexture(texture->guid)->getTextureID(),
        {(float)texture->texture->width(), (float)texture->texture->height()},
        {0, 0}, {1, 1});
   }

   ImGui::End();
}
}  // namespace SIByL::Editor

namespace SIByL::Editor {
auto ContentWidget::onDrawGui() noexcept -> void {
   if (!ImGui::Begin("Content Browser", 0)) {
    ImGui::End();
    return;
   }

   if (modal_state.showModal) ImGui::OpenPopup("Warning");
   if (ImGui::BeginPopupModal("Warning")) {
    ImGui::Text("Selected resource not registered in resource libarary!");
    if (ImGui::Button("OK", ImVec2(120, 0))) {
      ImGui::CloseCurrentPopup();
      modal_state.showModal = false;
    }
    ImGui::SameLine();
    if (ImGui::Button("Load", ImVec2(120, 0))) {
      if (modal_state.entry_ptr->resourceLoader == nullptr) {
        Core::LogManager::Error(
            "GFXEditor :: Resource loading of this type is not supported yet!");
      } else {
        Core::GUID guid =
            modal_state.entry_ptr->resourceLoader(modal_state.path.c_str());
        inspectorWidget->setCustomDraw(std::bind(
            &(ResourceViewer::onDrawGui), &(inspectorWidget->resourceViewer),
            modal_state.entry_ptr->resourceName.c_str(), guid));
        ImGui::CloseCurrentPopup();
        modal_state.showModal = false;
      }
    }
   }

   {
    ImGuiWindowFlags window_flags = ImGuiWindowFlags_None;
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 5.0f);
    ImGui::BeginChild("ChildL",
                      ImVec2(ImGui::GetContentRegionAvail().x * 0.2f, 0), true,
                      window_flags);
    // Select content browser source
    {
      ImGui::AlignTextToFramePadding();
      bool const selectFileExplorer =
          ImGui::TreeNodeEx("File Exploror", ImGuiTreeNodeFlags_Leaf);
      if (ImGui::IsItemClicked()) {
        source = Source::FileSystem;
      }
      if (selectFileExplorer) {
        ImGui::TreePop();
      }
      if (ImGui::TreeNodeEx("Runtime Resources")) {
        auto const& resourcePool =
            Core::ResourceManager::get()->getResourcePool();
        for (auto const& pair : resourcePool) {
          bool const selectResourceType =
              ImGui::TreeNodeEx(pair.first, ImGuiTreeNodeFlags_Leaf);
          if (ImGui::IsItemClicked()) {
            runtimeResourceInfo.type = pair.first;
            runtimeResourceInfo.GUIDs = &pair.second->getAllGUID();
            source = Source::RuntimeResources;
          }
          if (selectResourceType) {
            ImGui::TreePop();
          }
        }
        ImGui::TreePop();
      }
    }
    ImGui::EndChild();
    ImGui::PopStyleVar();
   }
   ImGui::SameLine();

   // Child 2: rounded border
   {
    ImGuiWindowFlags window_flags = ImGuiWindowFlags_None;
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 5.0f);
    ImGui::BeginChild("ChildR", ImVec2(0, 0), true, window_flags);
    {
      if (source == Source::FileSystem) {
        static std::filesystem::path root("./content");
        int iconCount = 0;
        if (root.compare(currentDirectory)) ++iconCount;
        //// calculate column number
        static float padding = 16.0f * ImGuiLayer::get()->getDPI();
        static float thumbnailSize = 64.f * ImGuiLayer::get()->getDPI();
        float cellSize = thumbnailSize + padding;
        float panelWidth = ImGui::GetContentRegionAvail().x;
        int columnCount = (int)(panelWidth / cellSize);
        if (columnCount < 1) columnCount = 1;
        // Do Columns
        int subdirCount =
            std::distance(std::filesystem::directory_iterator(currentDirectory),
                          std::filesystem::directory_iterator{});
        ImGui::Text(
            ("File Explorer Path: " + currentDirectory.string()).c_str());
        if (ImGui::BeginTable(
                "FileSystemBrowser", columnCount,
                ImGuiTableFlags_Resizable | ImGuiTableFlags_NoBordersInBody)) {
          ImGui::TableNextRow();
          {
            int iconIdx = 0;
            // for back
            if (root.compare(currentDirectory)) {
              ImGui::TableSetColumnIndex(iconIdx);
              ImGui::PushID("Back");
              ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
              ImGuiTexture* icon =
                  Editor::TextureUtils::getImGuiTexture(icons.back);
              ImGui::ImageButton(icon->getTextureID(),
                                 {thumbnailSize, thumbnailSize}, {0, 0},
                                 {1, 1});
              ImGui::PopStyleColor();
              if (ImGui::IsItemHovered() &&
                  ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
                currentDirectory = currentDirectory.parent_path();
              }
              ImGui::TextWrapped("       Back");
              ImGui::PopID();
              // set table position
              ++iconIdx;
              if (iconIdx >= columnCount) {
                iconIdx = 0;
                ImGui::TableNextRow();
              }
            }
            // for each subdir
            std::vector<std::filesystem::directory_entry> path_sorted = {};
            std::vector<std::filesystem::directory_entry> folder_pathes = {};
            std::vector<std::vector<std::filesystem::directory_entry>>
                other_pathes(resourceRegistries.size());
            std::vector<std::filesystem::directory_entry> unkown_pathes = {};
            for (auto& directoryEntry :
                 std::filesystem::directory_iterator(currentDirectory)) {
              const auto& path = directoryEntry.path();
              std::string filenameExtension = path.extension().string();
              if (directoryEntry.is_directory()) {
                folder_pathes.push_back(directoryEntry);
              } else {
                size_t idx = 0;
                for (auto const& entry : resourceRegistries) {
                  if (entry.matchExtensions(filenameExtension)) {
                    other_pathes[idx].push_back(directoryEntry);
                    break;
                  }
                  ++idx;
                }
              }
            }
            path_sorted.insert(path_sorted.end(), folder_pathes.begin(),
                               folder_pathes.end());
            for (auto& vec : other_pathes)
              path_sorted.insert(path_sorted.end(), vec.begin(), vec.end());
            path_sorted.insert(path_sorted.end(), unkown_pathes.begin(),
                               unkown_pathes.end());
            for (auto& directoryEntry : path_sorted) {
              ImGui::TableSetColumnIndex(iconIdx);
              const auto& path = directoryEntry.path();
              std::string pathString = path.string();
              auto relativePath =
                  std::filesystem::relative(directoryEntry.path(), root);
              std::string relativePathString = relativePath.string();
              std::string filenameString = relativePath.filename().string();
              std::string filenameExtension = relativePath.extension().string();

              ImGui::PushID(filenameString.c_str());
              // If is directory
              ImGuiTexture* icon =
                  Editor::TextureUtils::getImGuiTexture(icons.file);
              ResourceRegistry const* entry_ptr = nullptr;
              if (directoryEntry.is_directory()) {
                icon = Editor::TextureUtils::getImGuiTexture(icons.folder);
              } else {
                for (auto const& entry : resourceRegistries) {
                  if (entry.matchExtensions(filenameExtension)) {
                    icon = Editor::TextureUtils::getImGuiTexture(
                        entry.resourceIcon);
                    entry_ptr = &entry;
                    break;
                  }
                }
              }

              ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
              ImGui::ImageButton(icon->getTextureID(),
                                 {thumbnailSize, thumbnailSize}, {0, 0},
                                 {1, 1});

              // if (ImGui::BeginDragDropSource()) {
              //	ImGui::Text("Dragged");

              //	const wchar_t* itemPath = relativePath.c_str();
              //	ImGui::SetDragDropPayload("ASSET", itemPath,
              //(wcslen(itemPath) + 1) * sizeof(wchar_t));
              //	ImGui::EndDragDropSource();
              //}
              ImGui::PopStyleColor();
              if (ImGui::IsItemHovered() &&
                  ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
                if (directoryEntry.is_directory()) {
                  currentDirectory /= path.filename();
                } else if (entry_ptr && entry_ptr->guidFinder) {
                  Core::ORID orid =
                      Core::ResourceManager::get()->database.findResourcePath(
                          path.string().c_str());
                  if (orid != Core::INVALID_ORID) {
                    Core::GUID guid = entry_ptr->guidFinder(orid);
                    inspectorWidget->setCustomDraw(
                        std::bind(&(ResourceViewer::onDrawGui),
                                  &(inspectorWidget->resourceViewer),
                                  entry_ptr->resourceName.c_str(), guid));
                  } else {
                    modal_state.showModal = true;
                    modal_state.path = path.string();
                    modal_state.entry_ptr = entry_ptr;
                  }
                }
              }
              ImGui::TextWrapped(filenameString.c_str());
              ImGui::PopID();
              // set table position
              ++iconIdx;
              if (iconIdx >= columnCount) {
                iconIdx = 0;
                ImGui::TableNextRow();
              }
            }
          }
          ImGui::EndTable();
        }
      } else if (source == Source::RuntimeResources) {
        for (auto const& GUID : *runtimeResourceInfo.GUIDs) {
          std::string GUIDstr = std::to_string(GUID);
          if (GUIDstr.length() < 20) {
            GUIDstr = std::string(20 - GUIDstr.length(), '0') + GUIDstr;
          }
          bool const selectResource =
              ImGui::TreeNodeEx((GUIDstr + " :: " +
                                 Core::ResourceManager::get()->getResourceName(
                                     runtimeResourceInfo.type, GUID))
                                    .c_str(),
                                ImGuiTreeNodeFlags_Leaf);
          if (ImGui::IsItemClicked()) {
            inspectorWidget->setCustomDraw(
                std::bind(&(ResourceViewer::onDrawGui),
                          &(inspectorWidget->resourceViewer),
                          runtimeResourceInfo.type, GUID));
          }
          if (selectResource) {
            ImGui::TreePop();
          }
        }
      }
    }
    ImGui::EndChild();
    ImGui::PopStyleVar();
   }
   ImGui::End();
}

auto ContentWidget::reigsterIconResources() noexcept -> void {
   icons.back = GFX::GFXManager::get()->registerTextureResource(
       "../Engine/Binaries/Runtime/icons/back.png");
   icons.folder = GFX::GFXManager::get()->registerTextureResource(
       "../Engine/Binaries/Runtime/icons/folder.png");
   icons.file = GFX::GFXManager::get()->registerTextureResource(
       "../Engine/Binaries/Runtime/icons/file.png");
   icons.image = GFX::GFXManager::get()->registerTextureResource(
       "../Engine/Binaries/Runtime/icons/image.png");
   icons.material = GFX::GFXManager::get()->registerTextureResource(
       "../Engine/Binaries/Runtime/icons/material.png");
   icons.mesh = GFX::GFXManager::get()->registerTextureResource(
       "../Engine/Binaries/Runtime/icons/mesh.png");
   icons.scene = GFX::GFXManager::get()->registerTextureResource(
       "../Engine/Binaries/Runtime/icons/scene.png");
   icons.shader = GFX::GFXManager::get()->registerTextureResource(
       "../Engine/Binaries/Runtime/icons/shader.png");
   icons.video = GFX::GFXManager::get()->registerTextureResource(
       "../Engine/Binaries/Runtime/icons/video.png");
   Core::ResourceManager::get()
       ->getResource<GFX::Texture>(icons.back)
       ->texture->setName("editor_icon_back");
   Core::ResourceManager::get()
       ->getResource<GFX::Texture>(icons.folder)
       ->texture->setName("editor_icon_folder");
   Core::ResourceManager::get()
       ->getResource<GFX::Texture>(icons.file)
       ->texture->setName("editor_icon_file");
   Core::ResourceManager::get()
       ->getResource<GFX::Texture>(icons.image)
       ->texture->setName("editor_icon_image");
   Core::ResourceManager::get()
       ->getResource<GFX::Texture>(icons.material)
       ->texture->setName("editor_icon_material");
   Core::ResourceManager::get()
       ->getResource<GFX::Texture>(icons.mesh)
       ->texture->setName("editor_icon_mesh");
   Core::ResourceManager::get()
       ->getResource<GFX::Texture>(icons.scene)
       ->texture->setName("editor_icon_scene");
   Core::ResourceManager::get()
       ->getResource<GFX::Texture>(icons.shader)
       ->texture->setName("editor_icon_shader");
   Core::ResourceManager::get()
       ->getResource<GFX::Texture>(icons.video)
       ->texture->setName("editor_icon_videor");
}
}  // namespace SIByL::Editor

namespace SIByL::Editor {
/** draw each fragments */
auto GameObjectInspector::onDrawGui() noexcept -> void {
   for (auto& frag : fragmentSequence) {
    frag->onDrawGui(0, &data);
   }
   {  // add component
    ImGui::Separator();
    ImVec2 contentRegionAvailable = ImGui::GetContentRegionAvail();
    ImVec2 buttonSize(200, 30);
    ImGui::SetCursorPosX(contentRegionAvailable.x / 2 - 100 + 20);
    if (ImGui::Button(" Add Component", buttonSize))
      ImGui::OpenPopup("AddComponent");
    if (ImGui::BeginPopup("AddComponent")) {
      Core::Entity entity = data.scene->getGameObject(data.handle)->getEntity();
      for (auto& pair : componentsRegister) {
        if (pair.second.get()->getComponent(entity) == nullptr) {
          if (ImGui::MenuItem(pair.first.c_str())) {
            pair.second.get()->addComponent(entity);
            ImGui::CloseCurrentPopup();
          }
        }
      }
      ImGui::EndPopup();
    }
   }
}

auto SceneWidget::onDrawGui() noexcept -> void {
   ImGui::Begin("Scene", 0, ImGuiWindowFlags_MenuBar);
   ImGui::PushItemWidth(ImGui::GetFontSize() * -12);
   if (ImGui::BeginMenuBar()) {
    if (ImGui::Button("New")) {
    }
    if (ImGui::Button("Save")) {
      if (scene != nullptr) {
        std::string name = scene->name + ".scene";
        std::string path = ImGuiLayer::get()
                               ->rhiLayer->getRHILayerDescriptor()
                               .windowBinded->saveFile(nullptr, name);
        if (path != "") {
          scene->serialize(path);
          scene->isDirty = false;
        }
      }
    }
    if (ImGui::Button("Load")) {
      if (scene != nullptr) {
        std::string path = ImGuiLayer::get()
                               ->rhiLayer->getRHILayerDescriptor()
                               .windowBinded->openFile("scene");
        if (path != "") scene->deserialize(path);
        // scene->isDirty = false;
      }
    }
    if (ImGui::BeginMenu("Import")) {
      // Menu - File - Load
      if (ImGui::MenuItem("glTF 2.0 (.glb/.gltf)")) {
        std::string path = ImGuiLayer::get()
                               ->rhiLayer->getRHILayerDescriptor()
                               .windowBinded->openFile(".gltf");
        // GFX::SceneNodeLoader_obj::loadSceneNode(path, *scene,
        // SRenderer::meshLoadConfig);
      }
      if (ImGui::MenuItem("Wavefront (.obj)")) {
        std::string path = ImGuiLayer::get()
                               ->rhiLayer->getRHILayerDescriptor()
                               .windowBinded->openFile(".obj");
        GFX::SceneNodeLoader_obj::loadSceneNode(
            path, *scene, GFX::GFXManager::get()->config.meshLoaderConfig);
      }
      if (ImGui::MenuItem("FBX (.fbx)")) {
        std::string path = ImGuiLayer::get()
                               ->rhiLayer->getRHILayerDescriptor()
                               .windowBinded->openFile(".fbx");
        GFX::SceneNodeLoader_assimp::loadSceneNode(
            path, *scene, GFX::GFXManager::get()->config.meshLoaderConfig);
      }
      ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("Export")) {
      ImGui::EndMenu();
    }
    //// Menu - File
    // if (ImGui::BeginMenu("File")) {
    //	// Menu - File - Load
    //	if (ImGui::MenuItem("Load")) {
    //		std::string path = windowLayer->getWindow()->openFile("");
    //		hold_scene = MemNew<GFX::Scene>();
    //		bindScene(hold_scene.get());
    //		binded_scene->deserialize(path, assetLayer);
    //	}
    //	// Menu - File - Save
    //	bool should_save_as = false;
    //	if (ImGui::MenuItem("Save")) {
    //		if (currentPath == std::string()) should_save_as = true;
    //		else {}
    //	}
    //	// Menu - File - Save as
    //	if (ImGui::MenuItem("Save as") || should_save_as) {
    //		std::string path = windowLayer->getWindow()->saveFile("");
    //		binded_scene->serialize(path);
    //	}
    //	ImGui::EndMenu();
    // }
    ImGui::EndMenuBar();
   }
   ImGui::PopItemWidth();

   // Left-clock on blank space
   if (ImGui::IsMouseDown(0) && ImGui::IsWindowHovered()) {
    if (inspectorWidget) inspectorWidget->setEmpty();
    // if (viewport) viewport->selectedEntity = {};
   }
   // Right-click on blank space
   if (ImGui::BeginPopupContextWindow(0, 1, false)) {
    if (ImGui::MenuItem("Create Empty Entity") && scene) {
      scene->createGameObject(GFX::NULL_GO);
      ImGui::SetNextItemOpen(true, ImGuiCond_Always);
    }
    ImGui::EndPopup();
   }
   // Draw scene hierarchy
   if (scene) {
    std::vector<GFX::GameObjectHandle> rootHandles = {};
    for (auto iter = scene->gameObjects.begin();
         iter != scene->gameObjects.end(); ++iter) {
      if (iter->second.parent == GFX::NULL_GO)
        rootHandles.push_back(iter->first);
    }
    for (auto handle : rootHandles) drawNode(handle);
   }
   ImGui::End();
}

auto SceneWidget::drawNode(GFX::GameObjectHandle const& node) -> bool {
   ImGui::PushID(node);
   GFX::TagComponent* tag = scene->getGameObject(node)
                                ->getEntity()
                                .getComponent<GFX::TagComponent>();
   ImGuiTreeNodeFlags node_flags = 0;
   if (scene->getGameObject(node)->children.size() == 0)
    node_flags |= ImGuiTreeNodeFlags_Leaf;
   if (node == forceNodeOpen) {
    ImGui::SetNextItemOpen(true, ImGuiCond_Always);
    forceNodeOpen = GFX::NULL_GO;
   }
   bool opened = ImGui::TreeNodeEx(tag->name.c_str(), node_flags);
   ImGuiID uid = ImGui::GetID(tag->name.c_str());
   ImGui::TreeNodeBehaviorIsOpen(uid);
   // Clicked
   if (ImGui::IsItemClicked()) {
    // ECS::Entity entity = binded_scene->tree.getNodeEntity(node);
    inspected = node;
    gameobjectInspector.data.scene = scene;
    gameobjectInspector.data.handle = node;
    if (inspectorWidget)
      gameobjectInspector.setInspectorWidget(inspectorWidget);
    // if (viewport) viewport->selectedEntity = entity;
   }
   // Right-click on blank space
   bool entityDeleted = false;
   if (ImGui::BeginPopupContextItem()) {
    if (ImGui::MenuItem("Create Empty Entity")) {
      scene->createGameObject(node);
      forceNodeOpen = node;
    }
    if (ImGui::MenuItem("Delete Entity")) entityDeleted = true;
    ImGui::EndPopup();
   }
   // If draged
   if (ImGui::BeginDragDropSource()) {
    ImGui::Text(tag->name.c_str());
    ImGui::SetDragDropPayload("SceneEntity", &node, sizeof(node));
    ImGui::EndDragDropSource();
   }
   // If dragged to
   if (ImGui::BeginDragDropTarget()) {
    if (const ImGuiPayload* payload =
            ImGui::AcceptDragDropPayload("SceneEntity")) {
      GFX::GameObjectHandle* dragged_handle = (uint64_t*)payload->Data;
      // binded_scene->tree.moveNode(*dragged_handle, node);
    }
    ImGui::EndDragDropTarget();
   }
   // Opened
   if (opened) {
    ImGui::NextColumn();
    for (int i = 0; i < scene->getGameObject(node)->children.size(); i++) {
      drawNode(scene->getGameObject(node)->children[i]);
    }
    ImGui::TreePop();
   }
   ImGui::PopID();
   if (entityDeleted) {
    bool isParentOfInspected = false;
    GFX::GameObjectHandle inspected_parent = inspected;
    while (inspected_parent != GFX::NULL_GO) {
      inspected_parent = scene->getGameObject(inspected_parent)->parent;
      if (node == node) {
        isParentOfInspected = true;
        break;
      }  // If set ancestor as child, no movement;
    }
    if (node == inspected || isParentOfInspected) {
      if (inspectorWidget) inspectorWidget->setEmpty();
      // if (viewport) viewport->selectedEntity = {};
      inspected = 0;
    }
    scene->removeGameObject(node);
    return true;
   }
   return false;
}
}  // namespace SIByL::Editor

namespace SIByL::Editor {
auto CameraState::setFromTransform(
    GFX::TransformComponent const& transform) noexcept -> void {
   Math::vec3 eulerAngles = transform.eulerAngles;
   Math::vec3 translation = transform.translation;

   pitch = eulerAngles.x;
   yaw = eulerAngles.y;
   roll = eulerAngles.z;

   x = translation.x;
   y = translation.y;
   z = translation.z;
}

auto CameraState::lerpTowards(CameraState const& target, float positionLerpPct,
                              float rotationLerpPct) noexcept -> void {
   yaw = std::lerp(yaw, target.yaw, rotationLerpPct);
   pitch = std::lerp(pitch, target.pitch, rotationLerpPct);
   roll = std::lerp(roll, target.roll, rotationLerpPct);

   x = std::lerp(x, target.x, positionLerpPct);
   y = std::lerp(y, target.y, positionLerpPct);
   z = std::lerp(z, target.z, positionLerpPct);
}

auto CameraState::updateTransform(GFX::TransformComponent& transform) noexcept
    -> void {
   transform.eulerAngles = Math::vec3(pitch, yaw, roll);
   transform.translation = Math::vec3(x, y, z);
}

auto SimpleCameraController::onEnable(
    GFX::TransformComponent const& transform) noexcept -> void {
   targetCameraState.setFromTransform(transform);
   interpolatingCameraState.setFromTransform(transform);
}

auto SimpleCameraController::getInputTranslationDirection() noexcept
    -> Math::vec3 {
   Math::vec3 direction(0.0f, 0.0f, 0.0f);
   if (input->isKeyPressed(Platform::SIByL_KEY_W)) {
    direction += Math::vec3(0, 0, +1);  // forward
   }
   if (input->isKeyPressed(Platform::SIByL_KEY_S)) {
    direction += Math::vec3(0, 0, -1);  // back
   }
   if (input->isKeyPressed(Platform::SIByL_KEY_A)) {
    direction += Math::vec3(-1, 0, 0);  // left
   }
   if (input->isKeyPressed(Platform::SIByL_KEY_D)) {
    direction += Math::vec3(1, 0, 0);  // right
   }
   if (input->isKeyPressed(Platform::SIByL_KEY_Q)) {
    direction += Math::vec3(0, -1, 0);  // down
   }
   if (input->isKeyPressed(Platform::SIByL_KEY_E)) {
    direction += Math::vec3(0, 1, 0);  // up
   }
   return direction;
}

auto SimpleCameraController::bindTransform(
    GFX::TransformComponent* transform) noexcept -> void {
   if (this->transform != transform) {
    targetCameraState.setFromTransform(*transform);
    interpolatingCameraState.setFromTransform(*transform);
    this->transform = transform;
   }
}

auto SimpleCameraController::onUpdate() noexcept -> void {
   // check the viewport is hovered
   bool hovered = viewport->info.isHovered;
   // rotation
   static bool justPressedMouse = true;
   static float last_x = 0;
   static float last_y = 0;
   static bool inRotationMode = false;

   if (input->isMouseButtonPressed(Platform::SIByL_MOUSE_BUTTON_2) &&
       viewport->info.isHovered && viewport->info.isFocused)
    inRotationMode = true;
   if (!input->isMouseButtonPressed(Platform::SIByL_MOUSE_BUTTON_2)) {
    inRotationMode = false;
   }

   if (input->isMouseButtonPressed(Platform::SIByL_MOUSE_BUTTON_2)) {
    if (inRotationMode) {
      input->disableCursor();
      float x = input->getMouseX();
      float y = input->getMouseY();
      if (justPressedMouse) {
        last_x = x;
        last_y = y;
        justPressedMouse = false;
      } else {
        Math::vec2 mouseMovement = Math::vec2(x - last_x, y - last_y) *
                                   0.0005f * mouseSensitivityMultiplier *
                                   mouseSensitivity;
        if (invertY) mouseMovement.y = -mouseMovement.y;
        last_x = x;
        last_y = y;

        float mouseSensitivityFactor =
            mouseSensitivityCurve.evaluate(mouseMovement.length()) * 180. /
            3.1415926;

        targetCameraState.yaw += -mouseMovement.x * mouseSensitivityFactor;
        targetCameraState.pitch += mouseMovement.y * mouseSensitivityFactor;
      }
    }
   } else if (!justPressedMouse) {
    input->enableCursor();
    justPressedMouse = true;
   }

   // translation
   Math::vec3 translation = getInputTranslationDirection();
   translation *= timer->deltaTime() * 0.1;

   // speed up movement when shift key held
   if (input->isKeyPressed(Platform::SIByL_KEY_LEFT_SHIFT)) {
    translation *= 10.0f;
   }

   // modify movement by a boost factor ( defined in Inspector and modified in
   // play mode through the mouse scroll wheel)
   float y = input->getMouseScrollY();
   boost += y * 0.01f;
   translation *= powf(2.0f, boost);

   Math::vec4 rotatedFoward4 = Math::mat4::rotateZ(targetCameraState.roll) *
                               Math::mat4::rotateY(targetCameraState.yaw) *
                               Math::mat4::rotateX(targetCameraState.pitch) *
                               Math::vec4(0, 0, -1, 0);
   Math::vec3 rotatedFoward =
       Math::vec3(rotatedFoward4.x, rotatedFoward4.y, rotatedFoward4.z);
   Math::vec3 up = Math::vec3(0.0f, 1.0f, 0.0f);
   Math::vec3 cameraRight = Math::normalize(Math::cross(rotatedFoward, up));
   Math::vec3 cameraUp = Math::cross(cameraRight, rotatedFoward);
   Math::vec3 movement = translation.z * rotatedFoward +
                         translation.x * cameraRight + translation.y * cameraUp;

   targetCameraState.x += movement.x;
   targetCameraState.y += movement.y;
   targetCameraState.z += movement.z;

   // targetCameraState.translate(translation);

   // Framerate-independent interpolation
   // calculate the lerp amount, such that we get 99% of the way to our target
   // in the specified time
   float positionLerpPct =
       1.f - expf(log(1.f - 0.99f) / positionLerpTime * timer->deltaTime());
   float rotationLerpPct =
       1.f - expf(log(1.f - 0.99f) / rotationLerpTime * timer->deltaTime());
   interpolatingCameraState.lerpTowards(targetCameraState, positionLerpPct,
                                        rotationLerpPct);

   if (transform != nullptr)
    interpolatingCameraState.updateTransform(*transform);
}
}  // namespace SIByL::Editor

namespace SIByL::Editor {
auto ComponentElucidator::onDrawGui(uint32_t flags, void* data) noexcept
    -> void {
   GameObjectInspector::GameObjectData* goData =
       static_cast<GameObjectInspector::GameObjectData*>(data);
   elucidateComponent(goData);
}

auto TagComponentFragment::elucidateComponent(
    GameObjectInspector::GameObjectData* data) noexcept -> void {
   GFX::GameObject* go = data->scene->getGameObject(data->handle);
   GFX::TagComponent* tagComponent =
       go->getEntity().getComponent<GFX::TagComponent>();
   if (tagComponent) {
    char buffer[256];
    memset(buffer, 0, sizeof(buffer));
    strcpy_s(buffer, tagComponent->name.c_str());
    if (ImGui::InputText(" ", buffer, sizeof(buffer)))
      tagComponent->name = std::string(buffer);
   }
}

auto TransformComponentFragment::elucidateComponent(
    GameObjectInspector::GameObjectData* data) noexcept -> void {
   GFX::GameObject* go = data->scene->getGameObject(data->handle);
   GFX::TransformComponent* transformComponent =
       go->getEntity().getComponent<GFX::TransformComponent>();
   if (transformComponent) {
    drawComponent<GFX::TransformComponent>(
        go, "Transform",
        [](GFX::TransformComponent* component) {
          // set translation
          Math::vec3 translation = component->translation;
          drawVec3Control("Translation", translation, 0, 100);
          bool translation_modified = (component->translation != translation);
          if (translation_modified) component->translation = translation;
          // set rotation
          Math::vec3 rotation = component->eulerAngles;
          drawVec3Control("Rotation", rotation, 0, 100);
          bool rotation_modified = (component->eulerAngles != rotation);
          if (rotation_modified) component->eulerAngles = rotation;
          // set scale
          Math::vec3 scaling = component->scale;
          drawVec3Control("Scaling", scaling, 1, 100);
          bool scale_modified = (component->scale != scaling);
          if (scale_modified) component->scale = scaling;
        },
        false);
   }
}

auto MeshReferenceComponentFragment::elucidateComponent(
    GameObjectInspector::GameObjectData* data) noexcept -> void {
   GFX::GameObject* go = data->scene->getGameObject(data->handle);
   GFX::MeshReference* meshReference =
       go->getEntity().getComponent<GFX::MeshReference>();
   if (meshReference) {
    drawComponent<GFX::MeshReference>(
        go, "MeshReference", [](GFX::MeshReference* component) {
          if (component->mesh) {
            const int index_size =
                component->mesh->primitiveState.stripIndexFormat ==
                        RHI::IndexFormat::UINT16_t
                    ? sizeof(uint16_t)
                    : sizeof(uint32_t);
            const int index_count =
                component->mesh->indexBufferInfo.size / index_size;
            const int primitive_count = index_count / 3;
            if (ImGui::TreeNode("Vertex Buffer")) {
              ImGui::BulletText(
                  (std::string("Size (bytes): ") +
                   std::to_string(component->mesh->vertexBufferInfo.size))
                      .c_str());
              if (ImGui::TreeNode("Buffer Layout")) {
                ImGui::BulletText(
                    (std::string("Array Stride: ") +
                     std::to_string(
                         component->mesh->vertexBufferLayout.arrayStride))
                        .c_str());
                ImGui::BulletText(
                    (std::string("Step Mode: ") +
                     to_string(component->mesh->vertexBufferLayout.stepMode))
                        .c_str());
                if (ImGui::TreeNode("Attributes Layouts:")) {
                  for (auto& item :
                       component->mesh->vertexBufferLayout.attributes) {
                    ImGui::BulletText(
                        (to_string(item.format) + std::string(" | OFF: ") +
                         std::to_string(item.offset) + std::string(" | LOC: ") +
                         std::to_string(item.shaderLocation))
                            .c_str());
                  }
                  ImGui::TreePop();
                }
                ImGui::TreePop();
              }
              ImGui::TreePop();
            }
            if (ImGui::TreeNode("Index Buffer")) {
              ImGui::BulletText(
                  (std::string("Size (bytes): ") +
                   std::to_string(component->mesh->indexBufferInfo.size))
                      .c_str());
              ImGui::BulletText(
                  (std::string("Index Count: ") + std::to_string(index_count))
                      .c_str());
              ImGui::TreePop();
            }
            if (ImGui::TreeNode("Primitive Status")) {
              ImGui::BulletText((std::string("Primitive Count: ") +
                                 std::to_string(primitive_count))
                                    .c_str());
              ImGui::BulletText(
                  (std::string("Topology: ") +
                   to_string(component->mesh->primitiveState.topology))
                      .c_str());

              ImGui::TreePop();
            }
          }
          ImGui::BulletText("Custom Primitive ID: ");
          ImGui::SameLine();
          ImGui::PushItemWidth(
              std::max(int(ImGui::GetContentRegionAvail().x), 5));
          int i = static_cast<int>(component->customPrimitiveFlag);
          ImGui::InputInt(" ", &i, 1, 10);
          ImGui::PopItemWidth();
          component->customPrimitiveFlag = static_cast<size_t>(i);
        });
   }
}

auto MeshRendererComponentFragment::elucidateComponent(
    GameObjectInspector::GameObjectData* data) noexcept -> void {
   GFX::GameObject* go = data->scene->getGameObject(data->handle);
   GFX::MeshRenderer* meshRenderer =
       go->getEntity().getComponent<GFX::MeshRenderer>();
   if (meshRenderer) {
    drawComponent<GFX::MeshRenderer>(
        go, "MeshRenderer", [](GFX::MeshRenderer* component) {
          uint32_t idx = 0;
          for (auto& material : component->materials) {
            if (ImGui::TreeNode(
                    ("Material ID: " + std::to_string(idx)).c_str())) {
              MaterialElucidator::onDrawGui_PTR(material);
              ImGui::TreePop();
            }
            ++idx;
          }
        });
   }
}

auto LightComponentFragment::elucidateComponent(
    GameObjectInspector::GameObjectData* data) noexcept -> void {
   GFX::GameObject* go = data->scene->getGameObject(data->handle);
   GFX::LightComponent* lightComp =
       go->getEntity().getComponent<GFX::LightComponent>();
   if (lightComp) {
    drawComponent<GFX::LightComponent>(
        go, "LightComponent", [](GFX::LightComponent* component) {
          // set type
          static std::string lightTypeNames;
          static bool inited = false;
          if (!inited) {
            for (uint32_t i = 0;
                 i < static_cast<uint32_t>(
                         GFX::LightComponent::LightType::MAX_ENUM);
                 ++i) {
              if (i != 0) lightTypeNames.push_back('\0');
              lightTypeNames += GFX::to_string(
                  static_cast<GFX::LightComponent::LightType>(i));
            }
            lightTypeNames.push_back('\0');
            lightTypeNames.push_back('\0');
            inited = true;
          }
          int item_current = static_cast<int>(component->type);
          ImGui::Combo("Light Type", &item_current, lightTypeNames.c_str());
          if (item_current != static_cast<int>(component->type))
            component->type = GFX::LightComponent::LightType(item_current);
          // set scale
          Math::vec3 intensity = component->intensity;
          drawVec3Control("Intensity", intensity, 0, 100);
          if (intensity != component->intensity)
            component->intensity = intensity;
        });
   }
}

auto CameraComponentFragment::elucidateComponent(
    GameObjectInspector::GameObjectData* data) noexcept -> void {
   GFX::GameObject* go = data->scene->getGameObject(data->handle);
   GFX::CameraComponent* cameraComp =
       go->getEntity().getComponent<GFX::CameraComponent>();
   if (cameraComp) {
    drawComponent<GFX::CameraComponent>(
        go, "CameraComponent", [](GFX::CameraComponent* component) {
          if (component->projectType ==
              GFX::CameraComponent::ProjectType::PERSPECTIVE) {
            ImGui::BulletText("Project Type: PERSPECTIVE");
            bool isDirty = false;
            float fovy = component->fovy;
            drawFloatControl("FoV", fovy, 45);
            if (fovy != component->fovy) {
              component->fovy = fovy;
              isDirty = true;
            }
            float near = component->near;
            drawFloatControl("Near", near, 0.001);
            if (near != component->near) {
              component->near = near;
              isDirty = true;
            }
            float far = component->far;
            drawFloatControl("Far", far, 1000);
            if (far != component->far) {
              component->far = far;
              isDirty = true;
            }
            bool isPrimary = component->isPrimaryCamera;
            drawBoolControl("Is Primary", isPrimary, 100);
            if (isPrimary != component->isPrimaryCamera) {
              component->isPrimaryCamera = isPrimary;
            }
          } else if (component->projectType ==
                     GFX::CameraComponent::ProjectType::ORTHOGONAL) {
            ImGui::BulletText("Project Type: ORTHOGONAL");
          }
        });
   }
}
}