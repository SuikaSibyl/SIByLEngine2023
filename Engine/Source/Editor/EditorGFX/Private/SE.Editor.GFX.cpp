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
#include <ImGuizmo.h>
#include <Config/SE.Core.Config.hpp>

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
    int width = (ImGui::GetContentRegionAvail().x - 50 + 5);
    ImGui::PushItemWidth(width);
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
    ImGui::PopItemWidth();
  }
  ImGui::Columns(1);
  ImGui::PopID();
}

auto drawCustomColume(const std::string& label, float columeWidth,
                      std::function<void()> const& func) noexcept -> void {
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
    int width = (ImGui::GetContentRegionAvail().x - 20 + 5);
    ImGui::PushItemWidth(width);
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2{0, 0});
    func();
    ImGui::PopStyleVar();
    ImGui::PopItemWidth();
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
    int width = (ImGui::GetContentRegionAvail().x - 120 + 19);
    ImGui::PushMultiItemsWidths(3, width);
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
    const ImGuiTreeNodeFlags treeNodeFlags =
      ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_Framed |
      ImGuiTreeNodeFlags_FramePadding | ImGuiTreeNodeFlags_SpanAvailWidth |
      ImGuiTreeNodeFlags_AllowItemOverlap;
    ImVec2 contentRegionAvailable = ImGui::GetContentRegionAvail();
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2{4, 4});
    float lineHeight =
        GImGui->Font->FontSize + GImGui->Style.FramePadding.y * 2.0f;
    ImGui::Separator();
    bool open = ImGui::TreeNodeEx(tex, treeNodeFlags, "Texture Resource");
    ImGui::PopStyleVar();

    if (open) {
        if (ImGui::Button("capture")) {
            captureImage(tex->guid);
        }
        float const texw = (float)tex->texture->width();
        float const texh = (float)tex->texture->height();
        float const wa =
            std::max(1.f, ImGui::GetContentRegionAvail().x - 15) / texw;
        float const ha = 1;
        float a = std::min(1.f, std::min(wa, ha));
        ImGui::Image(
            Editor::TextureUtils::getImGuiTexture(tex->guid)->getTextureID(),
            {a * texw, a * texh}, {0, 0}, {1, 1});
        drawCustomColume("Texture", 100, [&]() {
          char buffer[256];
          memset(buffer, 0, sizeof(buffer));
          strcpy_s(buffer, tex->texture->getName().c_str());
          if (ImGui::InputText(" ", buffer, sizeof(buffer)))
            tex->texture->setName(std::string(buffer));
        });
        drawCustomColume("GUID", 100,
                         [&]() { ImGui::Text(std::to_string(tex->guid).c_str()); });
        drawCustomColume("Size", 100, [&]() {
          ImGui::Text((std::string("- width: ") +
                       std::to_string(tex->texture->width()) +
                       std::string("  |  - height: ") +
                       std::to_string(tex->texture->height()))
                          .c_str());
        });
        ImGui::TreePop();
    }
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
  const ImGuiTreeNodeFlags treeNodeFlags =
      ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_Framed |
      ImGuiTreeNodeFlags_FramePadding | ImGuiTreeNodeFlags_SpanAvailWidth |
      ImGuiTreeNodeFlags_AllowItemOverlap;
  ImVec2 contentRegionAvailable = ImGui::GetContentRegionAvail();
  ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2{4, 4});
  float lineHeight =
      GImGui->Font->FontSize + GImGui->Style.FramePadding.y * 2.0f;
  ImGui::Separator();
  bool open = ImGui::TreeNodeEx(material, treeNodeFlags, "Material Resource");
  ImGui::PopStyleVar();

  if (ImGui::Button("Save")) {
    material->serialize();
  }

  if (open && material == nullptr) {

  }
  else if (open && material != nullptr) {
    drawCustomColume("Material", 150, [&]() {
      char buffer[256];
      memset(buffer, 0, sizeof(buffer));
      strcpy_s(buffer, material->name.c_str());
      if (ImGui::InputText(" ", buffer, sizeof(buffer)))
        material->name = std::string(buffer);
    });
    drawCustomColume("BSDF Type", 150, [&]() {
      const char* item_names[] = {
          "Lambertian",
          "Rough Plastic",
      };
      if (ImGui::Combo("##BSDFType", (int*)&material->BxDF, item_names,
                       IM_ARRAYSIZE(item_names), IM_ARRAYSIZE(item_names))) {
        material->isDirty = true;
      }
    });
        drawCustomColume("Path", 150,
                         [&]() { ImGui::Text(material->path.c_str()); });
        drawCustomColume("Emissive", 150, [&]() {
          ImGui::Checkbox("##Emissive", &material->isEmissive);
        });
        drawCustomColume("Alpha State", 150, [&]() {
          const char* item_names[] = {
              "Opaque",
              "Dither Discard",
              "Alpha Cut",
          };
          int alpha_state = int(material->alphaState);
          ImGui::Combo("##AlphaState", &alpha_state, item_names,
                       IM_ARRAYSIZE(item_names), IM_ARRAYSIZE(item_names));
          material->alphaState = GFX::Material::AlphaState(alpha_state);
        });
        drawCustomColume("Alpha Threshold", 150, [&]() {
          ImGui::DragFloat("##alphathresh", &material->alphaThreshold, 0.05f,
                           0.f, 1.f);
        });
        ImGuiColorEditFlags misc_flags =
            ImGuiColorEditFlags_HDR | ImGuiColorEditFlags_NoDragDrop;
        drawCustomColume("Diffuse Color", 150, [&]() {
          if (ImGui::ColorEdit3("##Basecolor",
                                (float*)&material->baseOrDiffuseColor,
                                ImGuiColorEditFlags_NoInputs |
                                    ImGuiColorEditFlags_NoLabel | misc_flags)) {
            material->isDirty = true;
          }
        });
        drawCustomColume("Specular Color", 150, [&]() {
          if (ImGui::ColorEdit3("##SpecularColor",
                                (float*)&material->specularColor,
                                ImGuiColorEditFlags_NoInputs |
                                    ImGuiColorEditFlags_NoLabel | misc_flags)) {
            material->isDirty = true;
          }
        });
        drawCustomColume("Emissive Color", 150, [&]() {
          if (ImGui::ColorEdit3("##EmissiveColor",
                                (float*)&material->emissiveColor,
                                ImGuiColorEditFlags_NoInputs |
                                    ImGuiColorEditFlags_NoLabel | misc_flags)) {
            material->isDirty = true;
          }
        });
        drawCustomColume("Eta", 150, [&]() {
          if (ImGui::DragFloat("##eta", &material->eta, 0.05f, 1.f, 2.f)) {
            material->isDirty = true;
          }
        });
        drawCustomColume("Roughness", 150, [&]() {
          if (ImGui::DragFloat("##roughness", &material->roughness, 0.05f, 0.f,
                               1.f)) {
            material->isDirty = true;
          }
        });

      if (ImGui::TreeNode("Textures:")) {
      uint32_t id = 0;
      for (auto& [name, texture] : material->textures) {
        ImGui::PushID(id);
        if (ImGui::TreeNode(
                (("Texture - " + std::to_string(id) + " - " + name).c_str()))) {
          if (ImGui::BeginDragDropTarget()) {
            if (const ImGuiPayload* payload =
                    ImGui::AcceptDragDropPayload("ASSET")) {
              const wchar_t* path = (const wchar_t*)payload->Data;
              std::filesystem::path texturePath = path;
              Core::GUID guid = GFX::GFXManager::get()->registerTextureResource(
                  texturePath.string().c_str());
              texture.guid = guid;
              material->serialize();
            }
            ImGui::EndDragDropTarget();
          }

          ImGui::SameLine();
          if (ImGui::Button("Remove")) {
            texture.guid = Core::INVALID_GUID;
            material->serialize();
          }

          if (texture.guid != Core::INVALID_GUID && texture.guid != 0) {
            TextureElucidator::onDrawGui_GUID(texture.guid);
          } else {
            ImGui::Text("No Resource Binded");
          }
          ImGui::TreePop();
        }
        ++id;
        ImGui::PopID();
      }
      ImGui::TreePop();
        }

        ImGui::TreePop();
  }
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
          "", Core::WorldTimePoint::get().to_string() + ".exr");
      Image::EXR::writeEXR(filepath, width, height, 4,
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

static const float identityMatrix[16] = {1.f, 0.f, 0.f, 0.f, 0.f, 1.f,
                                         0.f, 0.f, 0.f, 0.f, 1.f, 0.f,
                                         0.f, 0.f, 0.f, 1.f};

auto editTransform(float* cameraView, float* cameraProjection, float* matrix,
                   bool editTransformDecomposition,
                   ImGuizmoState& state) noexcept -> void {
   static ImGuizmo::MODE mCurrentGizmoMode(ImGuizmo::LOCAL);
   static bool useSnap = false;
   static float snap[3] = {1.f, 1.f, 1.f};
   static float bounds[] = {-0.5f, -0.5f, -0.5f, 0.5f, 0.5f, 0.5f};
   static float boundsSnap[] = {0.1f, 0.1f, 0.1f};
   static bool boundSizing = false;
   static bool boundSizingSnap = false;

   if (editTransformDecomposition) {
    if (ImGui::IsKeyPressed(ImGuiKey_T))
      state.mCurrentGizmoOperation = ImGuizmo::TRANSLATE;
    if (ImGui::IsKeyPressed(ImGuiKey_E))
      state.mCurrentGizmoOperation = ImGuizmo::ROTATE;
    if (ImGui::IsKeyPressed(ImGuiKey_R))  // r Key
      state.mCurrentGizmoOperation = ImGuizmo::SCALE;
    if (ImGui::RadioButton("Translate",
                           state.mCurrentGizmoOperation == ImGuizmo::TRANSLATE))
      state.mCurrentGizmoOperation = ImGuizmo::TRANSLATE;
    ImGui::SameLine();
    if (ImGui::RadioButton("Rotate",
                           state.mCurrentGizmoOperation == ImGuizmo::ROTATE))
      state.mCurrentGizmoOperation = ImGuizmo::ROTATE;
    ImGui::SameLine();
    if (ImGui::RadioButton("Scale",
                           state.mCurrentGizmoOperation == ImGuizmo::SCALE))
      state.mCurrentGizmoOperation = ImGuizmo::SCALE;
    if (ImGui::RadioButton("Universal",
                           state.mCurrentGizmoOperation == ImGuizmo::UNIVERSAL))
      state.mCurrentGizmoOperation = ImGuizmo::UNIVERSAL;
    float matrixTranslation[3], matrixRotation[3], matrixScale[3];
    ImGuizmo::DecomposeMatrixToComponents(matrix, matrixTranslation,
                                          matrixRotation, matrixScale);
    ImGui::InputFloat3("Tr", matrixTranslation);
    ImGui::InputFloat3("Rt", matrixRotation);
    ImGui::InputFloat3("Sc", matrixScale);
    ImGuizmo::RecomposeMatrixFromComponents(matrixTranslation, matrixRotation,
                                            matrixScale, matrix);

    if (state.mCurrentGizmoOperation != ImGuizmo::SCALE) {
      if (ImGui::RadioButton("Local", mCurrentGizmoMode == ImGuizmo::LOCAL))
        mCurrentGizmoMode = ImGuizmo::LOCAL;
      ImGui::SameLine();
      if (ImGui::RadioButton("World", mCurrentGizmoMode == ImGuizmo::WORLD))
        mCurrentGizmoMode = ImGuizmo::WORLD;
    }
    if (ImGui::IsKeyPressed(ImGuiKey_S)) useSnap = !useSnap;
    ImGui::Checkbox("##UseSnap", &useSnap);
    ImGui::SameLine();

    switch (state.mCurrentGizmoOperation) {
      case ImGuizmo::TRANSLATE:
        ImGui::InputFloat3("Snap", &snap[0]);
        break;
      case ImGuizmo::ROTATE:
        ImGui::InputFloat("Angle Snap", &snap[0]);
        break;
      case ImGuizmo::SCALE:
        ImGui::InputFloat("Scale Snap", &snap[0]);
        break;
    }
    ImGui::Checkbox("Bound Sizing", &boundSizing);
    if (boundSizing) {
      ImGui::PushID(3);
      ImGui::Checkbox("##BoundSizing", &boundSizingSnap);
      ImGui::SameLine();
      ImGui::InputFloat3("Snap", boundsSnap);
      ImGui::PopID();
    }
   }

   ImGuiIO& io = ImGui::GetIO();
   float viewManipulateRight = io.DisplaySize.x;
   float viewManipulateTop = 0;
   static ImGuiWindowFlags gizmoWindowFlags = 0;

   bool useWindow = true;
   ImGuizmo::SetRect(ImGui::GetWindowPos().x, ImGui::GetWindowPos().y,
                     state.width, state.height);
   ImGuizmo::SetDrawlist();
   viewManipulateRight = ImGui::GetWindowPos().x + state.width;
   viewManipulateTop = ImGui::GetWindowPos().y;
   ImGuiWindow* window = ImGui::GetCurrentWindow();
   gizmoWindowFlags = ImGui::IsWindowHovered() &&
                              ImGui::IsMouseHoveringRect(window->InnerRect.Min,
                                                         window->InnerRect.Max)
                          ? ImGuiWindowFlags_NoMove
                          : 0;

   ImGuizmo::DrawGrid(cameraView, cameraProjection, identityMatrix, 100.f);
   ImGuizmo::DrawCubes(cameraView, cameraProjection,
                       &state.objMatrix.data[0][0], 1);
   ImGuizmo::Manipulate(
       cameraView, cameraProjection, state.mCurrentGizmoOperation,
       mCurrentGizmoMode, matrix, NULL, useSnap ? &snap[0] : NULL,
       boundSizing ? bounds : NULL, boundSizingSnap ? boundsSnap : NULL);
}

auto ViewportWidget::setTarget(std::string const& name,
                               GFX::Texture* tex) noexcept -> void {
   this->name = name;
   texture = tex;
}

void Frustum(float left, float right, float bottom, float top, float znear,
             float zfar, float* m16) {
   float temp, temp2, temp3, temp4;
   temp = 2.0f * znear;
   temp2 = right - left;
   temp3 = top - bottom;
   temp4 = zfar - znear;
   m16[0] = temp / temp2;
   m16[1] = 0.0;
   m16[2] = 0.0;
   m16[3] = 0.0;
   m16[4] = 0.0;
   m16[5] = temp / temp3;
   m16[6] = 0.0;
   m16[7] = 0.0;
   m16[8] = (right + left) / temp2;
   m16[9] = (top + bottom) / temp3;
   m16[10] = (-zfar - znear) / temp4;
   m16[11] = -1.0f;
   m16[12] = 0.0;
   m16[13] = 0.0;
   m16[14] = (-temp * zfar) / temp4;
   m16[15] = 0.0;
}

void Perspective(float fovyInDegrees, float aspectRatio, float znear,
                 float zfar, float* m16) {
   float ymax, xmax;
   ymax = znear * tanf(fovyInDegrees * 3.141592f / 180.0f);
   xmax = ymax * aspectRatio;
   Frustum(-xmax, xmax, -ymax, ymax, znear, zfar, m16);
}

void Cross(const float* a, const float* b, float* r) {
   r[0] = a[1] * b[2] - a[2] * b[1];
   r[1] = a[2] * b[0] - a[0] * b[2];
   r[2] = a[0] * b[1] - a[1] * b[0];
}

float Dot(const float* a, const float* b) {
   return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

void Normalize(const float* a, float* r) {
   float il = 1.f / (sqrtf(Dot(a, a)) + FLT_EPSILON);
   r[0] = a[0] * il;
   r[1] = a[1] * il;
   r[2] = a[2] * il;
}

void LookAt(const float* eye, const float* at, const float* up, float* m16) {
   float X[3], Y[3], Z[3], tmp[3];

   tmp[0] = eye[0] - at[0];
   tmp[1] = eye[1] - at[1];
   tmp[2] = eye[2] - at[2];
   Normalize(tmp, Z);
   Normalize(up, Y);

   Cross(Y, Z, tmp);
   Normalize(tmp, X);

   Cross(Z, X, tmp);
   Normalize(tmp, Y);

   m16[0] = X[0];
   m16[1] = Y[0];
   m16[2] = Z[0];
   m16[3] = 0.0f;
   m16[4] = X[1];
   m16[5] = Y[1];
   m16[6] = Z[1];
   m16[7] = 0.0f;
   m16[8] = X[2];
   m16[9] = Y[2];
   m16[10] = Z[2];
   m16[11] = 0.0f;
   m16[12] = -Dot(X, eye);
   m16[13] = -Dot(Y, eye);
   m16[14] = -Dot(Z, eye);
   m16[15] = 1.0f;
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

   ImVec2 p = ImGui::GetCursorScreenPos();

   if (texture) {
    ImGui::Image(
        Editor::TextureUtils::getImGuiTexture(texture->guid)->getTextureID(),
        {(float)texture->texture->width(), (float)texture->texture->height()},
        {0, 0}, {1, 1});
   } else {
    ImGui::End();
    return;
   }

   float width = (float)texture->texture->width();
   float height = (float)texture->texture->height();

   Math::vec3 posW = camera_transform.translation;
   Math::vec3 target =
       camera_transform.translation + camera_transform.getRotatedForward();

   Math::mat4 view = Math::transpose(Math::lookAt(posW, target, Math::vec3(0, 1, 0)).m);
   Math::mat4 proj = Math::transpose(Math::perspective(camera->fovy, camera->aspect,
                                         camera->near, camera->far)
           .m);

   for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++) {
      float neg_view = (j == 0 || (i == 3 && j == 3)) ? 1 : -1;
      view.data[i][j] *= neg_view;
      float neg_proj = (i == 0 || i == 1) ? 1 : ((i == 2) ? -1 : 2);
      proj.data[i][j] *= neg_proj;
    }

    ImGuizmo::SetOrthographic(false);
    ImGuizmo::SetDrawlist();

    float windowWidth = (float)ImGui::GetWindowWidth();
   float windowHeight = (float)ImGui::GetWindowHeight();
    ImVec2 wp = ImGui::GetWindowPos();

   ImGuizmo::SetRect(wp.x + currPos.x, wp.y + currPos.y, width,
                     height);

   //Math::mat4 transform = tc.getAccumulativeTransform();
   Math::mat4 transform;

   ImGuiWindow* window = ImGui::GetCurrentWindow();
   //ImGuizmo::DrawGrid(&view.data[0][0], &proj.data[0][0], identityMatrix,
   //                   100.f);
   bool test = false;
   {
    if (ImGui::IsKeyPressed(ImGuiKey_R))
      gizmoState.mCurrentGizmoOperation = ImGuizmo::TRANSLATE;
    if (ImGui::IsKeyPressed(ImGuiKey_T))
      gizmoState.mCurrentGizmoOperation = ImGuizmo::ROTATE;
    if (ImGui::IsKeyPressed(ImGuiKey_Y))  // r Key
      gizmoState.mCurrentGizmoOperation = ImGuizmo::SCALE;
    if (ImGui::IsKeyPressed(ImGuiKey_U))
      gizmoState.mCurrentGizmoOperation = ImGuizmo::UNIVERSAL;
   }
    Math::mat4 objectTransform;
    Math::mat4 objectTransformPrecursor;
   if (selectedGO.has_value() && selectedScene != nullptr) {
    {//first compute the transform
       //  get transform
      float oddScaling = 1.f;
      Math::vec3 scaling = Math::vec3{1, 1, 1};
      {  // get mesh transform matrix
        GFX::GameObject* go = selectedScene->getGameObject(selectedGO.value());
        GFX::TransformComponent* transform =
            go->getEntity().getComponent<GFX::TransformComponent>();
        objectTransform = transform->getTransform() * objectTransform;
        oddScaling *=
            transform->scale.x * transform->scale.y * transform->scale.z;
        scaling *= transform->scale;
        while (go->parent != Core::NULL_ENTITY) {
          test = true;
          go = selectedScene->getGameObject(go->parent);
          GFX::TransformComponent* transform =
              go->getEntity().getComponent<GFX::TransformComponent>();
          objectTransform = transform->getTransform() * objectTransform;
          objectTransformPrecursor =
              transform->getTransform() * objectTransformPrecursor;
          oddScaling *=
              transform->scale.x * transform->scale.y * transform->scale.z;
          scaling *= transform->scale;
        }
      }
    }
    objectTransform = Math::transpose(objectTransform);
    ImGuizmo::Manipulate(
        &view.data[0][0], &proj.data[0][0], gizmoState.mCurrentGizmoOperation,
        gizmoState.mCurrentGizmoMode, &objectTransform.data[0][0], NULL,
        gizmoState.useSnap ? &gizmoState.snap[0] : NULL, NULL, NULL);

    if (test) {
      float a = 1.f;
    }
    float matrixTranslation[3], matrixRotation[3], matrixScale[3];

    Math::mat4 inv = Math::transpose(Math::inverse(objectTransformPrecursor));
    Math::mat4 thisMatrix = objectTransform * inv;
    ImGuizmo::DecomposeMatrixToComponents(
        &thisMatrix.data[0][0], matrixTranslation,
                                          matrixRotation, matrixScale);
    //ImGuizmo::RecomposeMatrixFromComponents(
    //    matrixTranslation, matrixRotation, matrixScale, &thisMatrix.data[0][0]);

    GFX::GameObject* go = selectedScene->getGameObject(selectedGO.value());
    GFX::TransformComponent* transform =
        go->getEntity().getComponent<GFX::TransformComponent>();
    transform->translation = Math::vec3{
        matrixTranslation[0], matrixTranslation[1], matrixTranslation[2]};
    transform->eulerAngles =
        Math::vec3{matrixRotation[0], matrixRotation[1], matrixRotation[2]};
    transform->scale =
        Math::vec3{matrixScale[0], matrixScale[1], matrixScale[2]};
   }
   float viewManipulateRight = wp.x + currPos.x + width;
   float viewManipulateTop = wp.y + currPos.y;
   Math::mat4 view_origin = view;
   ImGuizmo::ManipulateResult result = ImGuizmo::ViewManipulate_Custom(
       &view.data[0][0], 5,
       ImVec2(viewManipulateRight - 128, viewManipulateTop), ImVec2(128, 128),
       0x10101010);

   if (selectedGO.has_value() && selectedScene != nullptr && result.edited) {
    view = Math::transpose(view);
    for (int i = 0; i < 4; i++)
      for (int j = 0; j < 4; j++) {
        float neg_view = (j == 0 || (i == 3 && j == 3)) ? 1 : -1;
        view.data[i][j] *= neg_view;
      }
    Math::vec4 target =
        Math::transpose(objectTransform) * Math::vec4(0, 0, 0, 1);
    
    //Math::mat4::rotateZ(eulerAngles.z) * Math::mat4::rotateY(eulerAngles.y) *
    //    Math::mat4::rotateX(eulerAngles.x)
    Math::vec3 forward =
        Math::vec3{result.newDir[0], result.newDir[1], result.newDir[2]};
    Math::vec3 cameraPos =
        Math::vec3{target.x, target.y, target.z} + forward * 5;
    camera_transform_ref->translation =
        Math::vec3{cameraPos.data[0], cameraPos.data[1], cameraPos.data[2]};

    float pitch = std::atan2(forward.z, sqrt(forward.x * forward.x + forward.y * forward.y));
    float yaw = std::atan2(forward.x, forward.y);
    pitch *= 180. / IM_PI;
    yaw *= 180. / IM_PI;
    yaw = 180. - yaw;
    camera_transform_ref->eulerAngles = {pitch, yaw, 0};

    *forceReset = true;
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

   if (modal_state.showModal)
       ImGui::OpenPopup("Warning");
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
    ImGui::EndPopup();
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

               if (ImGui::BeginDragDropSource()) {
              	ImGui::Text("Dragged");

              	const wchar_t* itemPath = directoryEntry.path().c_str();
              	ImGui::SetDragDropPayload("ASSET", itemPath,
                  (wcslen(itemPath) + 1) * sizeof(wchar_t));
                    ImGui::EndDragDropSource();
                  }
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
   std::string engine_path =
       Core::RuntimeConfig::get()->string_property("engine_path");
   icons.back = GFX::GFXManager::get()->registerTextureResource(
       (engine_path + "../Engine/Binaries/Runtime/icons/back.png").c_str());
   icons.folder = GFX::GFXManager::get()->registerTextureResource(
       (engine_path + "../Engine/Binaries/Runtime/icons/folder.png").c_str());
   icons.file = GFX::GFXManager::get()->registerTextureResource(
       (engine_path + "../Engine/Binaries/Runtime/icons/file.png").c_str());
   icons.image = GFX::GFXManager::get()->registerTextureResource(
       (engine_path + "../Engine/Binaries/Runtime/icons/image.png").c_str());
   icons.material = GFX::GFXManager::get()->registerTextureResource(
       (engine_path + "../Engine/Binaries/Runtime/icons/material.png").c_str());
   icons.mesh = GFX::GFXManager::get()->registerTextureResource(
       (engine_path + "../Engine/Binaries/Runtime/icons/mesh.png").c_str());
   icons.scene = GFX::GFXManager::get()->registerTextureResource(
       (engine_path + "../Engine/Binaries/Runtime/icons/scene.png").c_str());
   icons.shader = GFX::GFXManager::get()->registerTextureResource(
       (engine_path + "../Engine/Binaries/Runtime/icons/shader.png").c_str());
   icons.video = GFX::GFXManager::get()->registerTextureResource(
       (engine_path + "../Engine/Binaries/Runtime/icons/video.png").c_str());
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
            if (pair.second.get()->initComponent(entity)) {

            }
            else {
              pair.second.get()->removeComponent(entity);
            }
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

   auto save_scene = [scene = scene]() {
     std::string name = scene->name + ".scene";
     std::string path = ImGuiLayer::get()
                            ->rhiLayer->getRHILayerDescriptor()
                            .windowBinded->saveFile(nullptr, name);
     if (path != "") {
       scene->serialize(path);
       scene->isDirty = false;
     }
   };

   if (ImGui::BeginMenuBar()) {
    if (ImGui::Button("New")) {
    }
    if (ImGui::Button("Save")) {
      if (scene != nullptr) {
        save_scene();
      }
    }
    if (ImGui::Button("Load")) {
      if (scene != nullptr) {
        std::string path = ImGuiLayer::get()
                               ->rhiLayer->getRHILayerDescriptor()
                               .windowBinded->openFile("scene");
        if (path != "") scene->deserialize(path);
         scene->isDirty = true;
      }
    }
    if (ImGui::BeginMenu("Import")) {
      // Menu - File - Load
      if (ImGui::MenuItem("glTF 2.0 (.glb/.gltf)")) {
        std::string path = ImGuiLayer::get()
                               ->rhiLayer->getRHILayerDescriptor()
                               .windowBinded->openFile(".gltf");
        GFX::SceneNodeLoader_glTF::loadSceneNode(
            path, *scene, GFX::GFXManager::get()->config.meshLoaderConfig);
        save_scene();
      }
      if (ImGui::MenuItem("Wavefront (.obj)")) {
        std::string path = ImGuiLayer::get()
                               ->rhiLayer->getRHILayerDescriptor()
                               .windowBinded->openFile(".obj");
        GFX::SceneNodeLoader_obj::loadSceneNode(
            path, *scene, GFX::GFXManager::get()->config.meshLoaderConfig);
        save_scene();
      }
      if (ImGui::MenuItem("FBX (.fbx)")) {
        std::string path = ImGuiLayer::get()
                               ->rhiLayer->getRHILayerDescriptor()
                               .windowBinded->openFile(".fbx");
        GFX::SceneNodeLoader_assimp::loadSceneNode(
            path, *scene, GFX::GFXManager::get()->config.meshLoaderConfig);
        save_scene();
      }
      if (ImGui::MenuItem("Mitsuba (.xml)")) {
        std::string path = ImGuiLayer::get()
                               ->rhiLayer->getRHILayerDescriptor()
                               .windowBinded->openFile(".xml");
        GFX::SceneNodeLoader_mitsuba::loadSceneNode(
            path, *scene, GFX::GFXManager::get()->config.meshLoaderConfig);
        save_scene();
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
    if (viewportWidget) {
      viewportWidget->selectedScene = nullptr;
      viewportWidget->selectedGO = std::nullopt;
    }
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
    if (viewportWidget) {
      viewportWidget->selectedScene = scene;
      viewportWidget->selectedGO = node;
    }
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
      if (viewportWidget) {
        viewportWidget->selectedScene = nullptr;
        viewportWidget->selectedGO = std::nullopt;
      }
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
   viewport->camera_transform = *transform;
   viewport->camera_transform_ref = transform;
   viewport->forceReset = &forceReset;
   if (this->transform != transform) {
    targetCameraState.setFromTransform(*transform);
    interpolatingCameraState.setFromTransform(*transform);
    this->transform = transform;
   }
}

auto SimpleCameraController::onUpdate() noexcept -> void {
   if (forceReset) {
    forceReset = false;
    targetCameraState.setFromTransform(*transform);
    interpolatingCameraState.setFromTransform(*transform);
    return;
   }
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

auto SimpleCameraController2D::onEnable(
    GFX::TransformComponent const& transform) noexcept -> void {
   targetCameraState.setFromTransform(transform);
   interpolatingCameraState.setFromTransform(transform);
}

auto SimpleCameraController2D::getInputTranslationDirection() noexcept
    -> Math::vec3 {
   Math::vec3 direction(0.0f, 0.0f, 0.0f);
   if (input->isKeyPressed(Platform::SIByL_KEY_W)) {
    scaling *= 1.05;  // forward
   }
   if (input->isKeyPressed(Platform::SIByL_KEY_S)) {
    scaling /= 1.05;  // back
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

auto SimpleCameraController2D::bindTransform(
    GFX::TransformComponent* transform) noexcept -> void {
   viewport->camera_transform = *transform;
   viewport->camera_transform_ref = transform;
   viewport->forceReset = &forceReset;
   if (this->transform != transform) {
    targetCameraState.setFromTransform(*transform);
    interpolatingCameraState.setFromTransform(*transform);
    this->transform = transform;
   }
}

auto SimpleCameraController2D::onUpdate() noexcept -> void {
   if (forceReset) {
    forceReset = false;
    targetCameraState.setFromTransform(*transform);
    interpolatingCameraState.setFromTransform(*transform);
    return;
   }
   // check the viewport is hovered
   bool hovered = viewport->info.isHovered;
   // rotation
   targetCameraState.yaw = 0;
   targetCameraState.pitch = 0;
   interpolatingCameraState.yaw = 0;
   interpolatingCameraState.pitch = 0;
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
            if (material == nullptr) {
              if (ImGui::TreeNode(
                      ("Material ID: " + std::to_string(idx)).c_str())) {
                if (ImGui::BeginDragDropTarget()) {
                  if (const ImGuiPayload* payload =
                          ImGui::AcceptDragDropPayload("ASSET")) {
                    const wchar_t* path = (const wchar_t*)payload->Data;
                    std::filesystem::path materialPath = path;
                    Core::GUID guid =
                        GFX::GFXManager::get()->registerMaterialResource(
                            materialPath.string().c_str());
                    component->materials[idx] =
                        Core::ResourceManager::get()
                            ->getResource<GFX::Material>(guid);
                  }
                  ImGui::EndDragDropTarget();
                }
                ImGui::TreePop();
              }
            } else {
              if (ImGui::TreeNode(
                      ("Material ID: " + std::to_string(idx)).c_str())) {
                MaterialElucidator::onDrawGui_PTR(material);
                ImGui::TreePop();
              }
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
          drawCustomColume("Type", 100, [&]() {
            int item_current = static_cast<int>(component->type);
            ImGui::Combo("##wwwww", &item_current, lightTypeNames.c_str());
            if (item_current != static_cast<int>(component->type)) {
              component->type = GFX::LightComponent::LightType(item_current);
              component->isDirty = true;
            }
          });
          // set scale
          Math::vec3 intensity = component->intensity;
          drawVec3Control("Intensity", intensity, 0, 100);
          if (intensity != component->intensity)
            component->intensity = intensity;
          switch (component->type) {
            case GFX::LightComponent::LightType::DIRECTIONAL:
            case GFX::LightComponent::LightType::POINT:
              break;
            case GFX::LightComponent::LightType::SPOT: {
                drawCustomColume("Cosine Total Width", 200, [&]() {
                if(ImGui::DragFloat("##Cosine Total Width",
                                     &component->packed_data_0.x, 0.01, 0, 1)) {
                  component->isDirty = true;
              }
              });
              drawCustomColume("Cosine Falloff Start", 200, [&]() {
                  if (ImGui::DragFloat(
                      "##Cosine Falloff Start",
                        &component->packed_data_0.y, 0.01, 0,
                                       1)) {
                    component->isDirty = true;
                  }
              });
            }
              break;
            default:
              break;
          }
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
            drawCustomColume("Project", 100, [&]() {
              int item_current = static_cast<int>(component->projectType);
              ImGui::Combo("##proj", &item_current, "Perspective\0Orthogonal\0");
              if (item_current != static_cast<int>(component->projectType))
                component->projectType =
                    GFX::CameraComponent::ProjectType(item_current);
            });
            bool isDirty = false;
            drawCustomColume("FoV", 100, [&]() {
              float fovy = component->fovy;
              ImGui::DragFloat("##FoV", &fovy, 1);
              if (fovy != component->fovy) {
                component->fovy = fovy;
                isDirty = true;
              }
            });
            drawCustomColume("Near", 100, [&]() {
              float near = component->near;
              ImGui::DragFloat("##near", &near, 0.001f);
              if (near != component->near) {
                component->near = near;
                isDirty = true;
              }
            });
            drawCustomColume("Far", 100, [&]() {
              float far = component->far;
              ImGui::DragFloat("##far", &far, 10.f);
              if (far != component->far) {
                component->far = far;
                isDirty = true;
              }
            });
            drawCustomColume("Is Primary", 100, [&]() {
              bool isPrimary = component->isPrimaryCamera;
              ImGui::Checkbox("##primary", &component->isPrimaryCamera);
            });
        });
   }
}
}