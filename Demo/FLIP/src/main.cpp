#include <crtdbg.h>
#include <glad/glad.h>
#include <imgui.h>
#include <tinygltf/tiny_gltf.h>

#include <Passes/FullScreenPasses/ScreenSpace/SE.SRenderer-BarGTPass.hpp>
#include <Pipeline/SE.SRendere-ForwardPipeline.hpp>
#include <Pipeline/SE.SRendere-RTGIPipeline.hpp>
#include <Pipeline/SE.SRendere-SSRXPipeline.hpp>
#include <Plugins/SE.SRendererExt.GeomTab.hpp>
#include <Resource/SE.Core.Resource.hpp>
#include <SE.Application.hpp>
#include <SE.Editor.Config.hpp>
#include <SE.Editor.Core.hpp>
#include <SE.Editor.DebugDraw.hpp>
#include <SE.SRenderer.hpp>
#include <array>
#include <chrono>
#include <filesystem>
#include <format>
#include <functional>
#include <iostream>
#include <memory>
#include <stack>
#include <typeinfo>

#include "FLIPDemo.hpp"

struct FLIPApplication : public Application::ApplicationBase {
  /** Initialize the application */
  FLIP::FLIPDemo script;

  virtual auto Init() noexcept -> void override {
    // create optional layers: rhi, imgui, editor
    rhiLayer = std::make_unique<RHI::RHILayer>(RHI::RHILayerDescriptor{
        RHI::RHIBackend::Vulkan,
        RHI::ContextExtensionsFlags(RHI::ContextExtension::BINDLESS_INDEXING |
                                    RHI::ContextExtension::ATOMIC_FLOAT),
        mainWindow.get(), true});
    GFX::GFXManager::get()->rhiLayer = rhiLayer.get();
    imguiLayer = std::make_unique<Editor::ImGuiLayer>(rhiLayer.get());
    editorLayer = std::make_unique<Editor::EditorLayer>();
    Editor::Config::buildEditorLayer(editorLayer.get());

    // bind editor layer
    editorLayer->getWidget<Editor::SceneWidget>()->bindScene(&scene);
    editorLayer->getWidget<Editor::StatusWidget>()->timer = &timer;

    RHI::Device* device = rhiLayer->getDevice();
    RHI::SwapChain* swapchain = rhiLayer->getSwapChain();

    GFX::GFXManager::get()->config.meshLoaderConfig = SRenderer::meshLoadConfig;
    scene.deserialize("asset/demo.scene");
    GFX::GFXManager::get()->registerDefualtSamplers();

    script.onStart();
    script.input = mainWindow.get()->getInput();

    InvalidScene();
    
    editorLayer->getWidget<Editor::RDGViewerWidget>()->pipeline = script.pipeline.get();

    cameraController.init(mainWindow.get()->getInput(), &timer,
                          editorLayer->getWidget<Editor::ViewportWidget>());
  };

  void InvalidScene() {
    srenderer = std::make_unique<SRenderer>();
    srenderer->init(scene);
    GeometryTabulator::tabulate(512,
                                srenderer->sceneDataPack.position_buffer_cpu,
                                srenderer->sceneDataPack.index_buffer_cpu,
                                srenderer->sceneDataPack.geometry_buffer_cpu);
  }

  /** Update the application every loop */
  virtual auto Update(double deltaTime) noexcept -> void override {
    RHI::Device* device = rhiLayer->getDevice();
    if (scene.isDirty == true) {
      device->waitIdle();
      InvalidScene();
      scene.isDirty = false;
    }

    int width, height;
    mainWindow->getFramebufferSize(&width, &height);
    width = 1280;
    height = 720;
    {
      auto view = Core::ComponentManager::get()->view<GFX::CameraComponent>();
      for (auto& [entity, camera] : view) {
        if (camera.isPrimaryCamera) {
          camera.aspect = 1.f * width / height;
          camera.bottom_top = 1.55f;
          cameraController.scaling = camera.bottom_top;
          camera.projectType = GFX::CameraComponent::ProjectType::ORTHOGONAL;
          editorLayer->getWidget<Editor::ViewportWidget>()->camera = &camera;
          cameraController.bindTransform(
              scene.getGameObject(entity)
                  ->getEntity()
                  .getComponent<GFX::TransformComponent>());
          srenderer->updateCamera(
              *(scene.getGameObject(entity)
                    ->getEntity()
                    .getComponent<GFX::TransformComponent>()),
              camera);
          break;
        }
      }
      cameraController.onUpdate();
      // Core::LogManager::Log(std::to_string(timer.deltaTime()));
    }
    // start new frame
    imguiLayer->startNewFrame();

    // frame start
    RHI::SwapChain* swapChain = rhiLayer->getSwapChain();
    RHI::MultiFrameFlights* multiFrameFlights =
        rhiLayer->getMultiFrameFlights();
    multiFrameFlights->frameStart();

    std::unique_ptr<RHI::CommandEncoder> commandEncoder =
        device->createCommandEncoder({multiFrameFlights->getCommandBuffer()});
    GFX::GFXManager::get()->onUpdate();
    device->waitIdle();
    srenderer->invalidScene(scene);
    std::vector<RDG::Graph*> graphs = script.pipeline->getActiveGraphs();

    script.onUpdate(commandEncoder.get(), multiFrameFlights->getFlightIndex());

    for (auto* graph : graphs) srenderer->updateRDGData(graph);

    for (auto* graph : graphs)
      for (size_t i : graph->getFlattenedPasses()) {
        graph->getPass(i)->onInteraction(
            mainWindow.get()->getInput(),
            &(editorLayer->getWidget<Editor::ViewportWidget>()->info));
      }

    script.pipeline->execute(commandEncoder.get());

    device->getGraphicsQueue()->submit(
        {commandEncoder->finish({})},
        multiFrameFlights->getImageAvailableSeamaphore(),
        multiFrameFlights->getRenderFinishedSeamaphore(),
        multiFrameFlights->getFence());

    //device->waitIdle();

    // GUI Recording
    imguiLayer->startGuiRecording();
    bool show_demo_window = true;
    ImGui::ShowDemoWindow(&show_demo_window);

    editorLayer->getWidget<Editor::ViewportWidget>()->setTarget(
        "Main Viewport", script.pipeline->getOutput());
    editorLayer->onDrawGui();
    imguiLayer->render(multiFrameFlights->getRenderFinishedSeamaphore());

    multiFrameFlights->frameEnd();
  };

  /** Update the application every fixed update timestep */
  virtual auto FixedUpdate() noexcept -> void override{

  };

  virtual auto Exit() noexcept -> void override {
    rhiLayer->getDevice()->waitIdle();
    script.Exit();
    srenderer = nullptr;

    editorLayer = nullptr;
    imguiLayer = nullptr;

    Core::ResourceManager::get()->clear();

    rhiLayer = nullptr;
  }

 private:
  Core::GUID rtTarget;

  uint32_t indexCount = 0;

  std::unique_ptr<RHI::RHILayer> rhiLayer = nullptr;
  std::unique_ptr<Editor::ImGuiLayer> imguiLayer = nullptr;
  std::unique_ptr<Editor::EditorLayer> editorLayer = nullptr;
  std::unique_ptr<SRenderer> srenderer = nullptr;

  // the embedded scene, which should be removed in the future
  GFX::Scene scene;

  Editor::SimpleCameraController2D cameraController;
  GFX::TransformComponent cameraTransform;
};

int main() {
  // application root, control all managers
  Application::Root root;
  // run app
  FLIPApplication app;
  app.createMainWindow({Platform::WindowVendor::GLFW, L"SIByL Elastic Demo",
                        1920, 1080, Platform::WindowProperties::VULKAN_CONTEX});
  app.run();
}