#include <se.core.hpp>
#include <se.math.hpp>
#include <se.rhi.hpp>
#include <se.gfx.hpp>
#include <se.rdg.hpp>
#include <iostream>
#include <se.editor.hpp>
#include <imgui.h>
#define SIByL_API __declspec(dllimport)
#include <seditor.hpp>
#include <seditor-base.hpp>
#include <passes/se.pass.editor.hpp>
#include <passes/se.pass.rasterizer.hpp>
#include <bxdfs/se.bxdf.rglbrdf.hpp>
#include "source/geometry_inspector.hpp"

int main() {
  // initialize vulkan context
  std::unique_ptr<se::window> window = se::window::create(se::window::WindowOptions{
    se::window::Vendor::GLFW, L"SIByL 2024",
    1280, 720, se::window::Properties::VULKAN_CONTEX});
 // std::unique_ptr<se::rhi::Context> context = 
	//se::rhi::Context::create(se::rhi::Context::Backend::Vulkan);
 // context->init(window.get(), se::rhi::ContextExtensions(
 //   se::rhi::ContextExtensionBit::RAY_TRACING |
 //   se::rhi::ContextExtensionBit::BINDLESS_INDEXING |
 //   se::rhi::ContextExtensionBit::FRAGMENT_BARYCENTRIC |
 //   se::rhi::ContextExtensionBit::CONSERVATIVE_RASTERIZATION |
 //   se::rhi::ContextExtensionBit::COOPERATIVE_MATRIX |
 //   se::rhi::ContextExtensionBit::CUDA_INTEROPERABILITY |
 //   se::rhi::ContextExtensionBit::ATOMIC_FLOAT));
 // std::unique_ptr<se::rhi::Adapter> adapter = context->requestAdapter({});

 // std::unique_ptr<se::rhi::Device> device = adapter->requestDevice();
 // se::rhi::CUDAContext::initialize(device.get());
 // se::gfx::GFXContext::initialize(device.get());
 // se::gfx::GFXContext::createFlights(MULTIFRAME_FLIGHTS_COUNT, nullptr);
 // se::editor::ImGuiContext::initialize(device.get());
 // se::editor::EditorContext::initialize();
 // ImGui::SetCurrentContext(se::editor::ImGuiContext::getRawCtx());

 //se::gfx::SceneHandle scene = se::gfx::GFXContext::load_scene_gltf("C:\\Users\\suika\\Downloads\\KhronosGroup glTF-Sample-Models main 2.0-DamagedHelmet\\glTF\\DamagedHelmet.gltf");
 se::gfx::SceneHandle scene = se::gfx::GFXContext::load_scene_gltf("S:\\SIByL2024\\Sandbox\\examples\\nee\\_data\\scene.gltf");
 // //se::gfx::SceneHandle scene = se::gfx::GFXContext::load_scene_gltf("C:\\Users\\suika\\Downloads\\skull\\skull_downloadable\\scene.gltf");
 // //se::gfx::SceneHandle scene = se::gfx::GFXContext::load_scene_gltf("S:/SIByL2024/Sandbox/examples/rasterdiff/_data/sphere.gltf");
 // //se::gfx::SceneHandle scene = se::gfx::GFXContext::load_scene_gltf("D:/Art/Scenes/3d_material_ball/scene.gltf");
 // se::gfx::SceneHandle scene = se::gfx::GFXContext::load_scene_gltf("S:/SIByL2024/Sandbox/neumat/scene.gltf");
 // //se::gfx::SceneHandle scene = se::gfx::GFXContext::load_scene_gltf("S:/SIByL2024/Sandbox/neumat/scene.gltf");
 // //se::gfx::SceneHandle scene = se::gfx::GFXContext::create_scene("Default Scene");
 // //se::gfx::SceneHandle scene = se::gfx::GFXContext::load_scene_gltf("test2.gltf");
 // // initialize the window and gui context
 // //std::unique_ptr<se::rhi::SwapChain> swapChain = device->createSwapChain({});
 // se::rhi::MultiFrameFlights* multiFrameFlights = se::gfx::GFXContext::getFlights();
 scene->serialize("S:\\SIByL2024\\Sandbox\\examples\\nee\\_data\\scene-1.gltf");
 // se::timer timer;

 // //std::unique_ptr<se::EPFLBrdf> test = std::make_unique<se::EPFLBrdf>("C:/Users/suika/Downloads/cc_ibiza_sunset_rgb.bsdf");
 // //test = nullptr;

 // se::editor::EditorBase::bindScene(scene);
 // se::editor::EditorBase::bindInput(window->getInput());
 // se::editor::EditorBase::bindTimer(&timer);

 // scene->updateTransform();
 // scene->createTexcoord(se::gfx::Scene::TexcoordKind::CopyCoord0);
 // scene->updateGPUScene();
 // //scene->serialize("test2.gltf");

 // //auto pass = std::make_unique<se::GeometryInspectorPass>();
 // //auto pass = std::make_unique<se::NeumatFwdPass>();
 // //pass->scene = scene;
 // std::unique_ptr<se::GeometryInspectorPipeline> pipeline = std::make_unique<se::GeometryInspectorPipeline>();
 // //std::unique_ptr<se::CBTTestPipeline> pipeline = std::make_unique<se::CBTTestPipeline>();
 // pipeline->setStandardSize({ 1024,1024,1 });
 // pipeline->build();

 // se::editor::EditorBase::bindPipeline(pipeline.get());

 // bool should_exit = false;
 // // run the main loop
 // while (!should_exit) {
 //   //  fetch main window events
 //   window->fetchEvents();
 //   window->endFrame();
 //   
 //   se::editor::ImGuiContext::startNewFrame();

 //   multiFrameFlights->frameStart();

 //   std::unique_ptr<se::rhi::CommandEncoder> commandEncoder = device->createCommandEncoder({ multiFrameFlights->getCommandBuffer() });

 //   pipeline->bindScene(scene);
 //   pipeline->execute(commandEncoder.get());

 //   device->getGraphicsQueue()->submit(
 //     { commandEncoder->finish() },
 //     multiFrameFlights->getImageAvailableSeamaphore(),
 //     multiFrameFlights->getRenderFinishedSeamaphore(),
 //     multiFrameFlights->getFence());

 //   //std::unique_ptr<se::rhi::CommandEncoder> commandEncoder = device->createCommandEncoder({ nullptr });
 //   //se::editor::ImGuiContext::imguiBackend->render(multiFrameFlights->getRenderFinishedSeamaphore());
 //   se::editor::ImGuiContext::startGuiRecording();
 //   se::editor::EditorBase::onImGuiDraw();
 //   se::editor::EditorBase::onUpdate();


 //   //{
 //   //  se::Line3DPass* line3dpss =
 //   //    static_cast<se::Line3DPass*>(
 //   //    pipeline->getActiveGraphs()[0]->getPass("Line3D Pass"));
 //   //  line3dpss->clear();
 //   //  line3dpss->addAABB(se::bounds3{ se::vec3{0}, se::vec3{1} }, se::vec3{ 1,0,1 }, 1);
 //   //}

 //   timer.update();
 //   ImGui::Begin("Hello");
 //   ImGui::End();

 //   bool show_demo_window = true;
 //   ImGui::ShowDemoWindow(&show_demo_window);

 //   se::editor::ImGuiContext::render(multiFrameFlights->getRenderFinishedSeamaphore());
 //   //device->getGraphicsQueue()->submit({ commandEncoder->finish() });
 //   //device->waitIdle();

 //   multiFrameFlights->frameEnd();
 //   // Update window status, to check whether should exit
 //   should_exit |= !window->isRunning();

 //   device->waitIdle();
 //   scene->updateGPUScene();
 //   device->waitIdle();
 // }


 // //pass->save_output();

 // device->waitIdle();
 // pipeline = nullptr;
 // se::editor::EditorBase::finalize();
 // se::editor::ImGuiContext::finalize();
 // se::gfx::GFXContext::finalize();
  window->destroy();

  return 0;
}