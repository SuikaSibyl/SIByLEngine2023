#include <iostream>
#include <chrono>
#include <format>
#include <array>
#include <functional>
#include <filesystem>
#include <chrono>
#include <memory>
#include <typeinfo>
#include <glad/glad.h>
#include <imgui.h>
#include <stack>
#include <tinygltf/tiny_gltf.h>
#include <crtdbg.h>

#include <Resource/SE.Core.Resource.hpp>
#include <SE.Editor.Core.hpp>
#include <SE.Editor.Config.hpp>
#include <SE.Editor.DebugDraw.hpp>
#include <SE.Application.hpp>
#include <SE.SRenderer.hpp>
#include "CustomPipeline.hpp"

#include <SE.Addon.VPL.hpp>
#include <SE.Addon.SLC.hpp>
#include <Plugins/SE.SRendererExt.GeomTab.hpp>
#include <Pipeline/SE.SRendere-RTGIPipeline.hpp>
#include <Pipeline/SE.SRendere-ForwardPipeline.hpp>
#include <Pipeline/SE.SRendere-SSRXPipeline.hpp>
#include <Pipeline/SE.SRendere-UDPTPipeline.hpp>
#include <Pipeline/SE.SRendere-GeoInspectPipeline.hpp>
#include <Passes/FullScreenPasses/ScreenSpace/SE.SRenderer-BarGTPass.hpp>

struct SandBoxApplication :public Application::ApplicationBase {
	
	Video::VideoDecoder decoder;

	/** Initialize the application */
	virtual auto Init() noexcept -> void override {
		// create optional layers: rhi, imgui, editor
		rhiLayer = std::make_unique<RHI::RHILayer>(RHI::RHILayerDescriptor{
				RHI::RHIBackend::Vulkan,
				RHI::ContextExtensionsFlags(
					RHI::ContextExtension::RAY_TRACING
					| RHI::ContextExtension::BINDLESS_INDEXING
					| RHI::ContextExtension::FRAGMENT_BARYCENTRIC
					| RHI::ContextExtension::CONSERVATIVE_RASTERIZATION
					| RHI::ContextExtension::ATOMIC_FLOAT),
				mainWindow.get(),
				true
			});
		GFX::GFXManager::get()->rhiLayer = rhiLayer.get();
		imguiLayer = std::make_unique<Editor::ImGuiLayer>(rhiLayer.get());
		editorLayer = std::make_unique<Editor::EditorLayer>();
		Editor::Config::buildEditorLayer(editorLayer.get());

		// bind editor layer
		editorLayer->getWidget<Editor::SceneWidget>()->bindScene(&scene);
		editorLayer->getWidget<Editor::StatusWidget>()->timer = &timer;

		RHI::Device* device = rhiLayer->getDevice();
		RHI::SwapChain* swapchain = rhiLayer->getSwapChain();

        GFX::GFXManager::get()->config.meshLoaderConfig =
                    SRenderer::meshLoadConfig;

		scene.deserialize("P:/GitProjects/SIByLEngine2022/Sandbox/content/test_scene.scene");

		GFX::GFXManager::get()->commonSampler.defaultSampler = Core::ResourceManager::get()->requestRuntimeGUID<GFX::Sampler>();
		GFX::GFXManager::get()->registerSamplerResource(GFX::GFXManager::get()->commonSampler.defaultSampler, RHI::SamplerDescriptor{
				RHI::AddressMode::REPEAT,
				RHI::AddressMode::REPEAT,
				RHI::AddressMode::REPEAT,
			});
		GFX::GFXManager::get()->commonSampler.clamp_nearest = Core::ResourceManager::get()->requestRuntimeGUID<GFX::Sampler>();
		GFX::GFXManager::get()->registerSamplerResource(GFX::GFXManager::get()->commonSampler.clamp_nearest, RHI::SamplerDescriptor{});
		Core::ResourceManager::get()->getResource<GFX::Sampler>(GFX::GFXManager::get()->commonSampler.defaultSampler)->sampler->setName("DefaultSampler");
		Core::ResourceManager::get()->getResource<GFX::Sampler>(GFX::GFXManager::get()->commonSampler.clamp_nearest)->sampler->setName("ClampNearestSampler");
        
		InvalidScene();
		device->waitIdle();

		pipeline1 = std::make_unique<CustomPipeline>();
		pipeline2 = std::make_unique<Addon::SLC::SLCTestPipeline>();
		rtgi_pipeline = std::make_unique<SemiNEEPipeline>();
        geoinsp_pipeline = std::make_unique<SRP::GeoInspectPipeline>();
		vxgi_pipeline = std::make_unique<SSPGPipeline>();
        vxdi_pipeline = std::make_unique<VXGuidingPipeline>();
		pipeline1->build();
		pipeline2->build();
		rtgi_pipeline->build();
        geoinsp_pipeline->build();
		vxgi_pipeline->build();
        vxdi_pipeline->build();

		pipeline = geoinsp_pipeline.get();

		Editor::DebugDraw::Init(pipeline->getOutput(), nullptr);

		editorLayer->getWidget<Editor::RDGViewerWidget>()->pipeline = pipeline;

		//std::unique_ptr<Image::Texture_Host> dds_tex = Image::DDS::fromDDS("D:/Art/Scenes/Bistro_v5_2/Bistro_v5_2/Textures/Bollards_BaseColor.dds");
		//Core::GUID guid_dds = Core::ResourceManager::get()->requestRuntimeGUID<GFX::Texture>();
		//GFX::GFXManager::get()->registerTextureResource(guid_dds, dds_tex.get());

		cameraController.init(mainWindow.get()->getInput(), &timer, editorLayer->getWidget<Editor::ViewportWidget>());
	};

	void InvalidScene() {
        srenderer = std::make_unique<SRenderer>();
        srenderer->init(scene);
        GeometryTabulator::tabulate(
            512, srenderer->sceneDataPack.position_buffer_cpu,
            srenderer->sceneDataPack.index_buffer_cpu,
            srenderer->sceneDataPack.geometry_buffer_cpu);
	}

	/** Update the application every loop */
    virtual auto Update(double deltaTime) noexcept -> void override {
        RHI::Device* device = rhiLayer->getDevice();
		if (scene.isDirty == true) {
          device->waitIdle();
          InvalidScene();
          device->waitIdle();
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
                                  editorLayer
                                      ->getWidget<Editor::ViewportWidget>()
                                      ->camera = &camera;
					cameraController.bindTransform(scene.getGameObject(entity)->getEntity().getComponent<GFX::TransformComponent>());
                                  srenderer->updateCamera(
                                      *(scene.getGameObject(entity)
                                            ->getEntity()
                                            .getComponent<
                                                GFX::TransformComponent>()),
                                      camera, {width, height});
					break;
				}
			}
			cameraController.onUpdate();
			//Core::LogManager::Log(std::to_string(timer.deltaTime()));
		}
		// start new frame
		imguiLayer->startNewFrame();
		Editor::DebugDraw::Clear();

		// frame start
		RHI::SwapChain* swapChain = rhiLayer->getSwapChain();
		RHI::MultiFrameFlights* multiFrameFlights = rhiLayer->getMultiFrameFlights();
		multiFrameFlights->frameStart();

		std::unique_ptr<RHI::CommandEncoder> commandEncoder = device->createCommandEncoder({ multiFrameFlights->getCommandBuffer() });
		GFX::GFXManager::get()->onUpdate();
		//decoder.readFrame();

                // GUI Recording
                imguiLayer->startGuiRecording();
                bool show_demo_window = true;
                ImGui::ShowDemoWindow(&show_demo_window);

                ImGui::Begin("Pipeline Choose");
                {  // Select an item type
                        const char* item_names[] = {
                            "RayTracing", "Forward", "RTGI Pipeline",  "Geo Inspector",
                            "VXGI Inspector", "VXDI Inspector"};
                        static int pipeline_id = 3;
                        ImGui::Combo("Mode", &pipeline_id, item_names,
                                     IM_ARRAYSIZE(item_names),
                                     IM_ARRAYSIZE(item_names));
                        if (pipeline_id == 0) {
                                pipeline = pipeline1.get();
                                editorLayer
                                    ->getWidget<Editor::RDGViewerWidget>()
                                    ->pipeline = pipeline;
                        } else if (pipeline_id == 1) {
                                pipeline = pipeline2.get();
                                editorLayer
                                    ->getWidget<Editor::RDGViewerWidget>()
                                    ->pipeline = pipeline;
                        } else if (pipeline_id == 2) {
                                pipeline = rtgi_pipeline.get();
                                editorLayer
                                    ->getWidget<Editor::RDGViewerWidget>()
                                    ->pipeline = pipeline;
                        } else if (pipeline_id == 3) {
                                pipeline = geoinsp_pipeline.get();
                                editorLayer
                                    ->getWidget<Editor::RDGViewerWidget>()
                                    ->pipeline = pipeline;
                        } else if (pipeline_id == 4) {
                                pipeline = vxgi_pipeline.get();
                                editorLayer
                                    ->getWidget<Editor::RDGViewerWidget>()
                                    ->pipeline = pipeline;
                        } else if (pipeline_id == 5) {
                                pipeline = vxdi_pipeline.get();
                                editorLayer
                                    ->getWidget<Editor::RDGViewerWidget>()
                                    ->pipeline = pipeline;
                        }
                }
                ImGui::End();


		srenderer->invalidScene(scene);
		std::vector<RDG::Graph*> graphs = pipeline->getActiveGraphs();

		for(auto* graph: graphs) srenderer->updateRDGData(graph);

		for(auto* graph: graphs)
			for (size_t i : graph->getFlattenedPasses()) {
				graph->getPass(i)->onInteraction(mainWindow.get()->getInput(), &(editorLayer->getWidget<Editor::ViewportWidget>()->info));
			}

		pipeline->execute(commandEncoder.get());

		Editor::DebugDraw::DrawAABB(srenderer->statisticsData.aabb, 5., 5.);
		auto editor_graphs = Editor::DebugDraw::get()->pipeline->getActiveGraphs();
		for (auto* graph : editor_graphs)
			srenderer->updateRDGData(graph);
		Editor::DebugDraw::Draw(commandEncoder.get());

		device->getGraphicsQueue()->submit({ commandEncoder->finish({}) }, 
			multiFrameFlights->getImageAvailableSeamaphore(),
			multiFrameFlights->getRenderFinishedSeamaphore(),
			multiFrameFlights->getFence());

		editorLayer->getWidget<Editor::ViewportWidget>()->setTarget("Main Viewport", pipeline->getOutput());
		editorLayer->onDrawGui();
		imguiLayer->render(multiFrameFlights->getRenderFinishedSeamaphore());


		multiFrameFlights->frameEnd();
	};

	/** Update the application every fixed update timestep */
	virtual auto FixedUpdate() noexcept -> void override {

	};

	virtual auto Exit() noexcept -> void override {
		decoder.close();

		rhiLayer->getDevice()->waitIdle();
		srenderer = nullptr;
		pipeline1 = nullptr;
        pipeline2 = nullptr;
        rtgi_pipeline = nullptr;
        geoinsp_pipeline = nullptr;
        vxdi_pipeline = nullptr;
        vxgi_pipeline = nullptr;

		Editor::DebugDraw::Destroy();

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
	
	RDG::Pipeline* pipeline = nullptr;
	std::unique_ptr<RDG::Pipeline> pipeline1 = nullptr;
	std::unique_ptr<RDG::Pipeline> pipeline2 = nullptr;
    std::unique_ptr<RDG::Pipeline> rtgi_pipeline = nullptr;
    std::unique_ptr<RDG::Pipeline> geoinsp_pipeline = nullptr;
    std::unique_ptr<RDG::Pipeline> vxgi_pipeline = nullptr;
    std::unique_ptr<RDG::Pipeline> vxdi_pipeline = nullptr;
    
	// the embedded scene, which should be removed in the future
	GFX::Scene scene;

	Editor::SimpleCameraController cameraController;
	GFX::TransformComponent cameraTransform;
};

int main()
{
	// application root, control all managers
	Application::Root root;

	// run app
	SandBoxApplication app;
	app.createMainWindow({
			Platform::WindowVendor::GLFW,
			L"SIByL Engine 2023.1",
			1920, 1080,
			Platform::WindowProperties::VULKAN_CONTEX
		});
	app.run();
}