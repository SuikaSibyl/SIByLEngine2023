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

#include <SE.RHI.Profiler.hpp>
#include <Resource/SE.Core.Resource.hpp>
#include <SE.Editor.Core.hpp>
#include <SE.Editor.Config.hpp>
#include <SE.Editor.DebugDraw.hpp>
#include <SE.GFX-Utils.h>
#include <SE.Application.hpp>
#include <SE.SRenderer.hpp>
#include "CustomPipeline.hpp"

#include <SE.Addon.VPL.hpp>
#include <SE.Addon.SLC.hpp>
#include <SE.Addon.SST.hpp>
#include <Plugins/SE.SRendererExt.GeomTab.hpp>
#include <Pipeline/SE.SRendere-RTGIPipeline.hpp>
#include <Pipeline/SE.SRendere-ForwardPipeline.hpp>
#include <Pipeline/SE.SRendere-SSRXPipeline.hpp>
#include <Pipeline/SE.SRendere-UDPTPipeline.hpp>
#include <Pipeline/SE.SRendere-GeoInspectPipeline.hpp>
#include <Passes/FullScreenPasses/ScreenSpace/SE.SRenderer-BarGTPass.hpp>

using namespace SIByL::Math;
namespace Test{
Math::vec2 ToConcentricMap(Math::vec2 onSquare) {
    float phi;
    float r;
    // (a,b) is now on [-1,1]^2
    const float a = 2 * onSquare.x - 1;
    const float b = 2 * onSquare.y - 1;
    if (a > -b) {   // region 1 or 2
      if (a > b) {  // region 1, also |a| > |b|
        r = a;
        phi = (Math::float_Pi / 4) * (b / a);
      } else {  // region 2, also |b| > |a|
        r = b;
        phi = (Math::float_Pi / 4) * (2 - (a / b));
      }
    } else {        // region 3 or 4
      if (a < b) {  // region 3, also |a| > |b|, a!= 0
        r = -a;
        phi = (Math::float_Pi / 4) * (4 + (b / a));
      } else {  // region 4, |b| >= |a|, but a==0 and b==0 could occur.
        r = -b;
        if (b != 0)
          phi = (Math::float_Pi / 4) * (6 - (a / b));
        else
          phi = 0;
      }
    }
    float u = r * cos(phi);
    float v = r * sin(phi);
    return Math::vec2(u, v);
  }

  Math::vec2 FromConcentricMap(Math::vec2 onDisk) {
    const float r = sqrt(onDisk.x * onDisk.x + onDisk.y * onDisk.y);
    float phi = atan2(onDisk.y, onDisk.x);
    if (phi < -Math::float_Pi / 4)
      phi += 2 * Math::float_Pi;  // in range [-pi/4,7pi/4]
    float a;
    float b;
    if (phi < Math::float_Pi / 4) {  // region 1
      a = r;
      b = phi * a / (Math::float_Pi / 4);
    } else if (phi < 3 * Math::float_Pi / 4) {  // region 2
      b = r;
      a = -(phi - Math::float_Pi / 2) * b / (Math::float_Pi / 4);
    } else if (phi < 5 * Math::float_Pi / 4) {  // region 3
      a = -r;
      b = (phi - Math::float_Pi) * a / (Math::float_Pi / 4);
    } else {  // region 4
      b = -r;
      a = -(phi - 3 * Math::float_Pi / 2) * b / (Math::float_Pi / 4);
    }
    const float x = (a + 1) / 2;
    const float y = (b + 1) / 2;
    return Math::vec2(x, y);
  }

}

struct SandBoxApplication :public Application::ApplicationBase {
	
	Video::VideoDecoder decoder;

	/** Initialize the application */
	virtual auto Init() noexcept -> void override {
      Math::vec2 test = {0.25, 0.75};
      Math::vec2 res = Test::FromConcentricMap(Test::ToConcentricMap(test));

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

		Singleton<RHI::DeviceProfilerManager>::instance()->initialize(
                    device, "device_something");

        GFX::GFXManager::get()->config.meshLoaderConfig =
                    SRenderer::meshLoadConfig;

		scene.deserialize("P:/GitProjects/SIByLEngine2022/Sandbox/content/test_scene.scene");
		//scene.deserialize("C:/Users/suika/Desktop/testscene/bedroom_mask.scene");

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

		pipeline1 = std::make_unique<ADPipeline>();
		//pipeline1 = std::make_unique<CustomPipeline>();
		//pipeline2 = std::make_unique<VXPGReSTIRPipeline>();
		//pipeline1 = std::make_unique<Addon::SLC::SLCTestPipeline>();
		pipeline2 = std::make_unique<SSPGReSTIRPipeline>();
  //      rtgi_pipeline = std::make_unique<RestirGIPipeline>();
        rtgi_pipeline = std::make_unique<RestirGIPipeline>();

		//geoinsp_pipeline = std::make_unique<SSPGP_GMM_Pipeline>();
        geoinsp_pipeline = std::make_unique<GTPipeline>();
        //geoinsp_pipeline = std::make_unique<SRP::GeoInspectPipeline>();
		//vxgi_pipeline = std::make_unique<SSPGP_GMM_Pipeline>();
        vxgi_pipeline = std::make_unique<SSPGPipeline>();
        vxdi_pipeline = std::make_unique<VXPGPipeline>();
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

	void AnimateScene() {
        ImGui::Begin("Animate");
        static float speed = 1.f;
        ImGui::DragFloat("speed", &speed, 0.01);
        ImGui::End();

		auto view = Core::ComponentManager::get()->view<GFX::TagComponent, GFX::TransformComponent>();
        for (auto& [entity, tag, transform] : view) {
			if (tag.name == "mask.fbx") {
				transform.eulerAngles.y = timer.totalTime() * speed;
			}
			if (tag.name == "sphere.fbx") {
				transform.translation.y = 2 + 1.5 * std::cos(timer.totalTime() * speed);
			}
			//if (tag.name == "DirectionalLight") {
			//	transform.eulerAngles.z = 25 * std::cos(timer.totalTime() * speed * 0.6);
			//}
        }
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
		device->waitIdle();

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

        static int frames2capture = 0;
        imguiLayer->startGuiRecording();
        bool show_demo_window = true;
        ImGui::ShowDemoWindow(&show_demo_window);
		{
			ImGui::Begin("Device-Profiler");
				RHI::DeviceProfilerManager* profiler = Singleton<RHI::DeviceProfilerManager>::instance();
				ImGui::Checkbox("Enable Profiler", &profiler->enabled);
				bool clearProfiler = false;
				ImGui::Checkbox("Clear Profiler", &clearProfiler);
				if (clearProfiler) {
					profiler->clear();
				}
				if (profiler->enabled) {
					profiler->flushResults();
					profiler->reset(commandEncoder.get());
				}
				for (auto& pair : profiler->statistics) {
					double time = double(pair.second.accumulation) / (pair.second.count * 1000000);
                    if (ImGui::TreeNode((pair.first + ": " + std::to_string(time)).c_str())) {
						ImGui::TreePop();
					}
				}
				ImGui::End();
			}

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
								frames2capture = 50;
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
                                frames2capture = 50;
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
                                frames2capture = 50;
                                pipeline = vxgi_pipeline.get();
                                editorLayer
                                    ->getWidget<Editor::RDGViewerWidget>()
                                    ->pipeline = pipeline;
                        } else if (pipeline_id == 5) {
                                frames2capture = 50;
                                pipeline = vxdi_pipeline.get();
                                editorLayer
                                    ->getWidget<Editor::RDGViewerWidget>()
                                    ->pipeline = pipeline;
                        }
                }
                ImGui::End();

				AnimateScene();


		srenderer->invalidScene(scene);
		std::vector<RDG::Graph*> graphs = pipeline->getActiveGraphs();

		for(auto* graph: graphs) srenderer->updateRDGData(graph);

		for(auto* graph: graphs)
			for (size_t i : graph->getFlattenedPasses()) {
				graph->getPass(i)->onInteraction(mainWindow.get()->getInput(), &(editorLayer->getWidget<Editor::ViewportWidget>()->info));
			}

		Singleton<RHI::DeviceProfilerManager>::instance()->beginSegment(
                    commandEncoder.get(), RHI::PipelineStages::TOP_OF_PIPE_BIT,
                    "total_pipe");
		pipeline->execute(commandEncoder.get());

        Singleton<RHI::DeviceProfilerManager>::instance()->endSegment(
                    commandEncoder.get(), RHI::PipelineStages::BOTTOM_OF_PIPE_BIT,
                    "total_pipe");

		//Editor::DebugDraw::DrawAABB(srenderer->statisticsData.aabb, 5., 5.);
		//auto editor_graphs = Editor::DebugDraw::get()->pipeline->getActiveGraphs();
		//for (auto* graph : editor_graphs)
		//	srenderer->updateRDGData(graph);
		//Editor::DebugDraw::Draw(commandEncoder.get());

		device->getGraphicsQueue()->submit({ commandEncoder->finish({}) }, 
			multiFrameFlights->getImageAvailableSeamaphore(),
			multiFrameFlights->getRenderFinishedSeamaphore(),
			multiFrameFlights->getFence());

		editorLayer->getWidget<Editor::ViewportWidget>()->setTarget("Main Viewport", pipeline->getOutput());
		editorLayer->onDrawGui();
		imguiLayer->render(multiFrameFlights->getRenderFinishedSeamaphore());

		multiFrameFlights->frameEnd();

		//if (frames2capture > 0) {
		//	static int i = 0;
		//	if (i < 102) {
  //                      
		//	}
		//	device->waitIdle();
  //          GFX::CaptureImage(pipeline->getOutput(),
  //              "D:/data/adaptation/sspg_gmm/" + std::to_string(i++));
		//	frames2capture--;
		//}
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
        Singleton<RHI::DeviceProfilerManager>::instance()->finalize();
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