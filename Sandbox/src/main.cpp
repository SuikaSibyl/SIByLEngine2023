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

import SE.Utility;
import SE.Core.Log;
import SE.Core.Memory;
import SE.Core.IO;
import SE.Core.Event;
import SE.Core.Misc;
import SE.Core.ECS;
import SE.Core.Resource;
import SE.Core.UnitTest;

import SE.Math.Misc;
import SE.Math.Geometric;

import SE.Platform.Window;
import SE.Platform.Misc;
import SE.Platform.Socket;
import SE.Image;
import SE.RHI;
import SE.Parallelism;

import SE.GFX;
import SE.RDG;
import SE.Video;

import SE.Application;

import SE.Editor.Core;
import SE.Editor.GFX;
import SE.Editor.RDG;
import SE.Editor.Config;
import SE.Editor.DebugDraw;

import SE.SRenderer;
import SE.SRenderer.ForwardPipeline;
import SE.SRenderer.MMLTPipeline;
import SE.SRenderer.UDPTPipeline;
import SE.SRenderer.BDPTPipeline;

using namespace SIByL;
using namespace SIByL::Core;
using namespace SIByL::Math;


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

		//GFX::GFXManager::get()->registerTextureResource("P:/GitProjects/SIByLEngine2022/Sandbox/content/viking_room.png");
		//Core::ResourceManager::get()->getResource<GFX::Texture>(mat.textures["normal_bump"])->serialize();

		//GFX::GFXManager::get()->registerTextureResourceCubemap(
		//	Core::ResourceManager::get()->requestRuntimeGUID<GFX::Texture>(),
		//	std::array<char const*, 6>{
		//		"P:/GitProjects/SIByLEngine2022/Sandbox/content/textures/skybox/right.jpg",
		//		"P:/GitProjects/SIByLEngine2022/Sandbox/content/textures/skybox/left.jpg",
		//		"P:/GitProjects/SIByLEngine2022/Sandbox/content/textures/skybox/top.jpg",
		//		"P:/GitProjects/SIByLEngine2022/Sandbox/content/textures/skybox/bottom.jpg",
		//		"P:/GitProjects/SIByLEngine2022/Sandbox/content/textures/skybox/front.jpg",
		//		"P:/GitProjects/SIByLEngine2022/Sandbox/content/textures/skybox/back.jpg"
		//	}
		//);
		
		//mat.path = "./content/cerberus.mat";
		//mat.textures["normal_bump"] = GFX::GFXManager::get()->requestOfflineTextureResource(1658119028953168732);
		//Core::GUID matID = GFX::GFXManager::get()->registerMaterialResource("P:/GitProjects/SIByLEngine2022/Sandbox/content/default.mat");
		//GFX::Material* mat = Core::ResourceManager::get()->getResource<GFX::Material>(matID);
		//mat.textures["base_color"] = GFX::GFXManager::get()->requestOfflineTextureResource(1658119031102545654);
		//mat.serialize();
		//mat.textures["base_color"] = GFX::GFXManager::get()->registerTextureResource("P:/GitProjects/SIByLEngine2022/Sandbox/content/textures/Cerberus_B.png");
		//Core::ResourceManager::get()->getResource<GFX::Texture>(mat.textures["base_color"])->serialize();

		GFX::GFXManager::get()->config.meshLoaderConfig = SRenderer::meshLoadConfig;
		scene.deserialize("P:/GitProjects/SIByLEngine2022/Sandbox/content/test_scene.scene");
		//GFX::SceneNodeLoader_obj::loadSceneNode("P:/GitProjects/SIByLEngine2022/Sandbox/content/cerberus.obj", scene, SRenderer::meshLoadConfig);
		//GFX::SceneNodeLoader_obj::loadSceneNode("P:/GitProjects/SIByLEngine2022/Sandbox/content/scenes/wuson.obj", scene, SRenderer::meshLoadConfig);
		//for (auto handle : scene.gameObjects) {
		//	GFX::GameObject* go = scene.getGameObject(handle.first);
		//	Math::mat4 objectMat;
		//	GFX::MeshReference* meshref = go->getEntity().getComponent<GFX::MeshReference>();
		//	if (meshref) {
		//		GFX::MeshRenderer* meshrenderer = go->getEntity().addComponent<GFX::MeshRenderer>();
		//		meshrenderer->materials.push_back(mat);
		//	}
		//}
		//GFX::SceneNodeLoader_glTF::loadSceneNode("D:/Downloads/glTF-Sample-Models-master/glTF-Sample-Models-master/2.0/Sponza/glTF/Sponza.gltf", scene);
		//GFX::SceneNodeLoader_glTF::loadSceneNode("P:/GitProjects/SIByLEngine2022/Sandbox/content/scenes/cornellBox.gltf", scene);
		//scene.deserialize("P:/GitProjects/SIByLEngine2022/Sandbox/content/cornellBox.scene");
		//scene.deserialize("P:/GitProjects/SIByLEngine2022/Sandbox/content/cornellBoxSphere.scene");
		//scene.deserialize("P:/GitProjects/SIByLEngine2022/Sandbox/content/sponza.scene");

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

		srenderer = std::make_unique<SRenderer>();
		srenderer->init(scene);
		pipeline = std::make_unique<SRP::UDPTPipeline>();
		pipeline->build();

		Editor::DebugDraw::Init(pipeline->getOutput());

		editorLayer->getWidget<Editor::RDGViewerWidget>()->pipeline = pipeline.get();

		cameraController.init(mainWindow.get()->getInput(), &timer);
	};

	/** Update the application every loop */
	virtual auto Update(double deltaTime) noexcept -> void override {
		{
			cameraController.onUpdate();
			//Core::LogManager::Log(std::to_string(timer.deltaTime()));
		}
		// start new frame
		imguiLayer->startNewFrame();
		Editor::DebugDraw::Clear();

		// frame start
		RHI::Device* device = rhiLayer->getDevice();
		RHI::SwapChain* swapChain = rhiLayer->getSwapChain();
		RHI::MultiFrameFlights* multiFrameFlights = rhiLayer->getMultiFrameFlights();
		multiFrameFlights->frameStart();

		std::unique_ptr<RHI::CommandEncoder> commandEncoder = device->createCommandEncoder({ multiFrameFlights->getCommandBuffer() });
		GFX::GFXManager::get()->onUpdate();
		//decoder.readFrame();

		int width, height;
		mainWindow->getFramebufferSize(&width, &height);
		width = 1280;
		height = 720;

		auto view = Core::ComponentManager::get()->view<GFX::CameraComponent>();
		for (auto& [entity, camera] : view) {
			if (camera.isPrimaryCamera) {
				camera.aspect = 1.f * width / height;
				cameraController.bindTransform(scene.getGameObject(entity)->getEntity().getComponent<GFX::TransformComponent>());
				srenderer->updateCamera(*(scene.getGameObject(entity)->getEntity().getComponent<GFX::TransformComponent>()), camera);
				break;
			}
		}
		srenderer->invalidScene(scene);
		srenderer->updateRDGData(pipeline->getActiveGraph());

		pipeline->execute(commandEncoder.get());

		Editor::DebugDraw::DrawLine2D(Math::vec2{ 0.,0. }, Math::vec2{ 1280,720 }, 5., 5.);

		Editor::DebugDraw::Draw(commandEncoder.get());

		device->getGraphicsQueue()->submit({ commandEncoder->finish({}) }, 
			multiFrameFlights->getImageAvailableSeamaphore(),
			multiFrameFlights->getRenderFinishedSeamaphore(),
			multiFrameFlights->getFence());

		// GUI Recording
		imguiLayer->startGuiRecording();
		bool show_demo_window = true;
		ImGui::ShowDemoWindow(&show_demo_window);

		editorLayer->getWidget<Editor::ViewportWidget>()->setTarget("Main Viewport", pipeline->getOutput());
		editorLayer->onDrawGui();
		imguiLayer->render();

		multiFrameFlights->frameEnd();
	};

	/** Update the application every fixed update timestep */
	virtual auto FixedUpdate() noexcept -> void override {

	};

	virtual auto Exit() noexcept -> void override {
		decoder.close();

		rhiLayer->getDevice()->waitIdle();
		srenderer = nullptr;
		pipeline = nullptr;

		Editor::DebugDraw::Destroy();

		editorLayer = nullptr;
		imguiLayer = nullptr;

		ResourceManager::get()->clear();

		rhiLayer = nullptr;
	}

private:
	Core::GUID rtTarget;

	uint32_t indexCount = 0;

	std::unique_ptr<RHI::RHILayer> rhiLayer = nullptr;
	std::unique_ptr<Editor::ImGuiLayer> imguiLayer = nullptr;
	std::unique_ptr<Editor::EditorLayer> editorLayer = nullptr;
	std::unique_ptr<SRenderer> srenderer = nullptr;
	
	std::unique_ptr<RDG::Pipeline> pipeline = nullptr;

	// the embedded scene, which should be removed in the future
	GFX::Scene scene;

	Editor::SimpleCameraController cameraController;
	GFX::TransformComponent cameraTransform;
};

int main()
{
	// application root, control all managers
	Application::Root root;
	Core::UnitTestManager::run();

	// run app
	SandBoxApplication app;
	app.createMainWindow({
			Platform::WindowVendor::GLFW,
			L"SIByL Engine 2023.0",
			1280, 720,
			Platform::WindowProperties::VULKAN_CONTEX
		});
	app.run();
}