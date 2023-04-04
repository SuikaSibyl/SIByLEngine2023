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

import SE.Application;

import SE.Editor.Core;
import SE.Editor.GFX;
import SE.Editor.Config;

import SE.SRenderer;
import SE.SRenderer.ForwardPipeline;
import SE.SRenderer.MMLTPipeline;
import SE.SRenderer.UDPTPipeline;

using namespace SIByL;
using namespace SIByL::Core;
using namespace SIByL::Math;


struct SandBoxApplication :public Application::ApplicationBase {

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
		graph = std::make_unique<SRP::UDPTPipeline>();
		graph->build();

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
		// frame start
		RHI::Device* device = rhiLayer->getDevice();
		RHI::SwapChain* swapChain = rhiLayer->getSwapChain();
		RHI::MultiFrameFlights* multiFrameFlights = rhiLayer->getMultiFrameFlights();
		multiFrameFlights->frameStart();

		std::unique_ptr<RHI::CommandEncoder> commandEncoder = device->createCommandEncoder({ multiFrameFlights->getCommandBuffer() });

		RHI::RayTracingPassDescriptor rayTracingDescriptor = {};

		uint32_t index = multiFrameFlights->getFlightIndex();

		int width, height;
		mainWindow->getFramebufferSize(&width, &height);
		width = 800;
		height = 600;

		auto view = Core::ComponentManager::get()->view<GFX::CameraComponent>();
		for (auto& [entity, camera] : view) {
			if (camera.isPrimaryCamera) {
				camera.aspect = 1.f * 800 / 600;
				cameraController.bindTransform(scene.getGameObject(entity)->getEntity().getComponent<GFX::TransformComponent>());
				srenderer->updateCamera(*(scene.getGameObject(entity)->getEntity().getComponent<GFX::TransformComponent>()), camera);
				break;
			}
		}
		srenderer->invalidScene(scene);
		srenderer->updateRDGData(graph.get());

		graph->execute(commandEncoder.get());

		device->getGraphicsQueue()->submit({ commandEncoder->finish({}) }, 
			multiFrameFlights->getImageAvailableSeamaphore(),
			multiFrameFlights->getRenderFinishedSeamaphore(),
			multiFrameFlights->getFence());

		// GUI Recording
		imguiLayer->startGuiRecording();
		bool show_demo_window = true;
		ImGui::ShowDemoWindow(&show_demo_window);

		editorLayer->getWidget<Editor::ViewportWidget>()->setTarget("Main Viewport", graph->getOutput());
		editorLayer->onDrawGui();
		imguiLayer->render();

		multiFrameFlights->frameEnd();
	};

	auto captureImage(Core::GUID src) noexcept -> void {
		//RHI::Texture* cpySrc = Core::ResourceManager::get()->getResource<GFX::Texture>(src)->texture.get();
		//rhiLayer->getDevice()->readbackDeviceLocalTexture(cpySrc, );
		static Core::GUID copyDst = 0;
		if (copyDst == 0) {
			copyDst = Core::ResourceManager::get()->requestRuntimeGUID<GFX::Texture>();
			RHI::TextureDescriptor desc{
				{800,600,1},
				1, 1, RHI::TextureDimension::TEX2D,
				RHI::TextureFormat::RGBA32_FLOAT,
				(uint32_t)RHI::TextureUsage::COPY_DST,
				{ RHI::TextureFormat::RGBA32_FLOAT },
				RHI::TextureFlags::HOSTI_VISIBLE
			};
			GFX::GFXManager::get()->registerTextureResource(copyDst, desc);
		}
		rhiLayer->getDevice()->waitIdle();
		std::unique_ptr<RHI::CommandEncoder> commandEncoder = rhiLayer->getDevice()->createCommandEncoder({});
		commandEncoder->pipelineBarrier(RHI::BarrierDescriptor{
			(uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT,
			(uint32_t)RHI::PipelineStages::TRANSFER_BIT,
			(uint32_t)RHI::DependencyType::NONE,
			{}, {},
			{ RHI::TextureMemoryBarrierDescriptor{
				Core::ResourceManager::get()->getResource<GFX::Texture>(src)->texture.get(),
				RHI::ImageSubresourceRange{(uint32_t)RHI::TextureAspect::COLOR_BIT, 0,1,0,1},
				(uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT | (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT,
				(uint32_t)RHI::AccessFlagBits::TRANSFER_READ_BIT,
				RHI::TextureLayout::SHADER_READ_ONLY_OPTIMAL,
				RHI::TextureLayout::TRANSFER_SRC_OPTIMAL
			}}
		});
		commandEncoder->copyTextureToTexture(
			RHI::ImageCopyTexture{
				Core::ResourceManager::get()->getResource<GFX::Texture>(src)->texture.get()
			},
			RHI::ImageCopyTexture{
				Core::ResourceManager::get()->getResource<GFX::Texture>(copyDst)->texture.get()
			},
			RHI::Extend3D{ 800, 600, 1 }
		);
		commandEncoder->pipelineBarrier(RHI::BarrierDescriptor{
			(uint32_t)RHI::PipelineStages::TRANSFER_BIT,
			(uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT,
			(uint32_t)RHI::DependencyType::NONE,
			{}, {},
			{ RHI::TextureMemoryBarrierDescriptor{
				Core::ResourceManager::get()->getResource<GFX::Texture>(src)->texture.get(),
				RHI::ImageSubresourceRange{(uint32_t)RHI::TextureAspect::COLOR_BIT, 0,1,0,1},
				(uint32_t)RHI::AccessFlagBits::TRANSFER_READ_BIT,
				(uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT | (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT,
				RHI::TextureLayout::TRANSFER_SRC_OPTIMAL,
				RHI::TextureLayout::SHADER_READ_ONLY_OPTIMAL
			}}
		});
		commandEncoder->pipelineBarrier(RHI::BarrierDescriptor{
			(uint32_t)RHI::PipelineStages::TRANSFER_BIT,
			(uint32_t)RHI::PipelineStages::HOST_BIT,
			(uint32_t)RHI::DependencyType::NONE,
			{}, {},
			{ RHI::TextureMemoryBarrierDescriptor{
				Core::ResourceManager::get()->getResource<GFX::Texture>(copyDst)->texture.get(),
				RHI::ImageSubresourceRange{(uint32_t)RHI::TextureAspect::COLOR_BIT, 0,1,0,1},
				(uint32_t)RHI::AccessFlagBits::TRANSFER_WRITE_BIT,
				(uint32_t)RHI::AccessFlagBits::HOST_READ_BIT,
				RHI::TextureLayout::TRANSFER_DST_OPTIMAL,
				RHI::TextureLayout::TRANSFER_DST_OPTIMAL
			}}
			});
		rhiLayer->getDevice()->getGraphicsQueue()->submit({ commandEncoder->finish({}) });
		rhiLayer->getDevice()->waitIdle();
		std::future<bool> mapped = Core::ResourceManager::get()->getResource<GFX::Texture>(copyDst)->texture->mapAsync(
			(uint32_t)RHI::MapMode::READ, 0, 800 * 600 * sizeof(vec4));
		if (mapped.get()) {
			void* data = Core::ResourceManager::get()->getResource<GFX::Texture>(copyDst)->texture->getMappedRange(0, 800 * 600 * sizeof(vec4));
			std::string filepath = mainWindow->saveFile("", Core::WorldTimePoint::get().to_string() + ".hdr");
			Image::HDR::writeHDR(filepath, 800, 600, 4, reinterpret_cast<float*>(data));
			Core::ResourceManager::get()->getResource<GFX::Texture>(copyDst)->texture->unmap();
		}
	}
	
	/** Update the application every fixed update timestep */
	virtual auto FixedUpdate() noexcept -> void override {

	};

	virtual auto Exit() noexcept -> void override {
		rhiLayer->getDevice()->waitIdle();
		srenderer = nullptr;
		graph = nullptr;

		blases.clear();

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
	
	std::unique_ptr<RDG::Graph> graph = nullptr;

	std::vector<std::unique_ptr<RHI::BLAS>> blases;

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