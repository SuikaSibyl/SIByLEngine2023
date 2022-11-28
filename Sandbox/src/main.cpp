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
import Core.Log;
import Core.Memory;
import Core.IO;
import Core.Event;
import Core.Timer;
import Core.ECS;
import Core.Resource;

import Math.Vector;
import Math.Geometry;
import Math.Matrix;
import Math.Limits;
import Math.Transform;
import Math.Trigonometric;

import Tracer.Ray;
import Tracer.Camera;
import Tracer.Film;
import Tracer.Shape;
import Tracer.Filter;
import Tracer.Primitives;
import Tracer.Interactable;
import Tracer.Material;
import Tracer.Integrator;
import Tracer.Sampler;
import Tracer.Texture;
import Tracer.Light;

import Image.Color;
import Image.Image;
import Image.FileFormat;

import Platform.Window;
import Platform.System;
import Parallelism.Parallel;

import RHI;
import RHI.RHILayer;

import GFX.Resource;
import GFX.GFXManager;
import GFX.MeshLoader;
import GFX.Components;
import GFX.SceneNodeLoader;
import Application.Root;
import Application.Base;

import Editor.Core;
import Editor.Framework;
import Editor.GFX;
import Editor.Config;
import Editor.GFX.CameraController;

import Sandbox.Tracer;
import Sandbox.AAF_GI;
import Sandbox.Benchmark;
import Sandbox.MAAF;

import Image.FileFormat;

using namespace SIByL;
using namespace SIByL::Core;
using namespace SIByL::Math;

struct UniformBufferObject {
	Math::mat4 model;
	Math::mat4 view;
	Math::mat4 proj;
	Math::mat4 viewInverse;  // Camera inverse view matrix
	Math::mat4 projInverse;  // Camera inverse projection matrix
};

struct GlobalUniforms {
	Math::mat4 viewProj;     // Camera view * projection
	Math::mat4 viewInverse;  // Camera inverse view matrix
	Math::mat4 projInverse;  // Camera inverse projection matrix
};

struct PushConstantRay {
	Math::vec4 clearColor;
	Math::vec3 lightPosition;
	float lightIntensity;
	int   lightType;
};

struct SandBoxApplication :public Application::ApplicationBase {
	GFX::GameObjectHandle camera_go;
	/** Initialize the application */
	virtual auto Init() noexcept -> void override {
		// create optional layers: rhi, imgui, editor
		rhiLayer = std::make_unique<RHI::RHILayer>(RHI::RHILayerDescriptor{
				RHI::RHIBackend::Vulkan,
				(uint32_t)RHI::ContextExtension::RAY_TRACING
				| (uint32_t)RHI::ContextExtension::BINDLESS_INDEXING,
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

		//GFX::SceneNodeLoader_glTF::loadSceneNode("D:/Downloads/glTF-Sample-Models-master/glTF-Sample-Models-master/2.0/Sponza/glTF/Sponza.gltf", scene);
		//GFX::SceneNodeLoader_glTF::loadSceneNode("P:/GitProjects/SIByLEngine2022/Sandbox/content/scenes/cornellBox.gltf", scene);
		//scene.deserialize("P:/GitProjects/SIByLEngine2022/Sandbox/content/cornellBox.scene");
		//scene.deserialize("P:/GitProjects/SIByLEngine2022/Sandbox/content/cornellBoxSphere.scene");
		scene.deserialize("P:/GitProjects/SIByLEngine2022/Sandbox/content/sponza.scene");

		std::vector<RHI::TextureView*> textureViews;
		{
			// use tinygltf to load file
			std::filesystem::path path = "D:/Downloads/glTF-Sample-Models-master/glTF-Sample-Models-master/2.0/Sponza/glTF/Sponza.gltf";
			tinygltf::Model model;
			tinygltf::TinyGLTF loader;
			std::string err;
			std::string warn;
			bool ret = loader.LoadASCIIFromFile(&model, &err, &warn, path.string());
			if (!warn.empty()) Core::LogManager::Warning(std::format("GFX :: tinygltf :: {0}", warn.c_str()));
			if (!err.empty()) Core::LogManager::Error(std::format("GFX :: tinygltf :: {0}", err.c_str()));
			if (!ret) {
				Core::LogManager::Error("GFX :: tinygltf :: Failed to parse glTF");
				return;
			}
			// Iterate through all the meshes in the glTF file
			// Load meshes into Runtime resource managers.
			RHI::Device* device = GFX::GFXManager::get()->rhiLayer->getDevice();
			std::vector<Core::GUID> meshGUIDs = {};
			std::unordered_map<tinygltf::Mesh const*, Core::GUID> meshMap = {};
			for (auto const& gltfMat : model.materials) {
				auto const& glTFimage_base_color = model.images[model.textures[gltfMat.pbrMetallicRoughness.baseColorTexture.index].source];
				std::filesystem::path base_color_path = path.parent_path() / glTFimage_base_color.uri;
				Core::GUID guid;
				if (base_color_path.extension() == ".jpg") {
					std::unique_ptr<Image::Image<Image::COLOR_R8G8B8A8_UINT>> img = Image::JPEG::fromJPEG(base_color_path);
					guid = Core::ResourceManager::get()->requestRuntimeGUID<GFX::Texture>();
					GFX::GFXManager::get()->registerTextureResource(guid, img.get());
				}
				else if (base_color_path.extension() == ".png") {
					std::unique_ptr<Image::Image<Image::COLOR_R8G8B8A8_UINT>> img = Image::PNG::fromPNG(base_color_path);
					guid = Core::ResourceManager::get()->requestRuntimeGUID<GFX::Texture>();
					GFX::GFXManager::get()->registerTextureResource(guid, img.get());
				}
				else {
					Core::LogManager::Error("GFX :: glTF load error, not supported image extension!");
				}
				textureViews.push_back(Core::ResourceManager::get()->getResource<GFX::Texture>(guid)->originalView.get());
			}
		}

		camera_go = scene.createGameObject();
		cameraController.init(mainWindow.get()->getInput(), &timer);
		cameraController.bindTransform(scene.getGameObject(camera_go)->getEntity().getComponent<GFX::TransformComponent>());

		int i = 0;
		GFX::MeshReference* meshref = nullptr;
		for (auto handle : scene.gameObjects) {
			auto* go = scene.getGameObject(handle.first);
			Math::mat4 objectMat;
			meshref = go->getEntity().getComponent<GFX::MeshReference>();
			if (meshref) {
				GFX::TransformComponent* transform = go->getEntity().getComponent<GFX::TransformComponent>();
				objectMat = transform->getTransform() * objectMat;
				while (go->parent != Core::NULL_ENTITY) {
					go = scene.getGameObject(go->parent);
					GFX::TransformComponent* transform = go->getEntity().getComponent<GFX::TransformComponent>();
					objectMat = transform->getTransform() * objectMat;
				}
				RHI::BLASDescriptor blasDesc;
				for (auto& submehs : meshref->mesh->submeshes) {
					blasDesc.triangleGeometries.push_back(RHI::BLASTriangleGeometry{
						meshref->mesh->vertexBufferPosOnly.get(),
						meshref->mesh->indexBuffer.get(),
						meshref->mesh->vertexBuffer.get(),
						RHI::IndexFormat::UINT32_T,
						submehs.size,
						submehs.baseVertex,
						submehs.size / 3,
						uint32_t(submehs.offset * sizeof(uint32_t)),
						RHI::AffineTransformMatrix(objectMat),
						(uint32_t)RHI::BLASGeometryFlagBits::NO_DUPLICATE_ANY_HIT_INVOCATION,
						submehs.matID
						}
					);
				}
				blases.emplace_back(device->createBLAS(blasDesc));
			}
		}

		ASGroup = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ASGroup>();
		RHI::TLASDescriptor tlasDesc;
		uint32_t blasIdx = 0;
		for (auto& blas : blases) {
			tlasDesc.instances.push_back(RHI::BLASInstance{
				blas.get(),
				mat4{},
				blasIdx }
			);
		}
		GFX::GFXManager::get()->registerAsGroupResource(ASGroup, tlasDesc, 8);

		struct Vertex {
			Math::vec3 pos;
			Math::vec3 color;
			Math::vec2 uv;
		};

		std::unique_ptr<Image::Image<Image::COLOR_R8G8B8A8_UINT>> img = Image::JPEG::fromJPEG("./content/texture.jpg");
		Core::GUID guid = Core::ResourceManager::get()->requestRuntimeGUID<GFX::Texture>();
		GFX::GFXManager::get()->registerTextureResource(guid, img.get());
		GFX::Texture* texture = Core::ResourceManager::get()->getResource<GFX::Texture>(guid);

		GFX::GFXManager::get()->commonSampler.defaultSampler = Core::ResourceManager::get()->requestRuntimeGUID<GFX::Sampler>();
		GFX::GFXManager::get()->registerSamplerResource(GFX::GFXManager::get()->commonSampler.defaultSampler, RHI::SamplerDescriptor{
				RHI::AddressMode::REPEAT,
				RHI::AddressMode::REPEAT,
				RHI::AddressMode::REPEAT,
			});
		Core::ResourceManager::get()->getResource<GFX::Sampler>(GFX::GFXManager::get()->commonSampler.defaultSampler)->sampler->setName("DefaultSampler");
		//framebufferColorAttaches
		framebufferColorAttach = Core::ResourceManager::get()->requestRuntimeGUID<GFX::Texture>();
		framebufferDepthAttach = Core::ResourceManager::get()->requestRuntimeGUID<GFX::Texture>();
		rtTarget = Core::ResourceManager::get()->requestRuntimeGUID<GFX::Texture>();
		RHI::TextureDescriptor desc{
			{800,600,1},
			1, 1, RHI::TextureDimension::TEX2D,
			RHI::TextureFormat::RGBA8_UNORM,
			(uint32_t)RHI::TextureUsage::COLOR_ATTACHMENT | (uint32_t)RHI::TextureUsage::TEXTURE_BINDING,
			{ RHI::TextureFormat::RGBA8_UNORM }
		};
		GFX::GFXManager::get()->registerTextureResource(framebufferColorAttach, desc);
		desc.format = RHI::TextureFormat::RGBA32_FLOAT;
		desc.usage |= (uint32_t)RHI::TextureUsage::STORAGE_BINDING | (uint32_t)RHI::TextureUsage::COPY_SRC;
		GFX::GFXManager::get()->registerTextureResource(rtTarget, desc);
		desc.format = RHI::TextureFormat::DEPTH32_FLOAT;
		desc.usage = (uint32_t)RHI::TextureUsage::DEPTH_ATTACHMENT | (uint32_t)RHI::TextureUsage::TEXTURE_BINDING;
		desc.viewFormats = { RHI::TextureFormat::DEPTH32_FLOAT };
		GFX::GFXManager::get()->registerTextureResource(framebufferDepthAttach, desc);

		Buffer vert, frag;
		syncReadFile("../Engine/Binaries/Runtime/spirv/Common/test_shader_vert_vert.spv", vert);
		syncReadFile("../Engine/Binaries/Runtime/spirv/Common/test_shader_frag_frag.spv", frag);
		vert_module = device->createShaderModule({ &vert, RHI::ShaderStages::VERTEX });
		frag_module = device->createShaderModule({ &frag, RHI::ShaderStages::FRAGMENT });

		Core::GUID comp;
		comp = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();
		GFX::GFXManager::get()->registerShaderModuleResource(comp, "../Engine/Binaries/Runtime/spirv/Common/test_compute_comp.spv", { nullptr, RHI::ShaderStages::COMPUTE});
		
		// create uniformBuffer
		{
			for (int i = 0; i < 2; ++i) {
				RHI::BufferDescriptor uboDescriptor;
				uboDescriptor.size = sizeof(UniformBufferObject);
				uboDescriptor.usage = (uint32_t)RHI::BufferUsage::UNIFORM;
				uboDescriptor.memoryProperties = (uint32_t)RHI::MemoryProperty::HOST_VISIBLE_BIT
					| (uint32_t)RHI::MemoryProperty::HOST_COHERENT_BIT;
				uboDescriptor.mappedAtCreation = true;
				uniformBuffer[i] = device->createBuffer(uboDescriptor);
			}
		}
		{
			rtBindGroupLayout = device->createBindGroupLayout(
				RHI::BindGroupLayoutDescriptor{ {
					RHI::BindGroupLayoutEntry{ 0, 
						(uint32_t)RHI::ShaderStages::VERTEX 
						| (uint32_t)RHI::ShaderStages::RAYGEN 
						| (uint32_t)RHI::ShaderStages::COMPUTE
						| (uint32_t)RHI::ShaderStages::CLOSEST_HIT 
						| (uint32_t)RHI::ShaderStages::ANY_HIT, 
						RHI::BufferBindingLayout{RHI::BufferBindingType::UNIFORM}},
					RHI::BindGroupLayoutEntry{ 1,
							(uint32_t)RHI::ShaderStages::FRAGMENT
							| (uint32_t)RHI::ShaderStages::RAYGEN
							| (uint32_t)RHI::ShaderStages::COMPUTE
							| (uint32_t)RHI::ShaderStages::CLOSEST_HIT
							| (uint32_t)RHI::ShaderStages::ANY_HIT,
							RHI::BindlessTexturesBindingLayout{}},
					} }
			);

			bindGroupLayout_RT = device->createBindGroupLayout(
				RHI::BindGroupLayoutDescriptor{ {
					RHI::BindGroupLayoutEntry{0, (uint32_t)RHI::ShaderStages::RAYGEN | (uint32_t)RHI::ShaderStages::COMPUTE | (uint32_t)RHI::ShaderStages::CLOSEST_HIT | (uint32_t)RHI::ShaderStages::ANY_HIT, RHI::AccelerationStructureBindingLayout{}},
					RHI::BindGroupLayoutEntry{1, (uint32_t)RHI::ShaderStages::RAYGEN | (uint32_t)RHI::ShaderStages::COMPUTE | (uint32_t)RHI::ShaderStages::CLOSEST_HIT | (uint32_t)RHI::ShaderStages::ANY_HIT, RHI::StorageTextureBindingLayout{}},
					RHI::BindGroupLayoutEntry{2, (uint32_t)RHI::ShaderStages::RAYGEN | (uint32_t)RHI::ShaderStages::COMPUTE | (uint32_t)RHI::ShaderStages::CLOSEST_HIT | (uint32_t)RHI::ShaderStages::ANY_HIT, RHI::BufferBindingLayout{RHI::BufferBindingType::STORAGE}},
					RHI::BindGroupLayoutEntry{3, (uint32_t)RHI::ShaderStages::RAYGEN | (uint32_t)RHI::ShaderStages::COMPUTE | (uint32_t)RHI::ShaderStages::CLOSEST_HIT | (uint32_t)RHI::ShaderStages::ANY_HIT, RHI::BufferBindingLayout{RHI::BufferBindingType::STORAGE}},
					} }
			);

			for (int i = 0; i < 2; ++i) {
				rtBindGroup[i] = device->createBindGroup(RHI::BindGroupDescriptor{
					rtBindGroupLayout.get(),
					std::vector<RHI::BindGroupEntry>{
						{0,RHI::BindingResource{RHI::BufferBinding{uniformBuffer[i].get(), 0, uniformBuffer[i]->size()}}},
						{1,RHI::BindingResource{textureViews,
							Core::ResourceManager::get()->getResource<GFX::Sampler>(GFX::GFXManager::get()->commonSampler.defaultSampler)->sampler.get()}},
				} });
			}
		}

		for (int i = 0; i < 2; ++i) {
			pipelineLayout[i] = device->createPipelineLayout(RHI::PipelineLayoutDescriptor{
				{ {(uint32_t)RHI::ShaderStages::VERTEX, 0, sizeof(Math::mat4) + sizeof(uint32_t)}},
				{ rtBindGroupLayout.get() }
				});

			pipelineLayout_RT[i] = device->createPipelineLayout(RHI::PipelineLayoutDescriptor{
				{ {(uint32_t)RHI::ShaderStages::RAYGEN | (uint32_t)RHI::ShaderStages::CLOSEST_HIT | (uint32_t)RHI::ShaderStages::MISS | (uint32_t)RHI::ShaderStages::COMPUTE, 0, sizeof(PushConstantRay)}},
				{ bindGroupLayout_RT.get(), rtBindGroupLayout.get() }
				});

			renderPipeline[i] = device->createRenderPipeline(RHI::RenderPipelineDescriptor{
				pipelineLayout[i].get(),
				RHI::VertexState{
						// vertex shader
						vert_module.get(), "main",
						// vertex attribute layout
						{ RHI::VertexBufferLayout{sizeof(Vertex), RHI::VertexStepMode::VERTEX, {
							{ RHI::VertexFormat::FLOAT32X3, 0, 0},
							{ RHI::VertexFormat::FLOAT32X3, offsetof(Vertex,color), 1},
							{ RHI::VertexFormat::FLOAT32X2, offsetof(Vertex,uv), 2},}}}},
					RHI::PrimitiveState{ RHI::PrimitiveTopology::TRIANGLE_LIST, RHI::IndexFormat::UINT16_t },
					RHI::DepthStencilState{ RHI::TextureFormat::DEPTH32_FLOAT, true, RHI::CompareFunction::LESS },
					RHI::MultisampleState{},
					RHI::FragmentState{
						// fragment shader
						frag_module.get(), "main",
						{{RHI::TextureFormat::RGBA8_UNORM}}}
					});


			computePipeline[i] = device->createComputePipeline(RHI::ComputePipelineDescriptor{
				pipelineLayout_RT[i].get(),
				{Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp)->shaderModule.get(), "main"}
				});

			computePipeline[i]->setName("ComputeShader_RayTracer");
		}

		directTracer = std::make_unique<Sandbox::DirectTracer>(rhiLayer.get(), std::array<RHI::PipelineLayout*, 2>{ pipelineLayout_RT[0].get() ,pipelineLayout_RT[1].get() });

		aafPipeline = std::make_unique<Sandbox::AAFPipeline>(rhiLayer.get(),
			Core::ResourceManager::get()->getResource<GFX::ASGroup>(ASGroup), rtTarget,
			rtBindGroupLayout.get(), std::array<RHI::BindGroup*, 2>{rtBindGroup[0].get(), rtBindGroup[1].get()});
		aafGIPipeline = std::make_unique<Sandbox::AAF_GI_Pipeline>(rhiLayer.get(),
			Core::ResourceManager::get()->getResource<GFX::ASGroup>(ASGroup), rtTarget, 
			rtBindGroupLayout.get(), std::array<RHI::BindGroup*, 2>{rtBindGroup[0].get(), rtBindGroup[1].get()});
		benchmarkPipeline = std::make_unique<Sandbox::Benchmark_Pipeline>(rhiLayer.get(), 
			Core::ResourceManager::get()->getResource<GFX::ASGroup>(ASGroup), rtTarget, 
			rtBindGroupLayout.get(), std::array<RHI::BindGroup*, 2>{rtBindGroup[0].get(), rtBindGroup[1].get()});
		maafPipeline = std::make_unique<Sandbox::MAAF_Pipeline>(rhiLayer.get(),
			Core::ResourceManager::get()->getResource<GFX::ASGroup>(ASGroup), rtTarget, 
			rtBindGroupLayout.get(), std::array<RHI::BindGroup*, 2>{rtBindGroup[0].get(), rtBindGroup[1].get()});
	};

	/** Update the application every loop */
	virtual auto Update(double deltaTime) noexcept -> void override {
		{
			cameraController.onUpdate();
		}
		// start new frame
		imguiLayer->startNewFrame();
		// frame start
		RHI::Device* device = rhiLayer->getDevice();
		RHI::SwapChain* swapChain = rhiLayer->getSwapChain();
		RHI::MultiFrameFlights* multiFrameFlights = rhiLayer->getMultiFrameFlights();
		multiFrameFlights->frameStart();

		std::unique_ptr<RHI::CommandEncoder> commandEncoder = device->createCommandEncoder({ multiFrameFlights->getCommandBuffer() });

		auto* input = mainWindow.get()->getInput();
		if (input->isKeyPressed(Platform::SIByL_KEY_ENTER)) {
			Core::LogManager::Debug("Key pressed!");
		}

		RHI::RenderPassDescriptor renderPassDescriptor = {
			{ RHI::RenderPassColorAttachment{
				Core::ResourceManager::get()->getResource<GFX::Texture>
				(framebufferColorAttach)->originalView.get(),
			nullptr, {0,0,0,1}, RHI::LoadOp::CLEAR, RHI::StoreOp::STORE }},
			RHI::RenderPassDepthStencilAttachment{
				Core::ResourceManager::get()->getResource<GFX::Texture>(framebufferDepthAttach)->originalView.get(),
				1, RHI::LoadOp::CLEAR, RHI::StoreOp::DONT_CARE, false,
				0, RHI::LoadOp::CLEAR, RHI::StoreOp::DONT_CARE, false
		},
		};

		RHI::RayTracingPassDescriptor rayTracingDescriptor = {};

		uint32_t index = multiFrameFlights->getFlightIndex();

		int width, height;
		mainWindow->getFramebufferSize(&width, &height);
		width = 800;
		height = 600;

		UniformBufferObject ubo;
		//Math::vec4 campos = 
		//	Math::mul(Math::rotateY(std::sin(timer.totalTime()) * 20).m, Math::vec4(-0.001, 1.0, 6.0, 1));
		Math::vec4 campos = Math::mul(Math::rotateY(0 * 20).m, Math::vec4(-0.001, 1.0, 6.0, 1));

		GFX::TransformComponent* transform = scene.getGameObject(camera_go)->getEntity().getComponent<GFX::TransformComponent>();
		//ubo.model = Math::transpose(Math::rotate(timer.totalTime() * 80, Math::vec3(0, 1, 0)).m);
		
		//ubo.view = Math::transpose(Math::lookAt(Math::vec3(campos.x, campos.y, campos.z) , Math::vec3(0, 1, 0), Math::vec3(0, 1, 0)).m);

		ubo.view = Math::transpose(Math::lookAt(transform->translation, transform->translation + transform->getRotatedForward(), Math::vec3(0, 1, 0)).m);
		ubo.proj = Math::transpose(Math::perspective(22.f, 1.f * 800 / 600, 0.1f, 1000.f).m);
		//Math::vec4 campos = Math::vec4(-4.5f, 2.5f, 5.5f, 1);
		//{
		//	//campos.x = (float)(campos.x * sin(timer.totalTime() * 1));
		//	//campos.y = (float)(campos.y + cos(timer.totalTime() * 1 * 1.5));
		//	//campos.z = (float)(campos.z * cos(timer.totalTime() * 1));
		//}
		//ubo.view = Math::transpose(Math::lookAt(Math::vec3(campos.x, campos.y, campos.z), Math::vec3(-1, 0.5f, 0), Math::vec3(0, 1, 0)).m);
		//ubo.proj = Math::transpose(Math::perspective(60.f, 1.f * 800 / 600, 0.1f, 10.f).m);
		ubo.viewInverse = Math::inverse(ubo.view);
		//ubo.proj.data[1][1] *= -1;
		ubo.projInverse = Math::inverse(ubo.proj);
		Math::vec4 originPos = Math::vec4(0.f, 0.f, 0.f, 1.f);
		Math::mat4 invView = Math::transpose(ubo.viewInverse);
		auto originPost = Math::mul(invView, Math::vec4(0.f, 0.f, 0.f, 1.f));

		std::cout << 1.f / timer.deltaTime() << std::endl;
		//Math::rotate( )
		std::future<bool> mapped = uniformBuffer[index]->mapAsync(0, 0, sizeof(UniformBufferObject));
		if (mapped.get()) {
			void* data = uniformBuffer[index]->getMappedRange(0, sizeof(UniformBufferObject));
			memcpy(data, &ubo, sizeof(UniformBufferObject));
			uniformBuffer[index]->unmap();
		}

		commandEncoder->pipelineBarrier(RHI::BarrierDescriptor{
			(uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT,
			(uint32_t)RHI::PipelineStages::COLOR_ATTACHMENT_OUTPUT_BIT,
			(uint32_t)RHI::DependencyType::NONE,
			{}, {},
			{ RHI::TextureMemoryBarrierDescriptor{
				Core::ResourceManager::get()->getResource<GFX::Texture>(framebufferColorAttach)->texture.get(),
				RHI::ImageSubresourceRange{(uint32_t)RHI::TextureAspect::COLOR_BIT, 0,1,0,1},
				(uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT,
				(uint32_t)RHI::AccessFlagBits::COLOR_ATTACHMENT_WRITE_BIT,
				RHI::TextureLayout::SHADER_READ_ONLY_OPTIMAL,
				RHI::TextureLayout::COLOR_ATTACHMENT_OPTIMAL
			}}
			});

		commandEncoder->pipelineBarrier(RHI::BarrierDescriptor{
			(uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT,
			(uint32_t)RHI::PipelineStages::COLOR_ATTACHMENT_OUTPUT_BIT,
			(uint32_t)RHI::DependencyType::NONE,
			{}, {},
			{ RHI::TextureMemoryBarrierDescriptor{
				Core::ResourceManager::get()->getResource<GFX::Texture>(framebufferDepthAttach)->texture.get(),
				RHI::ImageSubresourceRange{(uint32_t)RHI::TextureAspect::DEPTH_BIT, 0,1,0,1},
				(uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT,
				(uint32_t)RHI::AccessFlagBits::DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
				RHI::TextureLayout::SHADER_READ_ONLY_OPTIMAL,
				RHI::TextureLayout::DEPTH_ATTACHMENT_OPTIMAL
			}}
			});

		//aafPipeline->composeCommands(commandEncoder.get(), index);
		aafGIPipeline->composeCommands(commandEncoder.get(), index);
		//benchmarkPipeline->composeCommands(commandEncoder.get(), index);
		//maafPipeline->composeCommands(commandEncoder.get(), index);

		//objectMat = Math::transpose(Math::mul(Math::translate(Math::vec3(0, 1, 0)).m, objectMat));
		passEncoder[index] = commandEncoder->beginRenderPass(renderPassDescriptor);
		passEncoder[index]->setPipeline(renderPipeline[index].get());
		passEncoder[index]->setViewport(0, 0, width, height, 0, 1);
		passEncoder[index]->setScissorRect(0, 0, width, height);


		struct PushConstant {
			Math::mat4 objectMat;
			uint32_t matID;
		};
		for (auto handle : scene.gameObjects) {
			auto* go = scene.getGameObject(handle.first);
			Math::mat4 objectMat;
			GFX::MeshReference* meshref = go->getEntity().getComponent<GFX::MeshReference>();
			if (meshref) {
				GFX::TransformComponent* transform = go->getEntity().getComponent<GFX::TransformComponent>();
				objectMat = transform->getTransform() * objectMat;
				while (go->parent != Core::NULL_ENTITY) {
					go = scene.getGameObject(go->parent);
					GFX::TransformComponent* transform = go->getEntity().getComponent<GFX::TransformComponent>();
					objectMat = transform->getTransform() * objectMat;
				}
				objectMat = Math::transpose(objectMat);
				passEncoder[index]->setVertexBuffer(0, meshref->mesh->vertexBuffer.get(),
					0, meshref->mesh->vertexBuffer->size());
				passEncoder[index]->setBindGroup(0, rtBindGroup[index].get(), 0, 0);
				passEncoder[index]->setIndexBuffer(meshref->mesh->indexBuffer.get(),
					RHI::IndexFormat::UINT32_T, 0, meshref->mesh->indexBuffer->size());
				for (auto& submehs : meshref->mesh->submeshes) {
					PushConstant constant{ objectMat, submehs.matID };
					passEncoder[index]->pushConstants(&constant,
						(uint32_t)RHI::ShaderStages::VERTEX,
						0, sizeof(PushConstant));

					passEncoder[index]->drawIndexed(submehs.size, 1, submehs.offset, submehs.baseVertex, 0);
				}
			}
		}


		passEncoder[index]->end();

		device->getGraphicsQueue()->submit({ commandEncoder->finish({}) }, 
			multiFrameFlights->getImageAvailableSeamaphore(),
			multiFrameFlights->getRenderFinishedSeamaphore(),
			multiFrameFlights->getFence());

		// GUI Recording
		imguiLayer->startGuiRecording();
		bool show_demo_window = true;
		ImGui::ShowDemoWindow(&show_demo_window);
		ImGui::Begin("Ray Tracer");
		ImGui::Image(
			Editor::TextureUtils::getImGuiTexture(rtTarget)->getTextureID(),
			{ (float)width,(float)height },
			{ 0,0 }, { 1, 1 });
		if (ImGui::Button("Capture", { 200,100 })) {
			captureImage(rtTarget);
		}

		ImGui::End();
		ImGui::Begin("Rasterizer");
		ImGui::Image(
			Editor::TextureUtils::getImGuiTexture(framebufferColorAttach)->getTextureID(),
			{ (float)width,(float)height },
			{ 0,0 }, { 1, 1 });
		if (ImGui::Button("Capture Rasterizer", { 200,100 })) {
			captureImage(framebufferColorAttach);
		}
		ImGui::End();
		editorLayer->onDrawGui();
		imguiLayer->render();

		multiFrameFlights->frameEnd();
	};

	auto captureImage(Core::GUID src) noexcept -> void {
		static Core::GUID copyDst = 0;
		if (copyDst == 0) {
			copyDst = Core::ResourceManager::get()->requestRuntimeGUID<GFX::Texture>();
			RHI::TextureDescriptor desc{
				{800,600,1},
				1, 1, RHI::TextureDimension::TEX2D,
				RHI::TextureFormat::RGBA32_FLOAT,
				(uint32_t)RHI::TextureUsage::COPY_DST,
				{ RHI::TextureFormat::RGBA32_FLOAT },
				true
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

		aafPipeline = nullptr;
		aafGIPipeline = nullptr;
		maafPipeline = nullptr;
		benchmarkPipeline = nullptr;
		directTracer = nullptr;
		renderPipeline[0] = nullptr;
		renderPipeline[1] = nullptr;
		computePipeline[0] = nullptr;
		computePipeline[1] = nullptr;
		passEncoder[0] = nullptr;
		passEncoder[1] = nullptr;
		compEncoder[1] = nullptr;
		compEncoder[1] = nullptr;
		pipelineLayout[0] = nullptr;
		pipelineLayout[1] = nullptr;
		pipelineLayout_RT[0] = nullptr;
		pipelineLayout_RT[1] = nullptr;
		pipelineLayout_COMP[0] = nullptr;
		pipelineLayout_COMP[1] = nullptr;

		bindGroupLayout_RT = nullptr;

		vert_module = nullptr;
		frag_module = nullptr;
		blas = nullptr;
		blases.clear();

		rtBindGroupLayout = nullptr;
		uniformBuffer[0] = nullptr;
		uniformBuffer[1] = nullptr;
		rtBindGroup[0] = nullptr;
		rtBindGroup[1] = nullptr;

		editorLayer = nullptr;
		imguiLayer = nullptr;

		ResourceManager::get()->clear();

		rhiLayer = nullptr;
	}

private:
	Core::GUID framebufferColorAttach;
	Core::GUID framebufferDepthAttach;
	Core::GUID rtTarget;

	PushConstantRay pcRay = {};
	uint32_t indexCount = 0;

	std::unique_ptr<RHI::RHILayer> rhiLayer = nullptr;
	std::unique_ptr<Editor::ImGuiLayer> imguiLayer = nullptr;
	std::unique_ptr<Editor::EditorLayer> editorLayer = nullptr;

	std::unique_ptr<RHI::BindGroupLayout> rtBindGroupLayout = nullptr;
	std::unique_ptr<RHI::BindGroup> rtBindGroup[2];

	std::unique_ptr<RHI::BindGroupLayout> bindGroupLayout_RT = nullptr;
	
	Core::GUID cornellBox;
	Core::GUID ASGroup;

	std::unique_ptr<RHI::Buffer> uniformBuffer[2];
	std::unique_ptr<RHI::PipelineLayout> pipelineLayout[2];
	std::unique_ptr<RHI::PipelineLayout> pipelineLayout_RT[2];
	std::unique_ptr<RHI::PipelineLayout> pipelineLayout_COMP[2];

	std::unique_ptr<RHI::ShaderModule> vert_module = nullptr;
	std::unique_ptr<RHI::ShaderModule> frag_module = nullptr;
	std::unique_ptr<RHI::RenderPassEncoder> passEncoder[2] = {};
	std::unique_ptr<RHI::ComputePassEncoder> compEncoder[2] = {};
	std::unique_ptr<RHI::BLAS> blas = nullptr;
	std::vector<std::unique_ptr<RHI::BLAS>> blases;

	std::unique_ptr<Sandbox::DirectTracer> directTracer = nullptr;
	std::unique_ptr<Sandbox::AAFPipeline> aafPipeline = nullptr;
	std::unique_ptr<Sandbox::AAF_GI_Pipeline> aafGIPipeline = nullptr;
	std::unique_ptr<Sandbox::Benchmark_Pipeline> benchmarkPipeline = nullptr;
	std::unique_ptr<Sandbox::MAAF_Pipeline> maafPipeline = nullptr;

	std::unique_ptr<RHI::ComputePipeline> computePipeline[2];
	std::unique_ptr<RHI::RenderPipeline> renderPipeline[2];
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
			L"SIByL Engine 2022.0",
			1280, 720,
			Platform::WindowProperties::VULKAN_CONTEX
		});
	app.run();

	//Scope<Platform::Window> window = Platform::Window::create({
	//		Platform::WindowVendor::WIN_64,
	//		L"Ray Trace Viewer",
	//		720, 480,
	//		Platform::WindowProperties::OPENGL_CONTEX
	//	});

	//int ncores = Platform::getNumSystemCores();
	//Image::Image<Image::COLOR_R8G8B8_UINT> image(720, 480);
	//std::fill((Image::COLOR_R8G8B8_UINT*) & ((reinterpret_cast<char*>(image.data.data))[0]), 
	//	(Image::COLOR_R8G8B8_UINT*)&((reinterpret_cast<char*>(image.data.data))[image.data.size]), 
	//	Image::COLOR_R8G8B8_UINT{ {255,255,0} });

	//window->bindPaintingBitmapRGB8(size_t(720), size_t(480), (char*)image.data.data);
	//window->resize(720.f, 480);

	//Math::Transform defaultTransform(mat4{});
	//Math::AnimatedTransform animatedDefaultTransform(&defaultTransform, 0, &defaultTransform, 0);
	//Tracer::Film film(Math::ipoint2{ 720, 480 }, Math::bounds2{ {0,0}, {1,1} }, 
	//	std::make_unique<Tracer::BoxFilter>(Math::vec2{1.f,1.f}), 1, "what.c", 1);
	//Tracer::OrthographicCamera camera(animatedDefaultTransform, 
	//	Math::bounds2{ {-1.f * 720.f / 480.f,-1.f}, {1.f * 720.f / 480.f, 1.f} } , 
	//	0, 0, 0, 0, &film, nullptr);

	//float const radius = 0.5f;
	//float const radius_2 = 3.f;
	//Math::Transform objectToWorld = Math::translate({ 0,0,radius_2 });
	//Math::Transform worldToObject = Math::translate({ 0,0,-radius_2 });
	//Tracer::Sphere sphere(&objectToWorld, &worldToObject, false, radius, -radius, radius, 360);
	//Tracer::MatteMaterial material(nullptr, nullptr, nullptr);
	//Tracer::GeometricPrimitive primitve;
	//primitve.shape = &sphere;
	//primitve.material = &material;

	//Math::Transform objectToWorld_2 = Math::translate({ 0,radius_2+ radius,radius_2 });
	//Math::Transform worldToObject_2 = Math::translate({ 0,-radius_2- radius,-radius_2 });

	//Tracer::Sphere sphere_2(&objectToWorld_2, &worldToObject_2, false, radius_2, -radius_2, radius_2, 360);
	//Tracer::GeometricPrimitive ground;
	//ground.shape = &sphere_2;
	//ground.material = &material;
	//Tracer::DummyAggregate aggregate{ std::vector<Tracer::Primitive*>{&primitve,& ground} };
	//Tracer::InfiniteAreaLight areaLight(Math::Transform{}, Tracer::Spectrum{ 1.f }, 1, "");
	//Tracer::Scene scene(&aggregate, { &areaLight });
	//Tracer::StratifiedSampler sampler(10, 10, true, 50);
	//Tracer::WhittedIntegrator integrator(5, &camera, &sampler);

	//std::thread render(std::bind(&Tracer::WhittedIntegrator::render, &integrator, scene));

	//int i = 0;
	//while (window->isRunning()) {
	//	auto startPoint = std::chrono::high_resolution_clock::now();
	//	camera.film->writeImage(image, 1.f);
	//	//// Clear background to black
	//	//std::fill((Image::COLOR_R8G8B8_UINT*)&((reinterpret_cast<char*>(image.data.data))[0]),
	//	//	(Image::COLOR_R8G8B8_UINT*)&((reinterpret_cast<char*>(image.data.data))[image.data.size]),
	//	//	Image::COLOR_R8G8B8_UINT{ 0 ,0 ,0 });

	//	//float tHit;
	//	//Math::ivec2 sampleExtent{ 720,480 };
	//	//int const tileSize = 16;
	//	//Math::ipoint2 nTiles((sampleExtent.x + tileSize - 1) / tileSize,
	//	//	(sampleExtent.y + tileSize - 1) / tileSize);

	//	//Parallelism::ParallelFor2D([&](Math::ipoint2 tile)->void {
	//	//	Tracer::Ray ray;
	//	//	for (int i = 0; i < tileSize; ++i)
	//	//		for (int j = 0; j < tileSize; ++j) {
	//	//			image[tile.y * tileSize + j][tile.x * tileSize + i] = Image::COLOR_R8G8B8_UINT{  0,0 ,255 };
	//	//		}
	//	//}, nTiles);

	//	window->fetchEvents();
	//	window->invalid();

	//	auto endPoint = std::chrono::high_resolution_clock::now();
	//	long long start = std::chrono::time_point_cast<std::chrono::microseconds>(startPoint).time_since_epoch().count();
	//	long long end = std::chrono::time_point_cast<std::chrono::microseconds>(endPoint).time_since_epoch().count();
	//	long long time = end - start;
	//	std::cout << "Time each frame: " << (time * 1. / 1000000) << std::endl;
	//}
	//render.join();
	//window->destroy();
	//Parallelism::clearThreadPool();
}