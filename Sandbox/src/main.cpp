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

import SE.Math.Misc;
import SE.Math.Geometric;

import SE.Platform.Window;
import SE.Platform.Misc;
import SE.Image;
import SE.RHI;
import SE.Parallelism;

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

import SE.GFX.Core;
import SE.GFX.RDG;

import SE.Application;

import SE.Editor.Core;
import SE.Editor.GFX;
import SE.Editor.Config;
//
//import Sandbox.Tracer;
//import Sandbox.AAF_GI;
//import Sandbox.Benchmark;
//import Sandbox.MAAF;


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

struct SandBoxApplication :public Application::ApplicationBase {
	GFX::GameObjectHandle camera_go;

	/** Initialize the application */
	virtual auto Init() noexcept -> void override {
		// create optional layers: rhi, imgui, editor
		rhiLayer = std::make_unique<RHI::RHILayer>(RHI::RHILayerDescriptor{
				RHI::RHIBackend::Vulkan,
				RHI::ContextExtensionsFlags(
					RHI::ContextExtension::RAY_TRACING
					| RHI::ContextExtension::BINDLESS_INDEXING),
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

		struct Vertex {
			Math::vec3 pos;
			Math::vec3 color;
			Math::vec2 uv;
		};


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

		GFX::GFXManager::get()->commonSampler.defaultSampler = Core::ResourceManager::get()->requestRuntimeGUID<GFX::Sampler>();
		GFX::GFXManager::get()->registerSamplerResource(GFX::GFXManager::get()->commonSampler.defaultSampler, RHI::SamplerDescriptor{
				RHI::AddressMode::REPEAT,
				RHI::AddressMode::REPEAT,
				RHI::AddressMode::REPEAT,
			});
		Core::ResourceManager::get()->getResource<GFX::Sampler>(GFX::GFXManager::get()->commonSampler.defaultSampler)->sampler->setName("DefaultSampler");

		rdg = std::make_unique<GFX::RDGraph>();
		GFX::RDGTexture* texRes_rasterizer_target_color = rdg->createTexture(
			"RasterizerTarget_Color", 
			GFX::RDGTexture::Desc{
				{800,600,1},
				1, 1, RHI::TextureDimension::TEX2D,
				RHI::TextureFormat::RGBA8_UNORM }
		);
		GFX::RDGTexture* texRes_rasterizer_target_depth = rdg->createTexture(
			"RasterizerTarget_Depth", 
			GFX::RDGTexture::Desc{
				{800,600,1},
				1, 1, RHI::TextureDimension::TEX2D,
				RHI::TextureFormat::DEPTH32_FLOAT }
		);
		GFX::RDGStructuredUniformBuffer<UniformBufferObject>* cameraUniformBuffer = rdg->createStructuredUniformBuffer<UniformBufferObject>("camera_uniform_buffer");

		GFX::RDGPassNode* pass = rdg->addPass("DrawAlbedoPass", GFX::RDGPassFlag::RASTER,
			[&]()->void {
				// consume
				rdg->getTexture("RasterizerTarget_Color")->consume(GFX::ConsumeType::RENDER_TARGET, RHI::TextureLayout::COLOR_ATTACHMENT_OPTIMAL, RHI::TextureUsage::COLOR_ATTACHMENT);
				rdg->getTexture("RasterizerTarget_Depth")->consume(GFX::ConsumeType::RENDER_TARGET, RHI::TextureLayout::DEPTH_ATTACHMENT_OPTIMAL, RHI::TextureUsage::DEPTH_ATTACHMENT);
			},
			[&]()->GFX::CustomPassExecuteFn {
				// construction
				// shader
				Core::GUID vert, frag;
				vert = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();
				frag = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();
				GFX::GFXManager::get()->registerShaderModuleResource(vert, "../Engine/Binaries/Runtime/spirv/Common/test_shader_vert_vert.spv", { nullptr, RHI::ShaderStages::VERTEX });
				GFX::GFXManager::get()->registerShaderModuleResource(frag, "../Engine/Binaries/Runtime/spirv/Common/test_shader_frag_frag.spv", { nullptr, RHI::ShaderStages::FRAGMENT });
				// bindgroup layout
				std::shared_ptr<RHI::BindGroupLayout> rtBindGroupLayout = device->createBindGroupLayout(
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
				std::shared_ptr<RHI::BindGroup> rtBindGroup[2];
				for (int i = 0; i < 2; ++i) {
					rtBindGroup[i] = device->createBindGroup(RHI::BindGroupDescriptor{
						rtBindGroupLayout.get(),
						std::vector<RHI::BindGroupEntry>{
							{0,RHI::BindingResource{rdg->getStructuredUniformBuffer<UniformBufferObject>("camera_uniform_buffer")->getBufferBinding(i)}},
							{1,RHI::BindingResource{textureViews,
								Core::ResourceManager::get()->getResource<GFX::Sampler>(GFX::GFXManager::get()->commonSampler.defaultSampler)->sampler.get()}},
					} });
				}
				//std::shared_ptr<RHI::BindGroupLayout> bindGroupLayout_RT = device->createBindGroupLayout(
				//	RHI::BindGroupLayoutDescriptor{ {
				//		RHI::BindGroupLayoutEntry{0, (uint32_t)RHI::ShaderStages::RAYGEN | (uint32_t)RHI::ShaderStages::COMPUTE | (uint32_t)RHI::ShaderStages::CLOSEST_HIT | (uint32_t)RHI::ShaderStages::ANY_HIT, RHI::AccelerationStructureBindingLayout{}},
				//		RHI::BindGroupLayoutEntry{1, (uint32_t)RHI::ShaderStages::RAYGEN | (uint32_t)RHI::ShaderStages::COMPUTE | (uint32_t)RHI::ShaderStages::CLOSEST_HIT | (uint32_t)RHI::ShaderStages::ANY_HIT, RHI::StorageTextureBindingLayout{}},
				//		RHI::BindGroupLayoutEntry{2, (uint32_t)RHI::ShaderStages::RAYGEN | (uint32_t)RHI::ShaderStages::COMPUTE | (uint32_t)RHI::ShaderStages::CLOSEST_HIT | (uint32_t)RHI::ShaderStages::ANY_HIT, RHI::BufferBindingLayout{RHI::BufferBindingType::STORAGE}},
				//		RHI::BindGroupLayoutEntry{3, (uint32_t)RHI::ShaderStages::RAYGEN | (uint32_t)RHI::ShaderStages::COMPUTE | (uint32_t)RHI::ShaderStages::CLOSEST_HIT | (uint32_t)RHI::ShaderStages::ANY_HIT, RHI::BufferBindingLayout{RHI::BufferBindingType::STORAGE}},
				//		} }
				//);

				std::shared_ptr<RHI::PipelineLayout> pipelineLayout = device->createPipelineLayout(RHI::PipelineLayoutDescriptor{
					{ {(uint32_t)RHI::ShaderStages::VERTEX, 0, sizeof(Math::mat4) + sizeof(uint32_t)}},
					{ rtBindGroupLayout.get() }
					});

				//std::unique_ptr<RHI::PipelineLayout> pipelineLayout_RT = device->createPipelineLayout(RHI::PipelineLayoutDescriptor{
				//	{ {(uint32_t)RHI::ShaderStages::RAYGEN | (uint32_t)RHI::ShaderStages::CLOSEST_HIT | (uint32_t)RHI::ShaderStages::MISS | (uint32_t)RHI::ShaderStages::COMPUTE, 0, sizeof(PushConstantRay)}},
				//	{ bindGroupLayout_RT.get(), rtBindGroupLayout.get() }
				//	});

				std::shared_ptr<RHI::RenderPipeline> renderPipeline[2];
				for (int i = 0; i < 2; ++i) {					
					renderPipeline[i] = device->createRenderPipeline(RHI::RenderPipelineDescriptor{
						pipelineLayout.get(),
						RHI::VertexState{
							// vertex shader
							Core::ResourceManager::get()->getResource<GFX::ShaderModule>(vert)->shaderModule.get(), "main",
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
							Core::ResourceManager::get()->getResource<GFX::ShaderModule>(frag)->shaderModule.get(), "main",
							{{RHI::TextureFormat::RGBA8_UNORM}}}
						});
				}
				std::shared_ptr<RHI::RenderPassEncoder> passEncoder[2] = {};
				RHI::MultiFrameFlights* multiFrameFlights = rhiLayer->getMultiFrameFlights();

				RHI::RenderPassDescriptor renderPassDescriptor = {
					{ RHI::RenderPassColorAttachment{
						rdg->getTexture("RasterizerTarget_Color")->texture->originalView.get(),
					nullptr, {0,0,0,1}, RHI::LoadOp::CLEAR, RHI::StoreOp::STORE }},
					RHI::RenderPassDepthStencilAttachment{
						rdg->getTexture("RasterizerTarget_Depth")->texture->originalView.get(),
						1, RHI::LoadOp::CLEAR, RHI::StoreOp::DONT_CARE, false,
						0, RHI::LoadOp::CLEAR, RHI::StoreOp::DONT_CARE, false
				},
				};

				// execute callback
				GFX::CustomPassExecuteFn fn = [
					multiFrameFlights, scene=&scene, renderPassDescriptor, pipelineLayout = std::move(pipelineLayout),
					pipelines = std::array<std::shared_ptr<RHI::RenderPipeline>, 2>{ std::move(renderPipeline[0]), std::move(renderPipeline[1]) },
					rtBindGroup = std::array<std::shared_ptr<RHI::BindGroup>, 2>{ std::move(rtBindGroup[0]), std::move(rtBindGroup[1]) },
					passEncoders = std::array<std::shared_ptr<RHI::RenderPassEncoder>, 2>{ std::move(passEncoder[0]), std::move(passEncoder[1]) }
				](GFX::RDGRegistry const& registry, RHI::CommandEncoder* cmdEncoder) mutable ->void {
					uint32_t index = multiFrameFlights->getFlightIndex();
					passEncoders[index] = cmdEncoder->beginRenderPass(renderPassDescriptor);
					passEncoders[index]->setPipeline(pipelines[index].get());
					int width = 800, height = 600;
					passEncoders[index]->setViewport(0, 0, width, height, 0, 1);
					passEncoders[index]->setScissorRect(0, 0, width, height);

					struct PushConstant {
						Math::mat4 objectMat;
						uint32_t matID;
					};
					for (auto handle : scene->gameObjects) {
						GFX::GameObject* go = scene->getGameObject(handle.first);
						Math::mat4 objectMat;
						GFX::MeshReference* meshref = go->getEntity().getComponent<GFX::MeshReference>();
						if (meshref) {
							GFX::TransformComponent* transform = go->getEntity().getComponent<GFX::TransformComponent>();
							objectMat = transform->getTransform() * objectMat;
							while (go->parent != Core::NULL_ENTITY) {
								go = scene->getGameObject(go->parent);
								GFX::TransformComponent* transform = go->getEntity().getComponent<GFX::TransformComponent>();
								objectMat = transform->getTransform() * objectMat;
							}
							objectMat = Math::transpose(objectMat);
							passEncoders[index]->setVertexBuffer(0, meshref->mesh->vertexBuffer.get(),
								0, meshref->mesh->vertexBuffer->size());
							passEncoders[index]->setBindGroup(0, rtBindGroup[index].get(), 0, 0);
							passEncoders[index]->setIndexBuffer(meshref->mesh->indexBuffer.get(),
								RHI::IndexFormat::UINT32_T, 0, meshref->mesh->indexBuffer->size());
							for (auto& submehs : meshref->mesh->submeshes) {
								PushConstant constant{ objectMat, submehs.matID };
								passEncoders[index]->pushConstants(&constant,
									(uint32_t)RHI::ShaderStages::VERTEX,
									0, sizeof(PushConstant));

								passEncoders[index]->drawIndexed(submehs.size, 1, submehs.offset, submehs.baseVertex, 0);
							}
						}
					}
					passEncoders[index]->end();
				};
				return fn;
			});
		rdg->compile();

		camera_go = scene.createGameObject();
		cameraController.init(mainWindow.get()->getInput(), &timer);
		cameraController.bindTransform(scene.getGameObject(camera_go)->getEntity().getComponent<GFX::TransformComponent>());

	/*	int i = 0;
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

		Core::GUID ASGroup = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ASGroup>();
		RHI::TLASDescriptor tlasDesc;
		uint32_t blasIdx = 0;
		for (auto& blas : blases) {
			tlasDesc.instances.push_back(RHI::BLASInstance{
				blas.get(),
				mat4{},
				blasIdx }
			);
		}
		GFX::GFXManager::get()->registerAsGroupResource(ASGroup, tlasDesc, 8);*/

		//directTracer = std::make_unique<Sandbox::DirectTracer>(rhiLayer.get(), std::array<RHI::PipelineLayout*, 2>{ pipelineLayout_RT.get() ,pipelineLayout_RT.get() });

		//aafPipeline = std::make_unique<Sandbox::AAFPipeline>(rhiLayer.get(),
		//	Core::ResourceManager::get()->getResource<GFX::ASGroup>(ASGroup), rtTarget,
		//	rtBindGroupLayout.get(), std::array<RHI::BindGroup*, 2>{rtBindGroup[0].get(), rtBindGroup[1].get()});
		//aafGIPipeline = std::make_unique<Sandbox::AAF_GI_Pipeline>(rhiLayer.get(),
		//	Core::ResourceManager::get()->getResource<GFX::ASGroup>(ASGroup), rtTarget, 
		//	rtBindGroupLayout.get(), std::array<RHI::BindGroup*, 2>{rtBindGroup[0].get(), rtBindGroup[1].get()});
		//benchmarkPipeline = std::make_unique<Sandbox::Benchmark_Pipeline>(rhiLayer.get(), 
		//	Core::ResourceManager::get()->getResource<GFX::ASGroup>(ASGroup), rtTarget, 
		//	rtBindGroupLayout.get(), std::array<RHI::BindGroup*, 2>{rtBindGroup[0].get(), rtBindGroup[1].get()});
		//maafPipeline = std::make_unique<Sandbox::MAAF_Pipeline>(rhiLayer.get(),
		//	Core::ResourceManager::get()->getResource<GFX::ASGroup>(ASGroup), rtTarget, 
		//	rtBindGroupLayout.get(), std::array<RHI::BindGroup*, 2>{rtBindGroup[0].get(), rtBindGroup[1].get()});
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

		RHI::RayTracingPassDescriptor rayTracingDescriptor = {};

		uint32_t index = multiFrameFlights->getFlightIndex();

		int width, height;
		mainWindow->getFramebufferSize(&width, &height);
		width = 800;
		height = 600;

		UniformBufferObject ubo;
		GFX::TransformComponent* transform = scene.getGameObject(camera_go)->getEntity().getComponent<GFX::TransformComponent>();
		ubo.view = Math::transpose(Math::lookAt(transform->translation, transform->translation + transform->getRotatedForward(), Math::vec3(0, 1, 0)).m);
		ubo.proj = Math::transpose(Math::perspective(22.f, 1.f * 800 / 600, 0.1f, 1000.f).m);
		ubo.viewInverse = Math::inverse(ubo.view);
		ubo.projInverse = Math::inverse(ubo.proj);
		Math::vec4 originPos = Math::vec4(0.f, 0.f, 0.f, 1.f);
		Math::mat4 invView = Math::transpose(ubo.viewInverse);
		auto originPost = Math::mul(invView, Math::vec4(0.f, 0.f, 0.f, 1.f));

		std::cout << 1.f / timer.deltaTime() << std::endl;
		//Math::rotate( )
		rdg->getStructuredUniformBuffer<UniformBufferObject>("camera_uniform_buffer")->setStructure(ubo, index);

		commandEncoder->pipelineBarrier(RHI::BarrierDescriptor{
			(uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT,
			(uint32_t)RHI::PipelineStages::COLOR_ATTACHMENT_OUTPUT_BIT,
			(uint32_t)RHI::DependencyType::NONE,
			{}, {},
			{ RHI::TextureMemoryBarrierDescriptor{
				rdg->getTexture("RasterizerTarget_Color")->texture->texture.get(),
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
				rdg->getTexture("RasterizerTarget_Depth")->texture->texture.get(),
				RHI::ImageSubresourceRange{(uint32_t)RHI::TextureAspect::DEPTH_BIT, 0,1,0,1},
				(uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT,
				(uint32_t)RHI::AccessFlagBits::DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
				RHI::TextureLayout::SHADER_READ_ONLY_OPTIMAL,
				RHI::TextureLayout::DEPTH_ATTACHMENT_OPTIMAL
			}}
			});

		rdg->execute(commandEncoder.get());


		device->getGraphicsQueue()->submit({ commandEncoder->finish({}) }, 
			multiFrameFlights->getImageAvailableSeamaphore(),
			multiFrameFlights->getRenderFinishedSeamaphore(),
			multiFrameFlights->getFence());

		// GUI Recording
		imguiLayer->startGuiRecording();
		bool show_demo_window = true;
		ImGui::ShowDemoWindow(&show_demo_window);
		//ImGui::Begin("Ray Tracer");
		//ImGui::Image(
		//	Editor::TextureUtils::getImGuiTexture(rtTarget)->getTextureID(),
		//	{ (float)width,(float)height },
		//	{ 0,0 }, { 1, 1 });
		//if (ImGui::Button("Capture", { 200,100 })) {
		//	captureImage(rtTarget);
		//}

		//ImGui::End();
		ImGui::Begin("Rasterizer");
		ImGui::Image(
			Editor::TextureUtils::getImGuiTexture(rdg->getTexture("RasterizerTarget_Color")->guid)->getTextureID(),
			{ (float)width,(float)height },
			{ 0,0 }, { 1, 1 });
		if (ImGui::Button("Capture Rasterizer", { 200,100 })) {
			captureImage(rdg->getTexture("RasterizerTarget_Color")->guid);
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
		rdg = nullptr;

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
	
	std::unique_ptr<GFX::RDGraph> rdg = nullptr;

	std::vector<std::unique_ptr<RHI::BLAS>> blases;

	//std::unique_ptr<Sandbox::DirectTracer> directTracer = nullptr;
	//std::unique_ptr<Sandbox::AAFPipeline> aafPipeline = nullptr;
	//std::unique_ptr<Sandbox::AAF_GI_Pipeline> aafGIPipeline = nullptr;
	//std::unique_ptr<Sandbox::Benchmark_Pipeline> benchmarkPipeline = nullptr;
	//std::unique_ptr<Sandbox::MAAF_Pipeline> maafPipeline = nullptr;

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
			L"SIByL Engine 2022.1",
			1280, 720,
			Platform::WindowProperties::VULKAN_CONTEX
		});
	app.run();
}