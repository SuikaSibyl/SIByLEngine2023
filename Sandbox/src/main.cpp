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
import Core.Log;
import Core.Memory;
import Core.IO;
import Core.Event;
import Core.Timer;
import Core.ECS;
import Core.Resource.RuntimeManage;

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

import Application.Root;
import Application.Base;

import Editor.Core;
import Editor.Framework;
import Editor.GFX;
import Editor.Config;

using namespace SIByL;
using namespace SIByL::Core;
using namespace SIByL::Math;

struct UniformBufferObject {
	Math::mat4 model;
	Math::mat4 view;
	Math::mat4 proj;
};

struct SandBoxApplication :public Application::ApplicationBase {
	/** Initialize the application */
	virtual auto Init() noexcept -> void override {
		// create optional layers: rhi, imgui, editor
		rhiLayer = std::make_unique<RHI::RHILayer>(RHI::RHILayerDescriptor{
				RHI::RHIBackend::Vulkan,
				(uint32_t)RHI::ContextExtension::RAY_TRACING,
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
			Math::vec2 pos;
			Math::vec3 color;
		};

		struct VertexRT {
			Math::vec3 pos;
		};

		std::vector<Vertex> const vertices = {
			{{-0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}},
			{{0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}},
			{{0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}},
			{{-0.5f, 0.5f}, {1.0f, 1.0f, 1.0f}}
		};
		std::vector<uint16_t> const indices = {
			0, 1, 2, 2, 3, 0
		};

		GFX::Mesh mesh;
		mesh.vertexBuffer = device->createDeviceLocalBuffer((void*)vertices.data(), sizeof(vertices[0]) * vertices.size(),
			(uint32_t)RHI::BufferUsage::VERTEX);
		mesh.indexBuffer = device->createDeviceLocalBuffer((void*)indices.data(), sizeof(indices[0]) * indices.size(),
			(uint32_t)RHI::BufferUsage::INDEX | (uint32_t)RHI::BufferUsage::SHADER_DEVICE_ADDRESS |
			(uint32_t)RHI::BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY);
		ResourceManager::get()->addResource<GFX::Mesh>(0, std::move(mesh));
		GFX::Mesh* mesh_resource = ResourceManager::get()->getResource<GFX::Mesh>(0);

		std::vector<VertexRT> const verticesRT = {
			{{-0.5f, -0.5f, 0.0f}},
			{{0.5f, -0.5f, 0.0f}},
			{{0.5f, 0.5f, 0.0f}},
			{{-0.5f, 0.5f, 0.0f}},
		};
		vertexBufferRT = device->createDeviceLocalBuffer((void*)verticesRT.data(), sizeof(verticesRT[0]) * verticesRT.size(),
			(uint32_t)RHI::BufferUsage::VERTEX | (uint32_t)RHI::BufferUsage::SHADER_DEVICE_ADDRESS |
			(uint32_t)RHI::BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY);

		blas = device->createBLAS(RHI::BLASDescriptor{
			vertexBufferRT.get(),
			mesh_resource->indexBuffer.get(),
			(uint32_t)verticesRT.size(),
			2,
			RHI::IndexFormat::UINT16_t });
		tlas = device->createTLAS(RHI::TLASDescriptor{ {blas.get()} });

		{
			std::unique_ptr<Image::Image<Image::COLOR_R8G8B8A8_UINT>> img = Image::JPEG::fromJPEG("./content/texture.jpg");
			Core::GUID guid = Core::ResourceManager::get()->requestRuntimeGUID<GFX::Texture>();
			GFX::GFXManager::get()->registerTextureResource(guid, img.get());
			GFX::Texture* texture = Core::ResourceManager::get()->getResource<GFX::Texture>(guid);
			sampler = device->createSampler(RHI::SamplerDescriptor{});
			imguiTexture = imguiLayer->createImGuiTexture(sampler.get(), texture->originalView.get(), RHI::TextureLayout::SHADER_READ_ONLY_OPTIMAL);
		
			//framebufferColorAttaches
			framebufferColorAttaches[0] = Core::ResourceManager::get()->requestRuntimeGUID<GFX::Texture>();
			framebufferColorAttaches[1] = Core::ResourceManager::get()->requestRuntimeGUID<GFX::Texture>();
			framebufferDepthAttach = Core::ResourceManager::get()->requestRuntimeGUID<GFX::Texture>();
			RHI::TextureDescriptor desc{
				{720,480,1},
				1, 1, RHI::TextureDimension::TEX2D,
				RHI::TextureFormat::RGBA8_UNORM,
				(uint32_t)RHI::TextureUsage::COLOR_ATTACHMENT | (uint32_t)RHI::TextureUsage::TEXTURE_BINDING,
				{ RHI::TextureFormat::RGBA8_UNORM }
			};
			GFX::GFXManager::get()->registerTextureResource(framebufferColorAttaches[0], desc);
			GFX::GFXManager::get()->registerTextureResource(framebufferColorAttaches[1], desc);
			desc.format = RHI::TextureFormat::DEPTH32_FLOAT;
			desc.usage = (uint32_t)RHI::TextureUsage::DEPTH_ATTACHMENT | (uint32_t)RHI::TextureUsage::TEXTURE_BINDING;
			desc.viewFormats = { RHI::TextureFormat::DEPTH32_FLOAT };
			GFX::GFXManager::get()->registerTextureResource(framebufferDepthAttach, desc);

			imguiTextureFB[0] = imguiLayer->createImGuiTexture(sampler.get(), 
				Core::ResourceManager::get()->getResource<GFX::Texture>(framebufferColorAttaches[0])->originalView.get(), 
				RHI::TextureLayout::SHADER_READ_ONLY_OPTIMAL);
			imguiTextureFB[1] = imguiLayer->createImGuiTexture(sampler.get(), 
				Core::ResourceManager::get()->getResource<GFX::Texture>(framebufferColorAttaches[1])->originalView.get(), 
				RHI::TextureLayout::SHADER_READ_ONLY_OPTIMAL);

		}
		Buffer vert, frag;
		syncReadFile("../Engine/Binaries/Runtime/spirv/Common/test_shader_vert.spv", vert);
		syncReadFile("../Engine/Binaries/Runtime/spirv/Common/test_shader_frag.spv", frag);
		vert_module = device->createShaderModule({ &vert, RHI::ShaderStages::VERTEX });
		frag_module = device->createShaderModule({ &frag, RHI::ShaderStages::FRAGMENT });

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
			bindGroupLayout = device->createBindGroupLayout(
				RHI::BindGroupLayoutDescriptor{ {
					RHI::BindGroupLayoutEntry{0, (uint32_t)RHI::ShaderStages::VERTEX, RHI::BufferBindingLayout{RHI::BufferBindingType::UNIFORM}}
					} }
			);

			bindGroupLayout_RT = device->createBindGroupLayout(
				RHI::BindGroupLayoutDescriptor{ {
					RHI::BindGroupLayoutEntry{0, (uint32_t)RHI::ShaderStages::RAYGEN, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, RHI::AccelerationStructureBindingLayout{}},
					RHI::BindGroupLayoutEntry{1, (uint32_t)RHI::ShaderStages::RAYGEN, std::nullopt, std::nullopt, std::nullopt, RHI::StorageTextureBindingLayout{}},
					} }
			);

			for (int i = 0; i < 2; ++i) {
				bindGroup[i] = device->createBindGroup(RHI::BindGroupDescriptor{
					bindGroupLayout.get(),
					std::vector<RHI::BindGroupEntry>{
						{0,RHI::BindingResource{RHI::BindingResourceType::BUFFER_BINDING, nullptr, nullptr, nullptr, RHI::BufferBinding{uniformBuffer[i].get(), 0, uniformBuffer[i]->size()}}}
				} });
				//bindGroup_RT[i] = device->createBindGroup(RHI::BindGroupDescriptor{
				//	bindGroupLayout_RT.get(),
				//	std::vector<RHI::BindGroupEntry>{
				//		{0,RHI::BindingResource{RHI::BindingResourceType::BUFFER_BINDING, nullptr, nullptr, nullptr, RHI::BufferBinding{uniformBuffer[i].get(), 0, uniformBuffer[i]->size()}}}
				//} });
			}
		}

		for (int i = 0; i < 2; ++i) {
			pipelineLayout[i] = device->createPipelineLayout(RHI::PipelineLayoutDescriptor{
				{ bindGroupLayout.get() }
				});

			renderPipeline[i] = device->createRenderPipeline(RHI::RenderPipelineDescriptor{
				pipelineLayout[i].get(),
				RHI::VertexState{
						// vertex shader
						vert_module.get(), "main",
						// vertex attribute layout
						{ RHI::VertexBufferLayout{sizeof(Vertex), RHI::VertexStepMode::VERTEX, {
							{ RHI::VertexFormat::FLOAT32X2, 0, 0},
							{ RHI::VertexFormat::FLOAT32X3, offsetof(Vertex,color), 1},}}}},
					RHI::PrimitiveState{ RHI::PrimitiveTopology::TRIANGLE_LIST, RHI::IndexFormat::UINT16_t },
					RHI::DepthStencilState{ },
					RHI::MultisampleState{},
					RHI::FragmentState{
						// fragment shader
						frag_module.get(), "main",
						{{RHI::TextureFormat::RGBA8_UNORM}}}
					});
		}
	};

	/** Update the application every loop */
	virtual auto Update(double deltaTime) noexcept -> void override {
		// start new frame
		imguiLayer->startNewFrame();
		// frame start
		RHI::Device* device = rhiLayer->getDevice();
		RHI::SwapChain* swapChain = rhiLayer->getSwapChain();
		RHI::MultiFrameFlights* multiFrameFlights = rhiLayer->getMultiFrameFlights();
		multiFrameFlights->frameStart();

		std::unique_ptr<RHI::CommandEncoder> commandEncoder = device->createCommandEncoder({ multiFrameFlights->getCommandBuffer() });

		RHI::RenderPassDescriptor renderPassDescriptor = {
			{ RHI::RenderPassColorAttachment{
				Core::ResourceManager::get()->getResource<GFX::Texture>
				(framebufferColorAttaches[currentAttachmentIndex])->originalView.get(), 
			nullptr, {0,0,0,1}, RHI::LoadOp::CLEAR }},
		//	RHI::RenderPassDepthStencilAttachment{
		//		Core::ResourceManager::get()->getResource<GFX::Texture>(framebufferDepthAttach)->originalView.get(),
		//		1, RHI::LoadOp::CLEAR, RHI::StoreOp::DONT_CARE, false,
		//		0, RHI::LoadOp::CLEAR, RHI::StoreOp::DONT_CARE, false
		//},
		};

		uint32_t index = multiFrameFlights->getFlightIndex();

		int width, height;
		mainWindow->getFramebufferSize(&width, &height);
		width = 720;
		height = 480;

		UniformBufferObject ubo;
		ubo.model = Math::transpose(Math::rotate(timer.totalTime() * 80, Math::vec3(0,1,0)).m);
		ubo.view = Math::transpose(Math::lookAt(Math::vec3(0, 0, -2), Math::vec3(0, 0, 0), Math::vec3(0, 1, 0)).m);
		ubo.proj = Math::transpose(Math::perspective(45.f, 1.f * 720 / 480, 0.1f, 10.f).m);
		//ubo.proj.data[1][1] *= -1;
		std::cout << 1.f / timer.deltaTime() << std::endl;
		//Math::rotate( )
		std::future<bool> mapped = uniformBuffer[index]->mapAsync(0, 0, sizeof(UniformBufferObject));
		if (mapped.get()) {
			void* data = uniformBuffer[index]->getMappedRange(0, sizeof(UniformBufferObject));
			memcpy(data, &ubo, sizeof(UniformBufferObject));
			uniformBuffer[index]->unmap();
		}

		commandEncoder->pipelineBarrier(RHI::BarrierDescriptor{
			(uint32_t)RHI::PipelineStages::COLOR_ATTACHMENT_OUTPUT_BIT,
			(uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT,
			(uint32_t)RHI::DependencyType::NONE,
			{}, {},
			{ RHI::TextureMemoryBarrierDescriptor{
				Core::ResourceManager::get()->getResource<GFX::Texture>(framebufferColorAttaches[currentAttachmentIndex])->texture.get(),
				RHI::ImageSubresourceRange{(uint32_t)RHI::TextureAspect::COLOR_BIT, 0,1,0,1},
				(uint32_t)RHI::AccessFlagBits::COLOR_ATTACHMENT_WRITE_BIT,
				(uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT,
				RHI::TextureLayout::SHADER_READ_ONLY_OPTIMAL,
				RHI::TextureLayout::COLOR_ATTACHMENT_OPTIMAL
			}}
		});

		//commandEncoder->pipelineBarrier(RHI::BarrierDescriptor{
		//	(uint32_t)RHI::PipelineStages::COLOR_ATTACHMENT_OUTPUT_BIT,
		//	(uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT,
		//	(uint32_t)RHI::DependencyType::NONE,
		//	{}, {},
		//	{ RHI::TextureMemoryBarrierDescriptor{
		//		Core::ResourceManager::get()->getResource<GFX::Texture>(framebufferDepthAttach)->texture.get(),
		//		RHI::ImageSubresourceRange{(uint32_t)RHI::TextureAspect::DEPTH_BIT, 0,1,0,1},
		//		(uint32_t)RHI::AccessFlagBits::DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
		//		(uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT,
		//		RHI::TextureLayout::SHADER_READ_ONLY_OPTIMAL,
		//		RHI::TextureLayout::DEPTH_ATTACHMENT_OPTIMAL
		//	}}
		//});

		passEncoder[index] = commandEncoder->beginRenderPass(renderPassDescriptor);
		passEncoder[index]->setPipeline(renderPipeline[index].get());
		passEncoder[index]->setViewport(0, 0, width, height, 0, 1);
		passEncoder[index]->setScissorRect(0, 0, width, height);
		passEncoder[index]->setVertexBuffer(0, ResourceManager::get()->getResource<GFX::Mesh>(0)->vertexBuffer.get(), 
			0, ResourceManager::get()->getResource<GFX::Mesh>(0)->vertexBuffer->size());
		passEncoder[index]->setIndexBuffer(ResourceManager::get()->getResource<GFX::Mesh>(0)->indexBuffer.get(), 
			RHI::IndexFormat::UINT16_t, 0, ResourceManager::get()->getResource<GFX::Mesh>(0)->indexBuffer->size());
		passEncoder[index]->setBindGroup(0, bindGroup[index].get(), 0, 0);
		passEncoder[index]->drawIndexed(6, 1, 0, 0, 0);
		passEncoder[index]->end();

		device->getGraphicsQueue()->submit({ commandEncoder->finish({}) }, 
			multiFrameFlights->getImageAvailableSeamaphore(),
			multiFrameFlights->getRenderFinishedSeamaphore(),
			multiFrameFlights->getFence());

		// GUI Recording
		imguiLayer->startGuiRecording();
		bool show_demo_window = true;
		ImGui::ShowDemoWindow(&show_demo_window);
		ImGui::Begin("Hello");
		ImGui::Image(
			imguiTextureFB[currentAttachmentIndex]->getTextureID(),
			{ (float)width,(float)height },
			{ 0,0 }, { 1, 1 });
		ImGui::End();
		editorLayer->onDrawGui();
		imguiLayer->render();

		currentAttachmentIndex = (currentAttachmentIndex + 1) % 2;

		multiFrameFlights->frameEnd();
	};
	
	/** Update the application every fixed update timestep */
	virtual auto FixedUpdate() noexcept -> void override {

	};

	virtual auto Exit() noexcept -> void override {
		rhiLayer->getDevice()->waitIdle();
		renderPipeline[0] = nullptr;
		renderPipeline[1] = nullptr;
		passEncoder[0] = nullptr;
		passEncoder[1] = nullptr;
		pipelineLayout[0] = nullptr;
		pipelineLayout[1] = nullptr;

		bindGroupLayout_RT = nullptr;
		bindGroup_RT[0] = nullptr;
		bindGroup_RT[1] = nullptr;

		vertexBufferRT = nullptr;
		indexBufferRT = nullptr;
		vert_module = nullptr;
		frag_module = nullptr;
		tlas = nullptr;
		blas = nullptr;

		bindGroupLayout = nullptr;
		uniformBuffer[0] = nullptr;
		uniformBuffer[1] = nullptr;
		bindGroup[0] = nullptr;
		bindGroup[1] = nullptr;

		sampler = nullptr;

		editorLayer = nullptr;
		imguiLayer = nullptr;

		ResourceManager::get()->clear();

		rhiLayer = nullptr;
	}

private:
	uint32_t currentAttachmentIndex = 0;
	std::array<Core::GUID, 2> framebufferColorAttaches;
	Core::GUID framebufferDepthAttach;

	std::unique_ptr<RHI::RHILayer> rhiLayer = nullptr;
	std::unique_ptr<Editor::ImGuiLayer> imguiLayer = nullptr;
	std::unique_ptr<Editor::EditorLayer> editorLayer = nullptr;

	std::unique_ptr<RHI::Sampler> sampler = nullptr;
	std::unique_ptr<Editor::ImGuiTexture> imguiTexture = nullptr;
	std::unique_ptr<Editor::ImGuiTexture> imguiTextureFB[2];

	std::unique_ptr<RHI::BindGroupLayout> bindGroupLayout = nullptr;
	std::unique_ptr<RHI::BindGroup> bindGroup[2];

	std::unique_ptr<RHI::BindGroupLayout> bindGroupLayout_RT = nullptr;
	std::unique_ptr<RHI::BindGroup> bindGroup_RT[2];

	std::unique_ptr<RHI::Buffer> uniformBuffer[2];
	std::unique_ptr<RHI::PipelineLayout> pipelineLayout[2];

	std::unique_ptr<RHI::Buffer> vertexBufferRT = nullptr;
	std::unique_ptr<RHI::Buffer> indexBufferRT = nullptr;
	std::unique_ptr<RHI::ShaderModule> vert_module = nullptr;
	std::unique_ptr<RHI::ShaderModule> frag_module = nullptr;
	std::unique_ptr<RHI::RenderPassEncoder> passEncoder[2] = {};
	std::unique_ptr<RHI::BLAS> blas = nullptr;
	std::unique_ptr<RHI::TLAS> tlas = nullptr;

	std::unique_ptr<RHI::RenderPipeline> renderPipeline[2];
	// the embedded scene, which should be removed in the future
	GFX::Scene scene;
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
			720, 480,
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