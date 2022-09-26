#include <iostream>
#include <chrono>
#include <format>
#include <functional>
#include <filesystem>
#include <chrono>
#include <memory>
#include <glad/glad.h>
import Core.Log;
import Core.Memory;
import Core.IO;
import Core.Event;
import Core.Timer;
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

import Application.Root;
import Application.Base;

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
		bool initialized = context.init(mainWindow.get(), (uint32_t)RHI::ContextExtension::NONE);
		adapter = context.requestAdapter({});
		device = adapter->requestDevice();
		swapChain = device->createSwapChain({});
		multiFrameFlights = device->createMultiFrameFlights({ 2, swapChain.get() });
		mainWindow->connectResizeEvent([&](size_t w, size_t h)->void {swapChain->recreate(); });

		struct Vertex {
			Math::vec2 pos;
			Math::vec3 color;
		};

		std::vector<Vertex> const vertices = {
			{{-0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}},
			{{0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}},
			{{0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}},
			{{-0.5f, 0.5f}, {1.0f, 1.0f, 1.0f}}
		};

		{
			RHI::BufferDescriptor vertexDescriptor;
			vertexDescriptor.size = sizeof(vertices[0]) * vertices.size();
			vertexDescriptor.usage = (uint32_t)RHI::BufferUsage::VERTEX | (uint32_t)RHI::BufferUsage::COPY_DST;
			vertexDescriptor.memoryProperties = (uint32_t)RHI::MemoryProperty::DEVICE_LOCAL_BIT;
			vertexDescriptor.mappedAtCreation = true;
			vertexBuffer = device->createBuffer(vertexDescriptor);

			RHI::BufferDescriptor stagingBufferDescriptor;
			stagingBufferDescriptor.size = sizeof(vertices[0]) * vertices.size();
			stagingBufferDescriptor.usage = (uint32_t)RHI::BufferUsage::COPY_SRC;
			stagingBufferDescriptor.memoryProperties = (uint32_t)RHI::MemoryProperty::HOST_VISIBLE_BIT
				| (uint32_t)RHI::MemoryProperty::HOST_COHERENT_BIT;
			stagingBufferDescriptor.mappedAtCreation = true;
			std::unique_ptr<RHI::Buffer> stagingBuffer = device->createBuffer(stagingBufferDescriptor);
			std::future<bool> mapped = stagingBuffer->mapAsync(0, 0, vertexDescriptor.size);
			if (mapped.get()) {
				void* data = stagingBuffer->getMappedRange(0, vertexDescriptor.size);
				memcpy(data, vertices.data(), (size_t)vertexDescriptor.size);
				stagingBuffer->unmap();
			}
			std::unique_ptr<RHI::CommandEncoder> commandEncoder = device->createCommandEncoder({ nullptr });
			commandEncoder->copyBufferToBuffer(stagingBuffer.get(), 0, vertexBuffer.get(), 0, vertexDescriptor.size);
			device->getGraphicsQueue()->submit({ commandEncoder->finish({}) });
			device->getGraphicsQueue()->waitIdle();
		}

		std::vector<uint16_t> const indices = {
			0, 1, 2, 2, 3, 0
		};

		{
			RHI::BufferDescriptor indexDescriptor;
			indexDescriptor.size = sizeof(indices[0]) * indices.size();
			indexDescriptor.usage = (uint32_t)RHI::BufferUsage::INDEX | (uint32_t)RHI::BufferUsage::COPY_DST;
			indexDescriptor.memoryProperties = (uint32_t)RHI::MemoryProperty::DEVICE_LOCAL_BIT;
			indexDescriptor.mappedAtCreation = true;
			indexBuffer = device->createBuffer(indexDescriptor);

			RHI::BufferDescriptor stagingBufferDescriptor;
			stagingBufferDescriptor.size = sizeof(indices[0]) * indices.size();
			stagingBufferDescriptor.usage = (uint32_t)RHI::BufferUsage::COPY_SRC;
			stagingBufferDescriptor.memoryProperties = (uint32_t)RHI::MemoryProperty::HOST_VISIBLE_BIT
				| (uint32_t)RHI::MemoryProperty::HOST_COHERENT_BIT;
			stagingBufferDescriptor.mappedAtCreation = true;
			std::unique_ptr<RHI::Buffer> stagingBuffer = device->createBuffer(stagingBufferDescriptor);
			std::future<bool> mapped = stagingBuffer->mapAsync(0, 0, indexDescriptor.size);
			if (mapped.get()) {
				void* data = stagingBuffer->getMappedRange(0, indexDescriptor.size);
				memcpy(data, indices.data(), (size_t)indexDescriptor.size);
				stagingBuffer->unmap();
			}
			std::unique_ptr<RHI::CommandEncoder> commandEncoder = device->createCommandEncoder({ nullptr });
			commandEncoder->copyBufferToBuffer(stagingBuffer.get(), 0, indexBuffer.get(), 0, indexDescriptor.size);
			device->getGraphicsQueue()->submit({ commandEncoder->finish({}) });
			device->getGraphicsQueue()->waitIdle();
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

			for (int i = 0; i < 2; ++i) {
				bindGroup[i] = device->createBindGroup(RHI::BindGroupDescriptor{
					bindGroupLayout.get(),
					std::vector<RHI::BindGroupEntry>{
						{0,RHI::BindingResource{RHI::BindingResourceType::BUFFER_BINDING, nullptr, nullptr, nullptr, RHI::BufferBinding{uniformBuffer[i].get(), 0, uniformBuffer[i]->size()}}}
				} });
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
					RHI::DepthStencilState{},
					RHI::MultisampleState{},
					RHI::FragmentState{
						// fragment shader
						frag_module.get(), "main",
						{{RHI::TextureFormat::RGBA8_UINT}}}
					});
		}
	};

	/** Update the application every loop */
	virtual auto Update(double deltaTime) noexcept -> void override {
		multiFrameFlights->frameStart();
		std::unique_ptr<RHI::CommandEncoder> commandEncoder = device->createCommandEncoder({ multiFrameFlights->getCommandBuffer() });

		RHI::RenderPassDescriptor renderPassDescriptor = {
			{ RHI::RenderPassColorAttachment{ swapChain->getTextureView(multiFrameFlights->getSwapchainIndex()), nullptr, {0,0,0,1}, RHI::LoadOp::CLEAR }},
			RHI::RenderPassDepthStencilAttachment{},
		};

		uint32_t index = multiFrameFlights->getFlightIndex();

		int width, height;
		mainWindow->getFramebufferSize(&width, &height);

		UniformBufferObject ubo;
		ubo.model = Math::transpose(Math::rotate(timer.totalTime() * 80, Math::vec3(0,1,0)).m);
		ubo.view = Math::transpose(Math::lookAt(Math::vec3(0, 0, -2), Math::vec3(0, 0, 0), Math::vec3(0, 1, 0)).m);
		ubo.proj = Math::transpose(Math::perspective(45.f, 1.f * width / height, 0.1f, 10.f).m);
		//ubo.proj.data[1][1] *= -1;
		std::cout << 1.f / timer.deltaTime() << std::endl;
		//Math::rotate( )
		std::future<bool> mapped = uniformBuffer[index]->mapAsync(0, 0, sizeof(UniformBufferObject));
		if (mapped.get()) {
			void* data = uniformBuffer[index]->getMappedRange(0, sizeof(UniformBufferObject));
			memcpy(data, &ubo, sizeof(UniformBufferObject));
			uniformBuffer[index]->unmap();
		}

		passEncoder[index] = commandEncoder->beginRenderPass(renderPassDescriptor);
		passEncoder[index]->setPipeline(renderPipeline[index].get());
		passEncoder[index]->setViewport(0, 0, width, height, 0, 1);
		passEncoder[index]->setScissorRect(0, 0, width, height);
		passEncoder[index]->setVertexBuffer(0, vertexBuffer.get(), 0, vertexBuffer->size());
		passEncoder[index]->setIndexBuffer(indexBuffer.get(), RHI::IndexFormat::UINT16_t, 0, indexBuffer->size());
		passEncoder[index]->setBindGroup(0, bindGroup[index].get(), 0, 0);
		passEncoder[index]->drawIndexed(6, 1, 0, 0, 0);
		passEncoder[index]->end();

		device->getGraphicsQueue()->submit({ commandEncoder->finish({}) }, 
			multiFrameFlights->getImageAvailableSeamaphore(),
			multiFrameFlights->getRenderFinishedSeamaphore(),
			multiFrameFlights->getFence());

		multiFrameFlights->frameEnd();
	};
	
	/** Update the application every fixed update timestep */
	virtual auto FixedUpdate() noexcept -> void override {

	};

	virtual auto Exit() noexcept -> void override {
		device->waitIdle();
		renderPipeline[0] = nullptr;
		renderPipeline[1] = nullptr;
		passEncoder[0] = nullptr;
		passEncoder[1] = nullptr;
		pipelineLayout[0] = nullptr;
		pipelineLayout[1] = nullptr;

		vertexBuffer = nullptr;
		indexBuffer = nullptr;
		vert_module = nullptr;
		frag_module = nullptr;

		bindGroupLayout = nullptr;
		uniformBuffer[0] = nullptr;
		uniformBuffer[1] = nullptr;
		bindGroup[0] = nullptr;
		bindGroup[1] = nullptr;

		multiFrameFlights = nullptr;
		swapChain = nullptr;
		device = nullptr;
	}

private:
	RHI::Context_VK context;
	std::unique_ptr<RHI::Adapter> adapter = nullptr;
	std::unique_ptr<RHI::Device> device = nullptr;
	std::unique_ptr<RHI::SwapChain> swapChain = nullptr;
	std::unique_ptr<RHI::MultiFrameFlights> multiFrameFlights = nullptr;

	std::unique_ptr<RHI::BindGroupLayout> bindGroupLayout = nullptr;
	std::unique_ptr<RHI::Buffer> uniformBuffer[2];
	std::unique_ptr<RHI::BindGroup> bindGroup[2];
	std::unique_ptr<RHI::PipelineLayout> pipelineLayout[2];

	std::unique_ptr<RHI::Buffer> vertexBuffer = nullptr;
	std::unique_ptr<RHI::Buffer> indexBuffer = nullptr;
	std::unique_ptr<RHI::ShaderModule> vert_module = nullptr;
	std::unique_ptr<RHI::ShaderModule> frag_module = nullptr;
	std::unique_ptr<RHI::RenderPassEncoder> passEncoder[2] = {};

	std::unique_ptr<RHI::RenderPipeline> renderPipeline[2];

};

int main()
{
	Application::Root root;
	
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
	//	Image::COLOR_R8G8B8_UINT{255,255,0});

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