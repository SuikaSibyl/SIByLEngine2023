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
import Math.Vector;
import Math.Geometry;
import Math.Matrix;
import Math.Limits;
import Math.Transform;

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

struct SandBoxApplication :public Application::ApplicationBase {
	/** Initialize the application */
	virtual auto Init() noexcept -> void override {
		//glViewport(0, 0, 720, 480);

	};

	/** Update the application every loop */
	virtual auto Update() noexcept -> void override {
		//glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
		//glClear(GL_COLOR_BUFFER_BIT);
	};

	/** Update the application every fixed update timestep */
	virtual auto FixedUpdate() noexcept -> void override {

	};
};

int main()
{
	Application::Root root;
	
	//SandBoxApplication app;
	//app.createMainWindow({
	//		Platform::WindowVendor::GLFW,
	//		L"SIByL Engine 2022.0",
	//		720, 480,
	//		Platform::WindowProperties::VULKAN_CONTEX
	//	});

	//RHI::Context_VK context;
	//bool initialized = context.init(app.mainWindow.get(), (uint32_t)RHI::ContextExtension::NONE);
	//std::unique_ptr<RHI::Adapter> adapter = context.requestAdapter(RHI::RequestAdapterOptions{});
	//std::unique_ptr<RHI::Device> device = adapter->requestDevice();
	//std::unique_ptr<RHI::SwapChain> swapChain = device->createSwapChain({});

	//app.run();

	Scope<Platform::Window> window = Platform::Window::create({
			Platform::WindowVendor::WIN_64,
			L"Ray Trace Viewer",
			720, 480,
			Platform::WindowProperties::OPENGL_CONTEX
		});

	int ncores = Platform::getNumSystemCores();
	Image::Image<Image::COLOR_R8G8B8_UINT> image(720, 480);
	std::fill((Image::COLOR_R8G8B8_UINT*) & ((reinterpret_cast<char*>(image.data.data))[0]), 
		(Image::COLOR_R8G8B8_UINT*)&((reinterpret_cast<char*>(image.data.data))[image.data.size]), 
		Image::COLOR_R8G8B8_UINT{255,255,0});

	window->bindPaintingBitmapRGB8(size_t(720), size_t(480), (char*)image.data.data);
	window->resize(720.f, 480);

	Math::Transform defaultTransform(mat4{});
	Math::AnimatedTransform animatedDefaultTransform(&defaultTransform, 0, &defaultTransform, 0);
	Tracer::Film film(Math::ipoint2{ 720, 480 }, Math::bounds2{ {0,0}, {1,1} }, 
		std::make_unique<Tracer::BoxFilter>(Math::vec2{1.f,1.f}), 1, "what.c", 1);
	Tracer::OrthographicCamera camera(animatedDefaultTransform, 
		Math::bounds2{ {-1.f * 720.f / 480.f,-1.f}, {1.f * 720.f / 480.f, 1.f} } , 
		0, 0, 0, 0, &film, nullptr);

	float const radius = 0.5f;
	float const radius_2 = 3.f;
	Math::Transform objectToWorld = Math::translate({ 0,0,radius_2 });
	Math::Transform worldToObject = Math::translate({ 0,0,-radius_2 });
	Tracer::Sphere sphere(&objectToWorld, &worldToObject, false, radius, -radius, radius, 360);
	Tracer::MatteMaterial material(nullptr, nullptr, nullptr);
	Tracer::GeometricPrimitive primitve;
	primitve.shape = &sphere;
	primitve.material = &material;

	Math::Transform objectToWorld_2 = Math::translate({ 0,radius_2+ radius,radius_2 });
	Math::Transform worldToObject_2 = Math::translate({ 0,-radius_2- radius,-radius_2 });

	Tracer::Sphere sphere_2(&objectToWorld_2, &worldToObject_2, false, radius_2, -radius_2, radius_2, 360);
	Tracer::GeometricPrimitive ground;
	ground.shape = &sphere_2;
	ground.material = &material;
	Tracer::DummyAggregate aggregate{ std::vector<Tracer::Primitive*>{&primitve,& ground} };
	Tracer::InfiniteAreaLight areaLight(Math::Transform{}, Tracer::Spectrum{ 1.f }, 1, "");
	Tracer::Scene scene(&aggregate, { &areaLight });
	Tracer::StratifiedSampler sampler(10, 10, true, 50);
	Tracer::WhittedIntegrator integrator(5, &camera, &sampler);

	std::thread render(std::bind(&Tracer::WhittedIntegrator::render, &integrator, scene));

	int i = 0;
	while (window->isRunning()) {
		auto startPoint = std::chrono::high_resolution_clock::now();
		camera.film->writeImage(image, 1.f);
		//// Clear background to black
		//std::fill((Image::COLOR_R8G8B8_UINT*)&((reinterpret_cast<char*>(image.data.data))[0]),
		//	(Image::COLOR_R8G8B8_UINT*)&((reinterpret_cast<char*>(image.data.data))[image.data.size]),
		//	Image::COLOR_R8G8B8_UINT{ 0 ,0 ,0 });

		//float tHit;
		//Math::ivec2 sampleExtent{ 720,480 };
		//int const tileSize = 16;
		//Math::ipoint2 nTiles((sampleExtent.x + tileSize - 1) / tileSize,
		//	(sampleExtent.y + tileSize - 1) / tileSize);

		//Parallelism::ParallelFor2D([&](Math::ipoint2 tile)->void {
		//	Tracer::Ray ray;
		//	for (int i = 0; i < tileSize; ++i)
		//		for (int j = 0; j < tileSize; ++j) {
		//			image[tile.y * tileSize + j][tile.x * tileSize + i] = Image::COLOR_R8G8B8_UINT{  0,0 ,255 };
		//		}
		//}, nTiles);

		window->fetchEvents();
		window->invalid();

		auto endPoint = std::chrono::high_resolution_clock::now();
		long long start = std::chrono::time_point_cast<std::chrono::microseconds>(startPoint).time_since_epoch().count();
		long long end = std::chrono::time_point_cast<std::chrono::microseconds>(endPoint).time_since_epoch().count();
		long long time = end - start;
		std::cout << "Time each frame: " << (time * 1. / 1000000) << std::endl;
	}
	render.join();
	window->destroy();
	Parallelism::clearThreadPool();
}