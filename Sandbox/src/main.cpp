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
import Tracer.Interactable;

import Image.Color;
import Image.Image;
import Image.FileFormat;

import Platform.Window;
import Platform.System;
import Parallelism.Parallel;

import RHI.Device;

import Application.Root;
import Application.Base;

using namespace SIByL;
using namespace SIByL::Core;
using namespace SIByL::Math;

struct SandBoxApplication :public Application::ApplicationBase {
	/** Initialize the application */
	virtual auto Init() noexcept -> void override {
		glViewport(0, 0, 720, 480);

	};

	/** Update the application every loop */
	virtual auto Update() noexcept -> void override {
		glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);
	};

	/** Update the application every fixed update timestep */
	virtual auto FixedUpdate() noexcept -> void override {

	};
};

int main()
{
	Application::Root root;
	
	SandBoxApplication app;
	Scope<Platform::Window> window = Platform::Window::create({
			Platform::WindowVendor::WIN_64,
			L"SIByL Engine 2022.0",
			720, 480,
			Platform::WindowProperties::OPENGL_CONTEX
		});
	//app.createMainWindow({
	//		Platform::WindowVendor::WIN_64,
	//		L"SIByL Engine 2022.0",
	//		720, 480,
	//		Platform::WindowProperties::OPENGL_CONTEX
	//	});

	RHI::Context_GL context;
	bool win = context.init(app.mainWindow.get());

	//app.run();

	int ncores = Platform::getNumSystemCores();
	Image::Image<Image::COLOR_R8G8B8_UINT> image(720, 480);
	std::fill((Image::COLOR_R8G8B8_UINT*) & ((reinterpret_cast<char*>(image.data.data))[0]), 
		(Image::COLOR_R8G8B8_UINT*)&((reinterpret_cast<char*>(image.data.data))[image.data.size]), 
		Image::COLOR_R8G8B8_UINT{255,255,0});

	window->bindPaintingBitmapRGB8(size_t(720), size_t(480), (char*)image.data.data);
	window->resize(720.f, 480);

	Math::Transform defaultTransform(mat4{});
	Math::AnimatedTransform animatedDefaultTransform(&defaultTransform, 0, &defaultTransform, 0);
	Tracer::Film film(Math::ipoint2{ 720, 480 }, Math::bounds2{ {0,0}, {1,1} }, std::make_unique<Tracer::BoxFilter>(Math::vec2{1.f,1.f}), 1, "what.c", 1);
	Tracer::OrthographicCamera camera(animatedDefaultTransform, Math::bounds2{ {-1.f * 720.f / 480.f,-1.f}, {1.f * 720.f / 480.f, 1.f} } , 0, 0, 0, 0, &film, nullptr);

	Math::Transform objectToWorld = Math::translate({ 0,0,5 });
	Math::Transform worldToObject = Math::translate({ 0,0,-5 });

	Tracer::Sphere sphere(&objectToWorld, &worldToObject, false, 1, -1, 1, 360);

	int i = 0;
	while (window->isRunning()) {
		auto startPoint = std::chrono::high_resolution_clock::now();

		// Clear background to black
		std::fill((Image::COLOR_R8G8B8_UINT*)&((reinterpret_cast<char*>(image.data.data))[0]),
			(Image::COLOR_R8G8B8_UINT*)&((reinterpret_cast<char*>(image.data.data))[image.data.size]),
			Image::COLOR_R8G8B8_UINT{ 0 ,0 ,0 });

		float tHit;
		Parallelism::ParallelFor([&](int j)->void {
			Tracer::Ray ray;
			for (int i = 0; i < 720; ++i) {
				Tracer::CameraSample sample = { Math::point2{ 0.5f + i * 1.f, 0.5f + j * 1.f }, Math::point2{ 0.f,0.f }, 0.f };
				camera.generateRay(sample, &ray);
				bool intersected = sphere.intersect(ray, &tHit, nullptr, false);
				if (intersected)
					image[j][i] = Image::COLOR_R8G8B8_UINT{ 255 ,255 ,255 };
			}
		}, 480, 30);

		//for (int j = 0; j < 480; j++) {
		//	for (int i = 0; i < 720; ++i) {
		//		Tracer::CameraSample sample = { Math::point2{ 0.5f + i * 1.f, 0.5f + j * 1.f }, Math::point2{ 0.f,0.f }, 0.f };
		//		camera.generateRay(sample, &ray);
 	//			bool intersected = sphere.intersect(ray, &tHit, nullptr);
		//		if (intersected) 
		//			image[j][i] = Image::COLOR_R8G8B8_UINT{ 255 ,255 ,255 };
		//		if (j == 239 && i == 360)
		//			image[j][i] = Image::COLOR_R8G8B8_UINT{ 0 ,0 ,255 };

		//	}
		//}

		window->fetchEvents();
		window->invalid();

		auto endPoint = std::chrono::high_resolution_clock::now();
		long long start = std::chrono::time_point_cast<std::chrono::microseconds>(startPoint).time_since_epoch().count();
		long long end = std::chrono::time_point_cast<std::chrono::microseconds>(endPoint).time_since_epoch().count();
		long long time = end - start;
		std::cout << "Time each frame: " << (time * 1. / 1000000) << std::endl;
	}
	window->destroy();
	Parallelism::clearThreadPool();
}