#include <iostream>
#include <chrono>
#include <format>
#include <functional>
#include <filesystem>
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

import Application.Root;
import Image.Color;
import Image.Image;
import Image.FileFormat;

import Platform.Window;

using namespace SIByL;
using namespace SIByL::Core;
using namespace SIByL::Math;

int main()
{
	Application::Root root;

	Image::Image<Image::COLOR_R8G8B8_UINT> image(720, 480);
	std::fill((Image::COLOR_R8G8B8_UINT*) & ((reinterpret_cast<char*>(image.data.data))[0]), 
		(Image::COLOR_R8G8B8_UINT*)&((reinterpret_cast<char*>(image.data.data))[image.data.size]), 
		Image::COLOR_R8G8B8_UINT{255,255,0});

	Platform::Window_Win64 window(L"Hello World");
	window.create();
	auto paintbitmap = std::bind(Platform::paintRGB8Bitmap,
		std::placeholders::_1, size_t(720), size_t(480), (char*)image.data.data);
	window.onPaintSignal.connect(paintbitmap);
	window.resize(720, 480);

	int i = 0;
	while (window.isRunning()) {
		std::fill((Image::COLOR_R8G8B8_UINT*)&((reinterpret_cast<char*>(image.data.data))[0]),
			(Image::COLOR_R8G8B8_UINT*)&((reinterpret_cast<char*>(image.data.data))[image.data.size]),
			Image::COLOR_R8G8B8_UINT{ 255,255,(uint8_t)((i++) % 256)});
		window.run();
		window.invalid();
	}
	window.destroy();

}