#include <iostream>
#include <chrono>
#include <format>
#include <filesystem>
import Core.Log;
import Core.Memory;
import Core.IO;
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

using namespace SIByL;
using namespace SIByL::Core;
using namespace SIByL::Math;

int main()
{
	Application::Root root;

	Image::Image<Image::COLOR_R8G8B8_UINT> image(4, 4);
	image[0][0] = Image::COLOR_R8G8B8_UINT{200,200,200};
	Buffer buffer = Image::PPM::toPPM(image);
	syncWriteFile("./test.ppm", buffer);
}