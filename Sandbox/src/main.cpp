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
	Image::Image<Image::COLOR_R32G32B32_FLOAT> image(720, 360);
	Buffer buffer = Image::PFM::toPFM(image);
	syncWriteFile("./test.pfm", buffer);
}