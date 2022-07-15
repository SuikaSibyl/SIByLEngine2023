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

using namespace SIByL;
using namespace SIByL::Core;
using namespace SIByL::Math;

int main()
{
	Application::Root root;

	float test_f[4][4] = {
		{2, -1, 0, 0},
		{-1, 2, -1, 0},
		{0, -1, 2, 0},
		{0, 0, 0, 1}
	};

	float id[4][4] = {
		{1, 0, 0, 0},
		{0, 1, 0, 0},
		{0, 0, 1, 0},
		{0, 0, 0, 1}
	};

	size_t size;
	Buffer image_binary;
	syncReadFile("P:/SIByLEngine/Sandbox/cache/42", image_binary);

	int* i = New<int>(1);
	Delete<int>(i);

	Buffer buffer(1000);
	Buffer buffer2;
	buffer2 = std::move(buffer);

	mat4 test(id);
	Quaternion quat1(test);
	mat3 converted = quat1.toMat3();
	Quaternion quat2 = { 1,2,3,4 };
	Quaternion quat = normalize(quat2);
}