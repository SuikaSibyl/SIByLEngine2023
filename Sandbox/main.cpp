#include <se.core.hpp>
#include <seditor-base.hpp>
#include <cnpy.h>
#include <se.image.hpp>

int main() {
	float a = 1.f;
	cnpy::NpyArray test = cnpy::npy_load("C:/Users/suika/Desktop/tmp/test.npy");
	std::span<float> test_array((float*)test.data<float>(), 64 * 64 * 4);
	
	//se::image::EXR::fromEXR("../../Engine/binary/resources/textures/ltc_lut1.exr");
	se::image::EXR::writeEXR("hello.exr", 64, 64, 4, test.data<float>());
}