module;
#include <cstdint>
export module Math.Geometry:Ray3;
import Math.Vector;
import :Point3;
import :Normal3;

namespace SIByL::Math
{
	export struct ray3
	{
		point3 o;
		vec3 d;
	};
}