module;
#include <cmath>
#include <cstdint>
export module Math.Transform:AnimatedTransform;
import :Transform;
import Math.Matrix;
import Math.Vector;
import Math.Geometry;
import Math.Trigonometric;

namespace SIByL::Math
{
	export struct AnimatedTransform
	{
		//auto operator*(point3 const& p) const->point3;
		//auto operator*(vec3 const& v) const->vec3;
		//auto operator*(normal3 const& n) const->normal3;
		//auto operator*(ray3 const& s) const->ray3;
		//auto operator*(bounds3 const& b) const->bounds3;
		//auto operator*(Transform const& t2) const->Transform;

		Transform const* startTransform;
		Transform const* endTransform;
		float const startTime, endTime;
		bool const actuallyAnimated;
		Math::vec3 t[2];
		Math::Quaternion r[2];
		Math::mat4 s[2];
		bool hasRotation;
	};
}