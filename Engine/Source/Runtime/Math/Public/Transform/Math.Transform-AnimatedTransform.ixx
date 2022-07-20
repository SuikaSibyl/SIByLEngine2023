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
		AnimatedTransform(
			Transform const* startTransform, float startTime,
			Transform const* endTransform, float endTime);

		auto interpolate(float time, Transform* t) const -> void;

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