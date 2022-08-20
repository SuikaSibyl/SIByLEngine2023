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

		auto motionBounds(bounds3 const& b) const noexcept -> bounds3;

		auto boundPointMotion(point3 const& p) const noexcept -> bounds3;

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