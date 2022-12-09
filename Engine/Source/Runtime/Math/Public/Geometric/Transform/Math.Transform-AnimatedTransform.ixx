module;
#include <cmath>
#include <cstdint>
export module SE.Math.Geometric:AnimatedTransform;
import :Vector3;
import :Ray3;
import :Point3;
import :Bounds3;
import :Normal3;
import :Matrix4x4;
import :Quaternion;
import :Transform;
import SE.Math.Misc;

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