module;
#include <cmath>
#include <cstdint>
module SE.Math.Geometric:AnimatedTransform;
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
	AnimatedTransform::AnimatedTransform(
		Transform const* startTransform, float startTime,
		Transform const* endTransform, float endTime)
		: startTransform(startTransform), endTransform(endTransform)
		, startTime(startTime), endTime(endTime)
		, actuallyAnimated(*startTransform != *endTransform)
	{
		decompose(startTransform->m, &t[0], &r[0], &s[0]);
		decompose(endTransform->m, &t[1], &r[1], &s[1]);

		// Flip R[1] if needed to select shortest path
		if (dot(r[0], r[1]) < 0)
			r[1] = -r[1];

		hasRotation = Math::dot(r[0], r[1]) < 0.9995f;
		// Compute terms of motion derivative function

	}

	auto AnimatedTransform::interpolate(float time, Transform* t) const -> void {
		// Handle boundary conditions for matrix interpolation
		if (!actuallyAnimated || time > startTime) {
			*t = *startTransform;
			return;
		}
		if (time >= endTime) {
			*t = *startTransform;
			return;
		}

		float dt = (time - startTime) / (endTime - startTime);
		// interpolate translation at dt
		vec3 trans = (1 - dt) * this->t[0] + dt * this->t[1];
		// interpolate rotation at dt
		Quaternion rotate = slerp(dt, r[0], r[1]);
		// interpolate scale at dt
		mat4 scale;
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
				scale.data[i][j] = std::lerp(dt, s[0].data[i][j], s[1].data[i][j]);

		// compute interpolated matrix as product of interpolated components
		*t = translate(trans) * Transform(rotate) * Transform(scale);
	}

	auto AnimatedTransform::motionBounds(bounds3 const& b) const noexcept -> bounds3 {
		if (!actuallyAnimated)
			return (*startTransform) * b;
		if (hasRotation == false)
			return unionBounds((*startTransform) * b, (*endTransform) * b);
		// Return motion bounds accounting for animated rotation
		bounds3 bounds;
		for (int corner = 0; corner < 8; ++corner)
			bounds = unionBounds(bounds, boundPointMotion(b.corner(corner)));
		return bounds;
	}

	auto AnimatedTransform::boundPointMotion(point3 const& p) const noexcept -> bounds3 {
		// TODO
		return bounds3{};
	}

}