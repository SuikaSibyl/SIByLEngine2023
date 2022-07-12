module;
#include <cmath>
#include <cstdint>
export module Math.Geometry:Quaternion;
import Math.Vector;
import Math.Matrix;

namespace SIByL::Math
{
	export struct Quaternion
	{
		Quaternion() :v(0), s(1.f) {}
		Quaternion(float x, float y, float z, float w) :v(x, y, z), s(w) {}
		Quaternion(vec3 const& v, float s) :v(v), s(s) {}
		Quaternion(mat3 const& m);
		Quaternion(mat4 const& m);

		auto toMat3() noexcept -> mat3;

		auto lengthSquared() const -> float;
		auto length() const -> float;

		auto conjugate() noexcept -> Quaternion;
		auto reciprocal() noexcept -> Quaternion;

		auto operator/(float s) const-> Quaternion;
		auto operator+(Quaternion const& q2) const -> Quaternion;
		auto operator*(Quaternion const& q2) const -> Quaternion;
		auto operator+=(Quaternion const& q) -> Quaternion&;

		union
		{
			struct
			{
				vec3 v;
				float s;
			};
			float data[4];
			struct { float x, y, z, w; };
		};
	};

	export inline auto dot(Quaternion const& q1, Quaternion const& q2) noexcept -> float
	{
		return dot(q1.v, q2.v) + q1.s * q2.s;
	}

	export inline auto normalize(Quaternion const& q) noexcept -> Quaternion
	{
		return q / std::sqrt(dot(q, q));
	}

	// Quaternion::toMat3()
	//
	// Reference: https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
	// Reference: http://www.iri.upc.edu/files/scidoc/2068-Accurate-Computation-of-Quaternions-from-Rotation-Matrices.pdf
	Quaternion::Quaternion(mat3 const& m)
	{
		// Notice:
		// T = 4 - 4*qx2 - 4*qy2 - 4*qz2
		//   = 4(1 - qx2 - qy2 - qz2)
		//	 = m00 + m11 + m22 + 1

		float trace = m.data[0][0] + m.data[1][1] + m.data[2][2];
		if (trace > 0)
		{
			// case that the function inside the square root is positive
			float s = 0.5f / std::sqrt(trace + 1);
			w = 0.25f / s;
			x = (m.data[2][1] - m.data[1][2]) * s;
			y = (m.data[0][2] - m.data[2][0]) * s;
			z = (m.data[1][0] - m.data[0][1]) * s;
		}
		else if (m.data[0][0] > m.data[1][1] && m.data[0][0] > m.data[2][2])
		{
			// case [0][0] is the largest
			float s = 2.0f * std::sqrt(1.f + m.data[0][0] - m.data[1][1] - m.data[2][2]);
			w = (m.data[2][1] - m.data[1][2]) / s;
			x = 0.25f * s;
			y = (m.data[0][1] + m.data[1][0]) / s;
			z = (m.data[0][2] + m.data[2][0]) / s;

		}
		else if (m.data[1][1] > m.data[2][2])
		{
			// case [1][1] is the largest
			float s = 2.0f * std::sqrt(1.f + m.data[1][1] - m.data[0][0] - m.data[2][2]);
			w = (m.data[0][2] - m.data[2][0]) / s;
			x = (m.data[0][1] + m.data[1][0]) / s;
			y = 0.25f * s;
			z = (m.data[1][2] + m.data[2][1]) / s;
		}
		else
		{
			// case [2][2] is the largest
			float s = 2.0f * std::sqrt(1.f + m.data[2][2] - m.data[0][0] - m.data[1][1]);
			w = (m.data[1][0] - m.data[0][1]) / s;
			x = (m.data[0][2] + m.data[2][0]) / s;
			y = (m.data[1][2] + m.data[2][1]) / s;
			z = 0.25f * s;
		}
	}

	Quaternion::Quaternion(mat4 const& m)
		:Quaternion(mat3(m))
	{}

	auto Quaternion::lengthSquared() const -> float
	{
		return x * x + y * y + z * z + w * w;
	}

	auto Quaternion::length() const -> float
	{
		return std::sqrt(lengthSquared());
	}

	auto Quaternion::toMat3() noexcept -> mat3
	{
		return mat3(
			1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * x * z - y * w,
			2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w),
			w * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y));
	}

	auto Quaternion::conjugate() noexcept -> Quaternion
	{
		return Quaternion(-v, s);
	}

	auto Quaternion::reciprocal() noexcept -> Quaternion
	{
		return conjugate() / lengthSquared();
	}

	auto Quaternion::operator/(float s) const->Quaternion
	{
		Quaternion ret;
		ret.v = v / s;
		ret.s = s / s;
		return ret;
	}

	auto Quaternion::operator+(Quaternion const& q2) const -> Quaternion
	{
		Quaternion ret;
		ret.v = v + q2.v;
		ret.s = s + q2.s;
		return ret;
	}

	auto Quaternion::operator*(Quaternion const& q2) const -> Quaternion
	{
		Quaternion ret;
		ret.v = cross(v, q2.v) + q2.v * s + v * q2.s;
		ret.s = s * q2.s - dot(v, q2.v);
		return ret;
	}

	auto Quaternion::operator+=(Quaternion const& q)->Quaternion&
	{
		v += q.v;
		s += q.s;
		return *this;
	}

}