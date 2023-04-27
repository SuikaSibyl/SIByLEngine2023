module;
#include <cmath>
#include <cstdint>
export module SE.Math.Geometric:Quaternion;
import :Vector3;
import :Matrix3x3;
import :Matrix4x4;
import SE.Math.Misc;

namespace SIByL::Math
{
	export struct Quaternion
	{
		Quaternion() :v(0), s(1.f) {}
		Quaternion(float x, float y, float z, float w) :v(x, y, z), s(w) {}
		Quaternion(vec3 const& v, float s) :v(v), s(s) {}
		Quaternion(vec3 const& eulerAngle);
		Quaternion(mat3 const& m);
		Quaternion(mat4 const& m);

		auto toMat3() const noexcept -> mat3;
		auto toMat4() const noexcept -> mat4;

		auto lengthSquared() const -> float;
		auto length() const -> float;

		auto conjugate() noexcept -> Quaternion;
		auto reciprocal() noexcept -> Quaternion;

		auto operator/(float s) const-> Quaternion;
		auto operator+(Quaternion const& q2) const -> Quaternion;
		auto operator*(Quaternion const& q2) const -> Quaternion;
		auto operator+=(Quaternion const& q) -> Quaternion&;
		auto operator-() const -> Quaternion;

		union {
			struct {
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

	export auto operator*(float s, Quaternion const& q) -> Quaternion {
		return Quaternion{ s * q.x,s * q.y,s * q.z,s * q.w };
	}
	
	export auto operator*(Quaternion const& q, float s) -> Quaternion {
		return Quaternion{ s * q.x,s * q.y,s * q.z,s * q.w };
	}

	export auto operator-(Quaternion const& q1, Quaternion const& q2) -> Quaternion {
		return Quaternion{ q1.x - q2.x, q1.y - q2.y,q1.z - q2.z,q1.w - q2.w };
	}

	export inline auto slerp(float t, Quaternion const& q1, Quaternion const& q2) noexcept -> Quaternion
	{
		float cosTheta = dot(q1, q2);
		if (cosTheta > .9995f)
			return normalize((1 - t) * q1 + t * q2);
		else {
			float theta = std::acos(clamp(cosTheta, -1.f, 1.f));
			float thetap = theta * t;
			Quaternion qperp = normalize(q2 - q1 * cosTheta);
			return q1 * std::cos(thetap) + qperp * std::sin(thetap);
		}
	}

	Quaternion::Quaternion(vec3 const& eulerAngle) {
		Quaternion qx = Quaternion(vec3{ 1,0,0 }, eulerAngle.x);
		Quaternion qy = Quaternion(vec3{ 0,1,0 }, eulerAngle.y);
		Quaternion qz = Quaternion(vec3{ 0,0,1 }, eulerAngle.z);
		*this = qz * qy * qx;
		//vec3 c = Math::cos(eulerAngle * 0.5);
		//vec3 s = Math::sin(eulerAngle * 0.5);
		//this->w = c.x * c.y * c.z + s.x * s.y * s.z;
		//this->x = s.x * c.y * c.z - c.x * s.y * s.z;
		//this->y = c.x * s.y * c.z + s.x * c.y * s.z;
		//this->z = c.x * c.y * s.z - s.x * s.y * c.z;
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

	auto Quaternion::toMat3() const noexcept -> mat3
	{
		return mat3(
			1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w),
			2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w),
			2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y));
	}
	

	auto Quaternion::toMat4() const noexcept -> mat4
	{
		return mat4(
			1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w), 0,
			2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w), 0,
			2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y), 0,
			0, 0, 0, 1);
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
	
	auto Quaternion::operator-() const->Quaternion {
		return Quaternion{ -x,-y,-z,-w }; 
	}
}