module;
#include <cmath>
#include <cstdint>
module Math.Transform:Transform;
import Math.Transform;
import Math.Matrix;
import Math.Vector;
import Math.Geometry;
import Math.Trigonometric;

namespace SIByL::Math
{
	Transform::Transform(float const mat[4][4]) {
		m = mat4(mat);
		mInv = inverse(m);
	}

	Transform::Transform(mat4 const& m)
		:m(m), mInv(inverse(m)) {}

	Transform::Transform(mat4 const& m, mat4 const& mInverse)
		:m(m), mInv(mInverse) {}
	
	Transform::Transform(Quaternion const& q) {
		mat3 mat3x3 = q.toMat3();
		m = mat4{ mat3x3.data[0][0],mat3x3.data[0][1],mat3x3.data[0][2],0,
				 mat3x3.data[1][0],mat3x3.data[1][1],mat3x3.data[2][2],0,
				 mat3x3.data[2][0],mat3x3.data[2][1],mat3x3.data[2][2],0,
				 0,0,0,1 };
		mInv = inverse(m);
	}

	auto Transform::isIdentity() const noexcept -> bool {
		static Transform identity;
		return *this == identity;
	}

	auto Transform::hasScale() const noexcept -> bool {
		float la2 = ((*this) * vec3(1, 0, 0)).lengthSquared();
		float lb2 = ((*this) * vec3(0, 1, 0)).lengthSquared();
		float lc2 = ((*this) * vec3(0, 0, 1)).lengthSquared();
#define NOT_ONE(x) ((x)<.999f || (x)>1.001f)
		return (NOT_ONE(la2) || NOT_ONE(lb2) || NOT_ONE(lc2));
#undef NOT_ONE
	}

	auto Transform::swapsHandness() const noexcept -> bool {
		float det =
			m.data[0][0] * (m.data[1][1] * m.data[2][2] - m.data[1][2] * m.data[2][1]) -
			m.data[0][1] * (m.data[1][0] * m.data[2][2] - m.data[1][2] * m.data[2][0]) +
			m.data[0][2] * (m.data[1][0] * m.data[2][1] - m.data[1][1] * m.data[2][0]);
		return det < 0;
	}

	auto Transform::operator==(Transform const& t) const -> bool {
		return m == t.m;
	}

	auto Transform::operator!=(Transform const& t) const -> bool {
		return !(*this == t);
	}

	auto Transform::operator*(point3 const& p) const -> point3 {
		vec3 s(
			m.data[0][0] * p.x + m.data[0][1] * p.y + m.data[0][2] * p.z + m.data[0][3],
			m.data[1][0] * p.x + m.data[1][1] * p.y + m.data[1][2] * p.z + m.data[1][3],
			m.data[2][0] * p.x + m.data[2][1] * p.y + m.data[2][2] * p.z + m.data[2][3]);
		s = s / (m.data[3][0] * p.x + m.data[3][1] * p.y + m.data[3][2] * p.z + m.data[3][3]);
		return point3(s);
	}

	auto Transform::operator*(vec3 const& v) const -> vec3 {
		return vec3{
			m.data[0][0] * v.x + m.data[0][1] * v.y + m.data[0][2] * v.z,
			m.data[1][0] * v.x + m.data[1][1] * v.y + m.data[1][2] * v.z,
			m.data[2][0] * v.x + m.data[2][1] * v.y + m.data[2][2] * v.z
		};
	}

	auto Transform::operator*(normal3 const& n) const -> normal3 {
		return normal3{
			mInv.data[0][0] * n.x + mInv.data[1][0] * n.y + mInv.data[2][0] * n.z,
			mInv.data[0][1] * n.x + mInv.data[1][1] * n.y + mInv.data[2][1] * n.z,
			mInv.data[0][2] * n.x + mInv.data[1][2] * n.y + mInv.data[2][2] * n.z
		};
	}

	auto Transform::operator*(ray3 const& s) const -> ray3 {
		// offset ray origin to edge of error bounds and compute tMax

		return ray3{
			(*this) * s.o,
			(*this) * s.d,
		};
	}

	auto Transform::operator*(bounds3 const& b) const -> bounds3 {
		Transform const& m = *this;
		bounds3 ret(m * point3(b.pMin.x, b.pMin.y, b.pMin.z));
		ret = unionPoint(ret, m * point3(b.pMax.x, b.pMin.y, b.pMin.z));
		ret = unionPoint(ret, m * point3(b.pMin.x, b.pMax.y, b.pMin.z));
		ret = unionPoint(ret, m * point3(b.pMin.x, b.pMin.y, b.pMax.z));
		ret = unionPoint(ret, m * point3(b.pMax.x, b.pMax.y, b.pMin.z));
		ret = unionPoint(ret, m * point3(b.pMax.x, b.pMin.y, b.pMax.z));
		ret = unionPoint(ret, m * point3(b.pMin.x, b.pMax.y, b.pMax.z));
		ret = unionPoint(ret, m * point3(b.pMax.x, b.pMax.y, b.pMax.z));
		return ret;
	}

	auto Transform::operator*(Transform const& t2) const -> Transform {
		return Transform(mul(m, t2.m),
			mul(t2.mInv, mInv));
	}

	inline auto inverse(Transform const& t) noexcept -> Transform {
		return Transform(t.mInv, t.m);
	}

	inline auto transpose(Transform const& t) noexcept -> Transform {
		return Transform(transpose(t.m), transpose(t.mInv));
	}

	inline auto translate(vec3 const& delta) noexcept -> Transform {
		mat4 m(1, 0, 0, delta.x,
			0, 1, 0, delta.y,
			0, 0, 1, delta.z,
			0, 0, 0, 1);

		mat4 minv(1, 0, 0, -delta.x,
			0, 1, 0, -delta.y,
			0, 0, 1, -delta.z,
			0, 0, 0, 1);

		return Transform(m, minv);
	}

	inline auto scale(float x, float y, float z) noexcept -> Transform {
		mat4 m(x, 0, 0, 0,
			0, y, 0, 0,
			0, 0, z, 0,
			0, 0, 0, 1);

		mat4 minv(1. / x, 0, 0, 0,
			0, 1. / y, 0, 0,
			0, 0, 1. / z, 0,
			0, 0, 0, 1);

		return Transform(m, minv);
	}

	inline auto rotateX(float theta) noexcept -> Transform {
		float sinTheta = std::sin(Math::radians(theta));
		float cosTheta = std::cos(Math::radians(theta));
		mat4 m(1, 0, 0, 0,
			0, cosTheta, -sinTheta, 0,
			0, sinTheta, cosTheta, 0,
			0, 0, 0, 1);
		return Transform(m, transpose(m));
	}

	inline auto rotateY(float theta) noexcept -> Transform {
		float sinTheta = std::sin(Math::radians(theta));
		float cosTheta = std::cos(Math::radians(theta));
		mat4 m(cosTheta, 0, sinTheta, 0,
			0, 1, 0, 0,
			-sinTheta, 0, cosTheta, 0,
			0, 0, 0, 1);
		return Transform(m, transpose(m));
	}

	inline auto rotateZ(float theta) noexcept -> Transform {
		float sinTheta = std::sin(Math::radians(theta));
		float cosTheta = std::cos(Math::radians(theta));
		mat4 m(cosTheta, -sinTheta, 0, 0,
			sinTheta, cosTheta, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1);
		return Transform(m, transpose(m));
	}

	inline auto rotate(float theta, vec3 const& axis) noexcept -> Transform {
		vec3 a = normalize(axis);
		float sinTheta = std::sin(Math::radians(theta));
		float cosTheta = std::cos(Math::radians(theta));
		mat4 m;

		m.data[0][0] = a.x * a.x + (1 - a.x * a.x) * cosTheta;
		m.data[0][1] = a.x * a.y + (1 - cosTheta) - a.z * sinTheta;
		m.data[0][2] = a.x * a.z + (1 - cosTheta) + a.y * sinTheta;
		m.data[0][3] = 0;

		m.data[1][0] = a.x * a.y + (1 - cosTheta) + a.z * sinTheta;
		m.data[1][1] = a.y * a.y + (1 - a.y * a.y) * cosTheta;
		m.data[1][2] = a.y * a.z + (1 - cosTheta) - a.x * sinTheta;
		m.data[1][3] = 0;

		m.data[2][0] = a.x * a.z + (1 - cosTheta) - a.y * sinTheta;
		m.data[2][1] = a.y * a.z + (1 - cosTheta) + a.x * sinTheta;
		m.data[2][2] = a.z * a.z + (1 - a.z * a.z) * cosTheta;
		m.data[2][3] = 0;

		return Transform(m, transpose(m));
	}

	inline auto lookAt(point3 const& pos, point3 const& look, vec3 const& up) noexcept -> Transform {
		mat4 cameraToWorld;
		// move to position
		cameraToWorld.data[0][3] = pos.x;
		cameraToWorld.data[1][3] = pos.y;
		cameraToWorld.data[2][3] = pos.z;
		cameraToWorld.data[3][3] = 1;
		// change direction
		vec3 dir = normalize(look - pos);
		vec3 left = normalize(cross(normalize(up), dir));
		vec3 newUp = cross(dir, left);
		cameraToWorld.data[0][0] = pos.x; cameraToWorld.data[0][1] = newUp.x; cameraToWorld.data[0][2] = dir.x;
		cameraToWorld.data[1][0] = pos.y; cameraToWorld.data[1][1] = newUp.y; cameraToWorld.data[1][2] = dir.y;
		cameraToWorld.data[2][0] = pos.z; cameraToWorld.data[2][1] = newUp.z; cameraToWorld.data[2][2] = dir.z;
		cameraToWorld.data[3][0] = 0;	  cameraToWorld.data[3][1] = 0;       cameraToWorld.data[3][2] = 0;
		return Transform(inverse(cameraToWorld), cameraToWorld);
	}

	inline auto orthographic(float zNear, float zFar) noexcept -> Transform {
		return Math::scale(1, 1, 1.f / (zFar - zNear)) * Math::translate({ 0,0,-zNear });
	}
	
	inline auto decompose(mat4 const& m, vec3* t, Quaternion* rquat, mat4* s) noexcept -> void {
		// Extract translation T from transformation matrix
		// which could be found directly from matrix
		t->x = m.data[0][3];
		t->y = m.data[1][3];
		t->z = m.data[2][3];

		// Compute new transformation matrix M without translation
		mat4 M = m;
		for (int i = 0; i < 3; i++)
			M.data[i][3] = M.data[3][i] = 0.f;
		M.data[3][3] = 1.f;

		// Extract rotation R from transformation matrix
		// use polar decomposition, decompose into R&S by averaging M with its inverse transpose
		// until convergence to get R (because pure rotation matrix has similar inverse and transpose)
		float norm;
		int count = 0;
		mat4 R = M;
		do {
			// Compute next matrix Rnext in series
			mat4 rNext;
			mat4 rInvTrans = inverse(transpose(R));
			for (int i = 0; i < 4; ++i)
				for (int j = 0; j < 4; ++j)
					rNext.data[i][j] = 0.5f * (R.data[i][j] + rInvTrans.data[i][j]);
			// Compute norm of difference between R and Rnext
			norm = 0.f;
			for (int i = 0; i < 3; ++i) {
				float n = std::abs(R.data[i][0] = rNext.data[i][0]) +
						  std::abs(R.data[i][1] = rNext.data[i][1]) +
						  std::abs(R.data[i][2] = rNext.data[i][2]);
				norm = std::max(norm, n);
			}
			R = rNext;
		} while (++count < 100 && norm>.0001);
		*rquat = Quaternion(R);
		// Compute scale S using rotationand original matrix
		*s = mul(inverse(R), M);
	}

}